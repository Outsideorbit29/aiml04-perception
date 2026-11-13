"""Monocular relative depth with MiDaS-small (live) or DPT-Hybrid (offline only) (M4.1).

    est = DepthEstimator(load_depth_config("configs/depth.yaml"))   # onnx backend (laptop)
    est = DepthEstimator(load_depth_config("configs/depth.yaml", backend="torch"))  # Kaggle
    inv = est.estimate(frame_bgr)   # HxW float32 RELATIVE inverse depth (bigger = closer)

The output is NOT metric: it is inverse depth up to an unknown per-frame scale and shift.
Metric distances come only after ``perception.depth.alignment.ScaleAligner`` (see
perception/fusion/fusion.py).

Backends (``estimator.backend``):
- ``torch``: ``torch.hub`` model (intel-isl/MiDaS). Downloads code + weights on first use, so it
  runs on Kaggle only. torch is imported lazily inside the backend.
- ``onnx``: onnxruntime session on an exported model (export is M6, deploy/). Never imports
  torch, so it runs in the laptop runtime.
Pre/post-processing (resize to ``input_hw``, normalisation, resize back to the frame) is shared
numpy/OpenCV code, so both backends see identical tensors.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import yaml

from perception.depth.camera import resolve_path

ESTIMATOR_KEYS = ("model", "backend")
MODEL_KEYS = ("input_hw", "mean", "std", "multiple_of")
_INTERP = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}


def load_depth_config(path: str | Path, **estimator_overrides: Any) -> dict[str, Any]:
    """Read configs/depth*.yaml. A top-level ``base: <path>`` is loaded first and overridden
    (recursively). ``estimator_overrides`` replace keys of the ``estimator`` section."""
    cfg = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = merge_config(load_depth_config(base), cfg)
    if estimator_overrides:
        cfg["estimator"] = {**cfg.get("estimator", {}), **estimator_overrides}
    return cfg


def merge_config(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = (
            merge_config(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


class DepthBackend(Protocol):
    def __call__(self, blob: np.ndarray) -> np.ndarray: ...


def _trust_hub_repos(torch: Any, repos: list[str]) -> None:
    """Pre-trust repos that a hubconf loads internally without ``trust_repo``.

    torch.hub otherwise asks on stdin, which fails in a non-interactive Kaggle job. torch keeps
    the list as ``owner_repo`` lines in ``<hub dir>/trusted_list``.
    """
    if not repos:
        return
    hub_dir = Path(torch.hub.get_dir())
    hub_dir.mkdir(parents=True, exist_ok=True)
    path = hub_dir / "trusted_list"
    known = set(path.read_text().split()) if path.exists() else set()
    new = [r.replace("/", "_") for r in repos if r.replace("/", "_") not in known]
    if new:
        with path.open("a") as f:
            f.write("".join(f"{r}\n" for r in new))


class TorchHubBackend:
    """MiDaS / DPT via ``torch.hub.load(hub_repo, hub_model)``. Kaggle only."""

    def __init__(self, mcfg: dict[str, Any], ecfg: dict[str, Any]) -> None:
        import torch  # lazy: not installed in the laptop runtime or CI

        self.torch = torch
        dev = ecfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.half = bool(ecfg.get("half", False)) and self.device.type == "cuda"
        _trust_hub_repos(torch, mcfg.get("trusted_hub_repos", []))
        model = torch.hub.load(mcfg["hub_repo"], mcfg["hub_model"], trust_repo=True)
        model = model.to(self.device).eval()
        self.model = model.half() if self.half else model

    def __call__(self, blob: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(blob).to(self.device)
            x = x.half() if self.half else x
            out = self.model(x)
            return out.float().cpu().numpy()


class OnnxBackend:
    """onnxruntime session on ``models.<name>.onnx`` (exported in M6). No torch import."""

    def __init__(self, mcfg: dict[str, Any], ecfg: dict[str, Any]) -> None:
        import onnxruntime as ort  # lazy: only needed when this backend is selected

        path = resolve_path(mcfg["onnx"])
        if not path.exists():
            raise FileNotFoundError(f"ONNX depth model not found at {path} (exported in M6)")
        providers = ecfg.get("onnx_providers") or ort.get_available_providers()
        self.session = ort.InferenceSession(str(path), providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, blob: np.ndarray) -> np.ndarray:
        return np.asarray(self.session.run(None, {self.input_name: blob})[0])


BACKENDS: dict[str, type] = {"torch": TorchHubBackend, "onnx": OnnxBackend}


def preprocess(frame_bgr: np.ndarray, mcfg: dict[str, Any]) -> np.ndarray:
    """HxWx3 BGR uint8 -> 1x3xhxw float32 RGB, resized to ``input_hw`` and normalised."""
    h, w = (int(v) for v in mcfg["input_hw"])
    interp = _INTERP[mcfg.get("resize_interp", "cubic")]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (w, h), interpolation=interp).astype(np.float32) / 255.0
    img = (img - np.asarray(mcfg["mean"], np.float32)) / np.asarray(mcfg["std"], np.float32)
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)


def postprocess(raw: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Network output (1, h, w) or (1, 1, h, w) -> HxW float32 at the frame size."""
    out = np.asarray(raw, dtype=np.float32).squeeze()
    if out.ndim != 2:
        raise ValueError(f"expected a single depth map, got output shape {np.shape(raw)}")
    return cv2.resize(out, (int(hw[1]), int(hw[0])), interpolation=cv2.INTER_LINEAR)


def sample_box(inv_depth: np.ndarray, bbox_xyxy: np.ndarray, scfg: dict[str, Any]) -> float:
    """Robust MiDaS value of one object: percentile ``scfg["percentile"]`` of the box region
    ``rows_frac`` x ``cols_frac`` (the lower, central part avoids background). NaN if empty."""
    h, w = inv_depth.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    r0, r1 = (float(v) for v in scfg["rows_frac"])
    c0, c1 = (float(v) for v in scfg["cols_frac"])
    ya, yb = y1 + r0 * (y2 - y1), y1 + r1 * (y2 - y1)
    xa, xb = x1 + c0 * (x2 - x1), x1 + c1 * (x2 - x1)
    ia, ib = max(0, int(np.floor(ya))), min(h, int(np.ceil(yb)))
    ja, jb = max(0, int(np.floor(xa))), min(w, int(np.ceil(xb)))
    if ib <= ia or jb <= ja:
        return float("nan")
    patch = inv_depth[ia:ib, ja:jb]
    patch = patch[np.isfinite(patch)]
    return float(np.percentile(patch, float(scfg["percentile"]))) if patch.size else float("nan")


class DepthEstimator:
    """``DepthEstimator(cfg).estimate(frame) -> HxW relative inverse depth``.

    ``cfg`` is the whole configs/depth.yaml dict: ``cfg["estimator"]`` picks the model and
    backend, ``cfg["models"][name]`` holds the model's input size and normalisation.
    ``maybe_estimate`` honours ``estimator.every_n_frames`` for the live path (None on skipped
    frames; a stale map would put moved objects at the wrong depth, so it is never reused).
    """

    def __init__(self, cfg: dict[str, Any], backend: DepthBackend | None = None) -> None:
        ecfg = cfg["estimator"]
        missing = [k for k in ESTIMATOR_KEYS if k not in ecfg]
        if missing:
            raise KeyError(f"depth estimator config is missing {missing}")
        self.name = str(ecfg["model"])
        if self.name not in cfg.get("models", {}):
            raise KeyError(f"depth model {self.name!r} not in config models")
        self.mcfg = cfg["models"][self.name]
        missing = [k for k in MODEL_KEYS if k not in self.mcfg]
        if missing:
            raise KeyError(f"depth model {self.name!r} config is missing {missing}")
        mult = int(self.mcfg["multiple_of"])
        if any(int(v) % mult for v in self.mcfg["input_hw"]):
            raise ValueError(
                f"{self.name}: input_hw {self.mcfg['input_hw']} not a multiple of {mult}"
            )
        if backend is None:
            kind = ecfg["backend"]
            if kind not in BACKENDS:
                raise ValueError(f"unknown depth backend {kind!r}; choose from {list(BACKENDS)}")
            backend = BACKENDS[kind](self.mcfg, ecfg)
        self.backend = backend
        self.every_n = max(1, int(ecfg.get("every_n_frames", 1)))
        self._frame_idx = 0
        self.last_timing_ms: dict[str, float] = {}

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got shape {frame_bgr.shape}")
        t0 = time.perf_counter()
        blob = preprocess(frame_bgr, self.mcfg)
        t1 = time.perf_counter()
        raw = self.backend(blob)
        t2 = time.perf_counter()
        out = postprocess(raw, frame_bgr.shape[:2])
        t3 = time.perf_counter()
        self.last_timing_ms = {
            "pre_ms": 1e3 * (t1 - t0),
            "model_ms": 1e3 * (t2 - t1),
            "post_ms": 1e3 * (t3 - t2),
            "total_ms": 1e3 * (t3 - t0),
        }
        return out

    def maybe_estimate(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """``estimate`` on every ``every_n_frames``-th call, None otherwise."""
        run = self._frame_idx % self.every_n == 0
        self._frame_idx += 1
        return self.estimate(frame_bgr) if run else None

    def reset(self) -> None:
        self._frame_idx = 0
