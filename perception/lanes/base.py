"""Common lane-detector interface and config-driven factory.

Every backend turns a BGR frame into an image-space lane binary and hands it to the shared
`LaneFitter` + `LaneSmoother`, so all backends return the same `LaneResult` (ARCHITECTURE.md 3.4).
Switch backends with `method:` in configs/lanes.yaml.

Two calling modes:
- Video (default): `detect(frame)` keeps temporal state (EMA smoothing, `every_n_frames` reuse).
  State is reset automatically when the frame size changes.
- Independent images (dataset eval): `detect(frame, stateless=True)` always runs the backend,
  skips smoothing and the result cache, and leaves the video state untouched. No `reset()` needed.
"""

from __future__ import annotations

import copy
import importlib
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import yaml

from perception.lanes.fitting import LaneFitter, ego_mask_from_polylines
from perception.lanes.smoothing import LaneSmoother
from perception.types import LaneResult

REPO_ROOT = Path(__file__).resolve().parents[2]

# method name -> (module, class). Imported lazily so heavy backends (onnxruntime for YOLOP) only
# load when selected.
_BACKENDS: dict[str, tuple[str, str]] = {
    "classical": ("perception.lanes.classical", "ClassicalLaneDetector"),
    "yolop": ("perception.lanes.yolop", "YolopLaneDetector"),
}


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def merge_config(base: dict, override: dict) -> dict:
    """Recursive dict merge: `override` wins; nested dicts merge, everything else is replaced."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_lane_config(path: str | Path) -> dict:
    """Read a configs/lanes*.yaml file.

    A file with a top-level `base: <path>` key is an override: it is merged on top of the base
    file (recursively, so only changed keys need to be listed). Relative paths resolve from the
    working directory first, then from the repo root.
    """
    with open(resolve_path(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = merge_config(load_lane_config(base), cfg)
    return cfg


def build_lane_detector(cfg: dict) -> LaneDetector:
    """Create the backend named by `cfg["method"]` (classical | yolop)."""
    method = cfg.get("method", "classical")
    if method not in _BACKENDS:
        raise ValueError(f"Unknown lane method {method!r}; expected one of {sorted(_BACKENDS)}")
    module_name, class_name = _BACKENDS[method]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(cfg)


class LaneDetector(ABC):
    """Base class: `detect(frame_bgr) -> LaneResult`, plus `reset()` between unrelated clips.

    `cfg` is the whole configs/lanes.yaml dict. The backend's own section is `cfg[self.name]`;
    its `every_n_frames` lets heavy backends run on every n-th frame and reuse the last result.
    """

    name: str = "base"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        section = cfg.get(self.name) or {}
        self.every_n = max(1, int(section.get("every_n_frames", 1)))
        self.fitter = LaneFitter(cfg["fitting"])
        self.smoother = LaneSmoother(cfg["smoothing"])
        self.make_ego_mask = bool(cfg.get("ego_mask", True))
        self.last_debug: dict = {}
        self.last_timing_ms: dict[str, float] = {}  # stage times of the last backend run
        self._frame_idx = 0
        self._last: LaneResult | None = None
        self._frame_hw: tuple[int, int] | None = None

    def detect(self, frame_bgr: np.ndarray, stateless: bool = False) -> LaneResult:
        """Lane boundaries for one HxWx3 uint8 BGR frame, in image coordinates.

        `stateless=True` treats the frame as an independent image: the backend always runs, no
        smoothing is applied, and the temporal state used by video calls is neither read nor
        changed. Use it for dataset evaluation.
        """
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError(
                f"expected HxWx3 uint8 BGR frame, got shape {frame_bgr.shape} {frame_bgr.dtype}"
            )
        if stateless:
            return self._infer(frame_bgr, smooth=False)
        hw = (int(frame_bgr.shape[0]), int(frame_bgr.shape[1]))
        if self._frame_hw is not None and hw != self._frame_hw:
            self.reset()  # new camera: cached result and smoother state belong to the old one
        self._frame_hw = hw
        run = self._last is None or self._frame_idx % self.every_n == 0
        self._frame_idx += 1
        if run:
            self._last = self._infer(frame_bgr, smooth=True)
        assert self._last is not None
        return self._last

    def reset(self) -> None:
        """Forget temporal state (call between unrelated clips)."""
        self._frame_idx = 0
        self._last = None
        self._frame_hw = None
        self.smoother.reset()
        self.last_debug = {}

    @abstractmethod
    def lane_binary(self, frame_bgr: np.ndarray) -> np.ndarray:
        """HxW bool mask of lane-marking pixels in image coordinates."""

    def _infer(self, frame_bgr: np.ndarray, smooth: bool = True) -> LaneResult:
        """Backend binary + shared fit. Fills `last_timing_ms` with `binary_ms` (whole backend
        stage) and `fit_ms`; backends may add finer keys (YOLOP: pre/model/post)."""
        self.last_timing_ms = {}
        t0 = time.perf_counter()
        binary = self.lane_binary(frame_bgr)
        t1 = time.perf_counter()
        res = self._lanes_from_binary(binary, smooth=smooth)
        self.last_timing_ms.update(
            {"binary_ms": 1e3 * (t1 - t0), "fit_ms": 1e3 * (time.perf_counter() - t1)}
        )
        return res

    def _lanes_from_binary(self, binary: np.ndarray, smooth: bool = True) -> LaneResult:
        h, w = binary.shape
        fit = self.fitter.fit(binary)
        if smooth:
            left, right, c_l, c_r = self.smoother.update(
                fit.left, fit.right, fit.conf_left, fit.conf_right, y_ref=h - 1.0, width=w
            )
        else:
            left, right = fit.left, fit.right
            c_l = float(fit.conf_left) if left is not None else 0.0
            c_r = float(fit.conf_right) if right is not None else 0.0
        mask = None
        if self.make_ego_mask and left is not None and right is not None:
            mask = ego_mask_from_polylines(left, right, (h, w), fit.y_top, fit.y_bottom)
        self.last_debug = {
            "binary": binary,
            "y_top": fit.y_top,
            "y_bottom": fit.y_bottom,
            **fit.debug,
        }
        return LaneResult(left, right, mask, float(0.5 * (c_l + c_r)))
