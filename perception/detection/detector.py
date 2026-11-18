"""Object detector with swappable backends (ARCHITECTURE.md section 3.2, PLAN.md M2.4).

    det = Detector(load_config("configs/yolo_detector.yaml"))
    detections = det.detect(frame_bgr)          # -> list[perception.types.Detection]

Backends, chosen by ``cfg["backend"]``:
- ``torch``: Ultralytics YOLO on a ``.pt`` file (Kaggle, offline video). Imports Ultralytics lazily,
  so this module stays importable without torch (laptop runtime, CI).
- ``onnx`` / ``tensorrt`` (M6): ONNX Runtime on the ONNX export, or a TensorRT engine built from
  it. They subclass ``ExportedBackend`` and only implement ``infer(blob) -> raw``; letterbox,
  decode, NMS and rescale are shared (``perception.detection.processing``), so all backends
  post-process the same way. ``cfg["weights"]`` may keep the ``.pt`` name: the file with the
  backend's suffix next to it is used (``backend_weights``). Neither imports torch.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from perception.detection.processing import (
    RawDetections,
    as_hw,
    decode_yolo,
    letterbox,
    scale_boxes,
    to_blob,
)
from perception.detection.runtime import TrtRunner, check_input_hw, ort_providers
from perception.types import Detection

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_KEYS = ("backend", "weights", "imgsz", "conf", "iou")
OPTIONAL_DEFAULTS: dict[str, Any] = {
    "max_det": 300,
    "device": None,
    "half": False,
    "classes": None,
    "agnostic_nms": False,  # Ultralytics default; configs/yolo_detector.yaml turns it on
    "names": None,
    "data_config": None,
    "onnx_providers": None,  # onnx backend: null = CUDA, then CPU
}
BACKEND_SUFFIX = {"torch": ".pt", "onnx": ".onnx", "tensorrt": ".engine"}


def resolve_path(path: str | Path) -> Path:
    """Use ``path`` as given if it exists or is absolute, else resolve it from the repo root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def load_config(path: str | Path, **overrides: Any) -> dict[str, Any]:
    cfg = yaml.safe_load(resolve_path(path).read_text())
    cfg.update(overrides)
    return cfg


def load_names(cfg: dict[str, Any]) -> dict[int, str] | None:
    """Class names from ``cfg["names"]`` (list or dict) or from the ``data_config`` YAML."""
    names = cfg.get("names")
    if names is None and cfg.get("data_config"):
        names = yaml.safe_load(resolve_path(cfg["data_config"]).read_text()).get("names")
    if names is None:
        return None
    if isinstance(names, list):
        return dict(enumerate(str(n) for n in names))
    return {int(k): str(v) for k, v in names.items()}


def _to_numpy(x: Any) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


class Backend(Protocol):
    names: dict[int, str] | None

    def __call__(self, frame: np.ndarray) -> RawDetections: ...


def result_to_raw(result: Any) -> RawDetections:
    """Ultralytics ``Results`` (one image) -> ``RawDetections`` in original-image pixels."""
    b = result.boxes
    return RawDetections(
        _to_numpy(b.xyxy).reshape(-1, 4).astype(np.float32),
        _to_numpy(b.conf).reshape(-1).astype(np.float32),
        _to_numpy(b.cls).reshape(-1).astype(np.int64),
    )


class TorchBackend:
    """Ultralytics YOLO ``.pt`` model. Pre/post-processing is Ultralytics' own predictor."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from ultralytics import YOLO  # lazy: torch is only present on Kaggle

        self.model = YOLO(str(resolve_path(cfg["weights"])))
        self.kwargs: dict[str, Any] = {
            "imgsz": cfg["imgsz"],
            "conf": cfg["conf"],
            "iou": cfg["iou"],
            "max_det": cfg["max_det"],
            "half": bool(cfg["half"]),
            "classes": cfg["classes"],
            "agnostic_nms": bool(cfg.get("agnostic_nms", OPTIONAL_DEFAULTS["agnostic_nms"])),
            "verbose": False,
        }
        if cfg["device"] is not None:
            self.kwargs["device"] = cfg["device"]
        self.names = {int(k): str(v) for k, v in self.model.names.items()}

    def __call__(self, frame: np.ndarray) -> RawDetections:
        return result_to_raw(self.model.predict(frame, **self.kwargs)[0])


class ExportedBackend:
    """Shared pipeline for exported models. Subclasses implement ``infer`` only."""

    names: dict[int, str] | None = None

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.imgsz = cfg["imgsz"]
        self.conf = float(cfg["conf"])
        self.iou = float(cfg["iou"])
        self.max_det = int(cfg["max_det"])
        self.classes = cfg["classes"]
        self.agnostic = bool(cfg.get("agnostic_nms", OPTIONAL_DEFAULTS["agnostic_nms"]))

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """1x3xHxW float32 RGB blob -> raw head output (1, 4 + nc, N)."""
        raise NotImplementedError

    def __call__(self, frame: np.ndarray) -> RawDetections:
        img, ratio, pad = letterbox(frame, self.imgsz)
        raw = decode_yolo(
            self.infer(to_blob(img)), self.conf, self.iou, self.max_det, self.classes, self.agnostic
        )
        return raw._replace(boxes=scale_boxes(raw.boxes, ratio, pad, frame.shape[:2]))


def backend_weights(weights: str | Path, backend: str) -> Path:
    """Model file for ``backend``: ``weights`` with its suffix switched to the backend's
    (``.pt`` / ``.onnx`` / ``.engine``), so one ``weights: models/yolo11s-bdd.pt`` entry serves
    every backend. Paths with another suffix are used as given."""
    p = resolve_path(weights)
    want = BACKEND_SUFFIX[backend]
    if p.suffix.lower() in BACKEND_SUFFIX.values() and p.suffix.lower() != want:
        p = p.with_suffix(want)
    return p


def names_from_metadata(meta: dict[str, str]) -> dict[int, str] | None:
    """Class names from Ultralytics ONNX metadata (``names: "{0: 'car', ...}"``), else None."""
    text = meta.get("names")
    if not text:
        return None
    try:
        names = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else None


class OnnxBackend(ExportedBackend):
    """ONNX Runtime on the Ultralytics ONNX export (``deploy/export.py``, static 640x640).

    Providers come from ``cfg["onnx_providers"]`` (default CUDA, then CPU; never the TensorRT
    EP, see ``perception.detection.runtime.ort_providers``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        path = backend_weights(cfg["weights"], "onnx")
        if not path.exists():
            raise FileNotFoundError(f"ONNX detector not found at {path} (deploy/export.py)")
        import onnxruntime as ort  # lazy: CI and the torch path do not need it

        providers = ort_providers(cfg.get("onnx_providers"), ort.get_available_providers())
        self.session = ort.InferenceSession(str(path), providers=providers)
        inp = self.session.get_inputs()[0]
        check_input_hw(inp.shape, as_hw(self.imgsz), str(path))
        self.input_name = inp.name
        self.names = names_from_metadata(self.session.get_modelmeta().custom_metadata_map)
        self.path = path

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: blob})[0]


class TensorRTBackend(ExportedBackend):
    """TensorRT engine built from the ONNX export (``deploy/build_local_engine.py``).

    No torch: ``perception.detection.runtime.TrtRunner`` uses tensorrt + cuda-python.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        path = backend_weights(cfg["weights"], "tensorrt")
        if not path.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found at {path}; build it from the ONNX file on this GPU "
                "with deploy/build_local_engine.py"
            )
        self.runner = TrtRunner(path)
        if len(self.runner.inputs) != 1:
            raise ValueError(f"{path}: expected one input, got {self.runner.input_names}")
        check_input_hw(self.runner.inputs[0].shape, as_hw(self.imgsz), str(path))
        self.path = path

    @property
    def providers(self) -> list[str]:
        return ["TensorRT"]

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self.runner(blob)[0]


BACKENDS: dict[str, type] = {
    "torch": TorchBackend,
    "onnx": OnnxBackend,
    "tensorrt": TensorRTBackend,
}


class Detector:
    """``Detector(cfg).detect(frame) -> list[Detection]``. Frames are HxWx3 BGR uint8."""

    def __init__(self, cfg: dict[str, Any], backend: Backend | None = None) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in cfg]
        if missing:
            raise KeyError(f"detector config is missing {missing}")
        self.cfg = {**OPTIONAL_DEFAULTS, **cfg}
        if backend is None:
            kind = self.cfg["backend"]
            if kind not in BACKENDS:
                raise ValueError(f"unknown detector backend {kind!r}; choose from {list(BACKENDS)}")
            backend = BACKENDS[kind](self.cfg)
        self.backend = backend
        cfg_names = load_names(self.cfg)
        model_names = getattr(backend, "names", None)
        if cfg_names and model_names and cfg_names != model_names:
            log.warning(
                "class names in %s %s differ from the model's own names %s; using the config's",
                self.cfg.get("data_config") or "cfg['names']",
                cfg_names,
                model_names,
            )
        self.names: dict[int, str] = cfg_names or model_names or {}
        self.conf = float(self.cfg["conf"])

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got shape {frame.shape}")
        raw = self.backend(frame)
        return [
            Detection(
                np.asarray(box, np.float32).copy(),
                float(score),
                int(c),
                self.names.get(int(c), str(int(c))),
            )
            for box, score, c in zip(raw.boxes, raw.scores, raw.cls, strict=True)
            if score >= self.conf
        ]
