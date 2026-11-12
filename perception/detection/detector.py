"""Object detector with swappable backends (ARCHITECTURE.md section 3.2, PLAN.md M2.4).

    det = Detector(load_config("configs/yolo_detector.yaml"))
    detections = det.detect(frame_bgr)          # -> list[perception.types.Detection]

Backends, chosen by ``cfg["backend"]``:
- ``torch``: Ultralytics YOLO on a ``.pt`` file (Kaggle, offline video). Imports Ultralytics lazily,
  so this module stays importable without torch (laptop runtime, CI).
- ``onnx`` / ``tensorrt``: filled in M6 by edge-optimizer. They subclass ``ExportedBackend`` and
  only implement ``infer(blob) -> raw``; letterbox, decode, NMS and rescale are shared
  (``perception.detection.processing``), so all backends post-process the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from perception.detection.processing import (
    RawDetections,
    decode_yolo,
    letterbox,
    scale_boxes,
    to_blob,
)
from perception.types import Detection

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_KEYS = ("backend", "weights", "imgsz", "conf", "iou")
OPTIONAL_DEFAULTS: dict[str, Any] = {
    "max_det": 300,
    "device": None,
    "half": False,
    "classes": None,
    "names": None,
    "data_config": None,
}


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

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """1x3xHxW float32 RGB blob -> raw head output (1, 4 + nc, N)."""
        raise NotImplementedError

    def __call__(self, frame: np.ndarray) -> RawDetections:
        img, ratio, pad = letterbox(frame, self.imgsz)
        raw = decode_yolo(self.infer(to_blob(img)), self.conf, self.iou, self.max_det, self.classes)
        return raw._replace(boxes=scale_boxes(raw.boxes, ratio, pad, frame.shape[:2]))


class OnnxBackend(ExportedBackend):
    """ONNX Runtime backend. Filled in M6 (edge-optimizer)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "onnx backend is added in M6: open an onnxruntime.InferenceSession on cfg['weights'] "
            "here and implement infer(blob)"
        )


class TensorRTBackend(ExportedBackend):
    """TensorRT engine backend. Filled in M6 (edge-optimizer)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "tensorrt backend is added in M6: deserialize the engine at cfg['weights'] here and "
            "implement infer(blob)"
        )


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
        self.names: dict[int, str] = load_names(self.cfg) or getattr(backend, "names", None) or {}
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
