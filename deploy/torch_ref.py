"""PyTorch FP32 reference models and ONNX export helpers (Kaggle only; imports torch lazily).

The references run on the same preprocessed tensors as the exported models:
- ``TorchYoloBackend``: the fused Ultralytics ``DetectionModel`` behind ``ExportedBackend``, so
  PyTorch, ONNX Runtime and TensorRT share letterbox (static 640x640), decode, NMS and rescale.
  (The ``torch`` backend of ``Detector`` uses Ultralytics' own predictor with a minimal-padding
  letterbox, which would mix an input-shape change into the accuracy delta.)
- MiDaS-small: ``perception.depth.estimator.TorchHubBackend`` as used by the M4 evals.
- ``TorchYolopSession``: hustvl/YOLOP ``End-to-end.pth`` behind the onnxruntime session API that
  ``YolopLaneDetector`` uses. These are the weights of the shipped ``yolop-640-640.onnx``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from perception.detection.detector import ExportedBackend

YOLOP_OUTPUTS = ("det_out", "drive_area_seg", "lane_line_seg")


def cuda_device() -> Any:
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def first_tensor(out: Any) -> Any:
    """Ultralytics / YOLOP heads return ``(inference_out, train_out)`` in eval mode."""
    return out[0] if isinstance(out, (tuple, list)) else out


def load_yolo_module(weights: str | Path) -> tuple[Any, dict[int, str]]:
    """Fused FP32 ``DetectionModel`` on CUDA, and its class names."""
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    names = {int(k): str(v) for k, v in yolo.names.items()}
    model = yolo.model
    if hasattr(model, "fuse"):
        model = model.fuse(verbose=False)
    model = model.float().eval().to(cuda_device())
    for p in model.parameters():
        p.requires_grad_(False)
    return model, names


class TorchYoloBackend(ExportedBackend):
    """PyTorch FP32 YOLO with the shared exported-model pre/post-processing."""

    def __init__(self, cfg: dict[str, Any], weights: str | Path) -> None:
        import torch

        super().__init__(cfg)
        self.torch = torch
        self.model, self.names = load_yolo_module(weights)
        self.device = next(self.model.parameters()).device
        self.providers = [f"torch-{self.device.type}"]

    def infer(self, blob: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            y = first_tensor(self.model(torch.from_numpy(blob).to(self.device)))
            return y.float().cpu().numpy()


def load_yolop(hub_repo: str, hub_model: str) -> Any:
    """hustvl/YOLOP via torch.hub on CUDA, FP32, eval.

    Its checkpoint holds more than tensors, so torch>=2.6 (``weights_only=True`` by default)
    refuses it; the official repo is trusted here, so ``weights_only=False`` is set for this
    load only.
    """
    import torch

    orig = torch.load

    def load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    torch.load = load
    try:
        model = torch.hub.load(hub_repo, hub_model, pretrained=True, trust_repo=True)
    finally:
        torch.load = orig
    model = model.float().eval().to(cuda_device())
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class TorchYolopSession:
    """onnxruntime-like session on the PyTorch YOLOP model (outputs as in the ONNX file)."""

    def __init__(self, model: Any, input_name: str = "images",
                 output_names: tuple[str, ...] = YOLOP_OUTPUTS) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.device = next(model.parameters()).device
        self.input_name = input_name
        self.output_names = tuple(output_names)

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=self.input_name, shape=[1, 3, 640, 640])]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=n, shape=None) for n in self.output_names]

    def get_providers(self) -> list[str]:
        return [f"torch-{self.device.type}"]

    def run(self, names: list[str] | None, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(feed[self.input_name]).to(self.device)
            det, da, ll = self.model(x)
            outs = dict(zip(self.output_names, (first_tensor(det), da, ll), strict=True))
            return [outs[n].float().cpu().numpy() for n in (names or self.output_names)]


def export_onnx(
    model: Any,
    dummy: Any,
    path: str | Path,
    opset: int,
    input_names: list[str],
    output_names: list[str],
) -> Path:
    """TorchScript-based ``torch.onnx.export`` with static shapes, then ``onnx.checker``.

    torch>=2.9 defaults to the dynamo exporter (opset 18+); ``dynamo=False`` keeps opset 17.
    """
    import onnx
    import torch

    kwargs: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(False), torch.no_grad():
        torch.onnx.export(
            model, (dummy,), str(path), opset_version=int(opset), input_names=input_names,
            output_names=output_names, do_constant_folding=True, **kwargs,
        )
    onnx.checker.check_model(onnx.load(str(path)))
    return path
