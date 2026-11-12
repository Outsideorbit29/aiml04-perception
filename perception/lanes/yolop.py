"""YOLOP lane backend (M3.2): pretrained YOLOP ONNX lane-line head -> shared fitting -> LaneResult.

Uses the ONNX export shipped by hustvl/YOLOP (`weights/yolop-640-640.onnx`) through
onnxruntime, so the same code runs on Kaggle (CUDA provider) and later on the laptop without
PyTorch. Steps per frame:
1. Letterbox the RGB frame into the fixed model input (`yolop.input_size`), ImageNet-normalise.
2. Run the session, take the `lane_line_seg` output (1, 2, H_in, W_in), crop the padding.
3. Lane probability = channel 1 (softmax over the two channels; for YOLOP's per-channel sigmoid
   outputs a 0.5 threshold on it equals the argmax used by the YOLOP demo).
4. Resize to the frame, threshold with `yolop.lane_threshold`, and hand the binary to the same
   `LaneFitter`/`LaneSmoother` as the classical backend.

The drivable-area and detection heads are ignored: detection comes from perception.detection
and the ego lane is the region between the fitted boundaries for every backend.

Weights are never downloaded here. `ensure_weights` is called only by the Kaggle eval job
(scripts/eval_lanes.py); the detector raises if `yolop.weights` does not exist.
"""

from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perception.lanes.base import LaneDetector, resolve_path


def ensure_weights(ycfg: dict) -> Path:
    """Return `ycfg["weights"]`, downloading it from `ycfg["weights_url"]` if missing (Kaggle)."""
    path = resolve_path(ycfg["weights"])
    if path.exists():
        return path
    url = ycfg.get("weights_url")
    if not url:
        raise FileNotFoundError(f"YOLOP weights not found at {path} and no weights_url set")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"downloading YOLOP weights {url} -> {path}", flush=True)
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL from config
    min_bytes = int(ycfg.get("min_weight_bytes", 0))
    if tmp.stat().st_size < min_bytes:
        raise RuntimeError(
            f"downloaded {tmp.stat().st_size} bytes from {url}; expected >= {min_bytes} "
            "(Git LFS pointer or HTML page?)"
        )
    expected = ycfg.get("sha256")
    if expected and file_sha256(tmp) != expected:
        raise RuntimeError(f"sha256 mismatch for {url}")
    tmp.replace(path)
    return path


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_session(weights: str | Path, providers: list[str]) -> Any:
    """onnxruntime InferenceSession; onnxruntime is imported here so tests can avoid it."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    chosen = [p for p in providers if p in available] or ["CPUExecutionProvider"]
    return ort.InferenceSession(str(weights), providers=chosen)


def letterbox_params(
    hw: tuple[int, int], in_wh: tuple[int, int], allow_upscale: bool
) -> tuple[float, int, int, int, int]:
    """Scale and padding that fit an HxW frame into the model input keeping the aspect ratio.

    Returns (r, new_w, new_h, pad_x, pad_y): the frame is resized to new_w x new_h and pasted at
    (pad_x, pad_y) in the in_w x in_h canvas.
    """
    h, w = hw
    in_w, in_h = in_wh
    r = min(in_w / w, in_h / h)
    if not allow_upscale:
        r = min(r, 1.0)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    return r, new_w, new_h, (in_w - new_w) // 2, (in_h - new_h) // 2


def lane_probability(ll: np.ndarray) -> np.ndarray:
    """(2, H, W) lane-line output -> (H, W) float32 probability of the lane class."""
    ll = np.asarray(ll, dtype=np.float32)
    e = np.exp(ll - ll.max(axis=0, keepdims=True))
    return e[1] / e.sum(axis=0)


class YolopLaneDetector(LaneDetector):
    """YOLOP lane-line segmentation -> lane binary -> shared warp/fit/smooth stages."""

    name = "yolop"

    def __init__(self, cfg: dict, session: Any | None = None) -> None:
        super().__init__(cfg)
        y = cfg["yolop"]
        self.input_wh = (int(y["input_size"][0]), int(y["input_size"][1]))
        self.threshold = float(y["lane_threshold"])
        self.mean = np.asarray(y["mean"], dtype=np.float32)
        self.std = np.asarray(y["std"], dtype=np.float32)
        self.pad_value = int(y["pad_value"])
        self.allow_upscale = bool(y["allow_upscale"])
        self.output_name = str(y["lane_output"])
        if session is None:
            weights = resolve_path(y["weights"]) if y.get("weights") else None
            if weights is None or not weights.exists():
                raise FileNotFoundError(
                    f"YOLOP weights not found: {y.get('weights')!r}. They are fetched on Kaggle "
                    "by scripts/eval_lanes.py (ensure_weights); locally put the ONNX file there."
                )
            session = make_session(weights, list(y["providers"]))
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.last_timing_ms: dict[str, float] = {}

    @property
    def providers(self) -> list[str]:
        get = getattr(self.session, "get_providers", None)
        return list(get()) if get else []

    def preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """BGR frame -> (1, 3, H_in, W_in) float32 tensor and the (new_w, new_h, px, py) box."""
        h, w = frame_bgr.shape[:2]
        _, new_w, new_h, px, py = letterbox_params((h, w), self.input_wh, self.allow_upscale)
        in_w, in_h = self.input_wh
        canvas = np.full((in_h, in_w, 3), self.pad_value, dtype=np.uint8)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        canvas[py : py + new_h, px : px + new_w] = cv2.resize(
            rgb, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        x = (canvas.astype(np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)[None], (new_w, new_h, px, py)

    def lane_prob(self, frame_bgr: np.ndarray) -> np.ndarray:
        """HxW float32 lane probability at frame resolution."""
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        x, (new_w, new_h, px, py) = self.preprocess(frame_bgr)
        t1 = time.perf_counter()
        (ll,) = self.session.run([self.output_name], {self.input_name: x})
        t2 = time.perf_counter()
        ll = np.asarray(ll)[0, :, py : py + new_h, px : px + new_w]
        prob = cv2.resize(lane_probability(ll), (w, h), interpolation=cv2.INTER_LINEAR)
        t3 = time.perf_counter()
        self.last_timing_ms = {
            "pre_ms": 1e3 * (t1 - t0),
            "model_ms": 1e3 * (t2 - t1),
            "post_ms": 1e3 * (t3 - t2),
        }
        return prob

    def lane_binary(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.lane_prob(frame_bgr) >= self.threshold
