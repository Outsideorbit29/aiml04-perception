"""YOLOP lane backend (M3.2): pretrained YOLOP ONNX lane-line head -> shared fitting -> LaneResult.

Uses the ONNX export shipped by hustvl/YOLOP (`weights/yolop-640-640.onnx`) through
onnxruntime, so the same code runs on Kaggle (CUDA provider) and later on the laptop without
PyTorch. Steps per frame:
1. Letterbox the RGB frame into the fixed model input (`yolop.input_size`), ImageNet-normalise.
2. Run the session, take the `lane_line_seg` output (1, 2, H_in, W_in), crop the padding.
3. Lane probability = channel 1 (softmax over the two channels; for YOLOP's per-channel sigmoid
   outputs a 0.5 threshold on it equals the argmax used by the YOLOP demo). Computed as the
   equivalent threshold on the margin ll[1] - ll[0] (no exp at full resolution).
4. Resize the margin to the frame, threshold it (`yolop.lane_threshold` as a probability), and
   hand the binary to the same `LaneFitter`/`LaneSmoother` as the classical backend.

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


def lane_margin_threshold(threshold: float) -> float:
    """Probability threshold t on softmax(ll)[1] -> threshold on the margin ll[1] - ll[0].

    softmax(ll)[1] >= t  <=>  ll[1] - ll[0] >= log(t / (1 - t)), so the post-processing can skip
    the exp/sum over two full-resolution channels. t = 0.5 gives 0 (YOLOP's argmax).
    """
    t = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
    return float(np.log(t / (1.0 - t)))


class YolopLaneDetector(LaneDetector):
    """YOLOP lane-line segmentation -> lane binary -> shared warp/fit/smooth stages."""

    name = "yolop"

    def __init__(self, cfg: dict, session: Any | None = None) -> None:
        super().__init__(cfg)
        y = cfg["yolop"]
        self.input_wh = (int(y["input_size"][0]), int(y["input_size"][1]))
        self.threshold = float(y["lane_threshold"])
        self.margin_threshold = lane_margin_threshold(self.threshold)
        mean = np.asarray(y["mean"], dtype=np.float32)
        std = np.asarray(y["std"], dtype=np.float32)
        self.mean, self.std = mean, std
        # x = (rgb / 255 - mean) / std = rgb * scale + offset, one fused float32 pass per channel
        self._scale = (1.0 / (255.0 * std)).astype(np.float32)
        self._offset = (-mean / std).astype(np.float32)
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
        in_w, in_h = self.input_wh
        self._canvas = np.empty((in_h, in_w, 3), dtype=np.uint8)
        self._blob = np.empty((1, 3, in_h, in_w), dtype=np.float32)  # reused, C-contiguous

    @property
    def providers(self) -> list[str]:
        get = getattr(self.session, "get_providers", None)
        return list(get()) if get else []

    def preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """BGR frame -> (1, 3, H_in, W_in) float32 tensor and the (new_w, new_h, px, py) box.

        The returned tensor is an internal buffer reused by the next call.
        """
        h, w = frame_bgr.shape[:2]
        _, new_w, new_h, px, py = letterbox_params((h, w), self.input_wh, self.allow_upscale)
        canvas = self._canvas
        canvas[:] = self.pad_value
        small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas[py : py + new_h, px : px + new_w] = small
        # BGR canvas -> RGB planes of the blob, normalised in float32
        for c_out, c_in in enumerate((2, 1, 0)):
            np.multiply(canvas[:, :, c_in], self._scale[c_out], out=self._blob[0, c_out])
            self._blob[0, c_out] += self._offset[c_out]
        return self._blob, (new_w, new_h, px, py)

    def lane_margin(self, frame_bgr: np.ndarray) -> np.ndarray:
        """HxW float32 lane margin ll[1] - ll[0] at frame resolution (see lane_margin_threshold).
        Fills `last_timing_ms` with pre_ms, model_ms (session.run) and post_ms."""
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        x, (new_w, new_h, px, py) = self.preprocess(frame_bgr)
        t1 = time.perf_counter()
        (ll,) = self.session.run([self.output_name], {self.input_name: x})
        t2 = time.perf_counter()
        ll = np.asarray(ll, dtype=np.float32)[0, :, py : py + new_h, px : px + new_w]
        margin = cv2.resize(ll[1] - ll[0], (w, h), interpolation=cv2.INTER_LINEAR)
        t3 = time.perf_counter()
        self.last_timing_ms.update(
            {"pre_ms": 1e3 * (t1 - t0), "model_ms": 1e3 * (t2 - t1), "post_ms": 1e3 * (t3 - t2)}
        )
        return margin

    def lane_prob(self, frame_bgr: np.ndarray) -> np.ndarray:
        """HxW float32 lane probability at frame resolution (visualisation/debug only)."""
        m = self.lane_margin(frame_bgr)
        return (1.0 / (1.0 + np.exp(-m))).astype(np.float32)

    def lane_binary(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.lane_margin(frame_bgr) >= self.margin_threshold


def benchmark_session(session: Any, input_hw: tuple[int, int], runs: int, warmup: int) -> dict:
    """Model-only latency of an onnxruntime session on a fixed zero tensor (Kaggle eval only).

    `run_ms`: session.run with host input/output (includes the H2D/D2H copies).
    `iobind_ms`: same with input and all outputs bound on the CUDA device (pure GPU compute plus
    launch overhead); None if the CUDA provider or I/O binding is unavailable.
    """
    import onnxruntime as ort  # only reached on Kaggle with a real session

    h, w = input_hw
    x = np.zeros((1, 3, h, w), dtype=np.float32)
    name = session.get_inputs()[0].name
    outs = [o.name for o in session.get_outputs()]

    def timed(fn) -> list[float]:
        for _ in range(warmup):
            fn()
        ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            ms.append(1e3 * (time.perf_counter() - t0))
        return ms

    res: dict[str, Any] = {"runs": runs}
    run_ms = timed(lambda: session.run(outs, {name: x}))
    res["run_ms"] = float(np.median(run_ms))
    res["iobind_ms"] = None
    if "CUDAExecutionProvider" in session.get_providers():
        try:
            xv = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
            binding = session.io_binding()
            binding.bind_ortvalue_input(name, xv)
            for o in outs:
                binding.bind_output(o, "cuda", 0)

            def run_bound() -> None:
                session.run_with_iobinding(binding)
                binding.synchronize_outputs()

            res["iobind_ms"] = float(np.median(timed(run_bound)))
        except Exception as exc:  # report, never fail the eval on the micro-benchmark
            res["iobind_error"] = f"{type(exc).__name__}: {exc}"
    return res
