"""Shared helpers for deploy/export.py and deploy/benchmark.py (Kaggle side of M6).

Nothing here imports torch at module level; preprocessing is the runtime's own code
(``perception.detection.processing``, ``perception.depth.estimator.preprocess``,
``YolopLaneDetector.preprocess``), so exported models are checked on exactly the tensors the
live pipeline feeds them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import yaml

from perception.depth.estimator import load_depth_config, merge_config
from perception.detection.detector import load_config as load_detector_config
from perception.detection.detector import resolve_path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("yolo", "midas_small", "yolop")
BACKENDS = ("torch", "onnx", "tensorrt")
PACKAGES = (
    "torch", "onnxruntime-gpu", "onnxruntime", "onnx", "tensorrt", "tensorrt-cu12",
    "cuda-python", "ultralytics", "timm", "numpy", "opencv-python-headless",
)


def load_bench_config(path: str | Path, smoke: bool = False) -> dict[str, Any]:
    """configs/bench.yaml; with ``smoke`` its ``smoke:`` block is merged over the top."""
    cfg = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    smoke_cfg = cfg.pop("smoke", {}) or {}
    if smoke:
        cfg = merge_config(cfg, smoke_cfg)
    cfg["smoke"] = bool(smoke)
    return cfg


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GIT_COMMIT", "unknown")


def onnx_path(cfg: dict[str, Any], model: str) -> Path:
    return Path(cfg["models_dir"]) / cfg["files"][model]


def engine_path(cfg: dict[str, Any], model: str) -> Path:
    return onnx_path(cfg, model).with_suffix(".engine")


# ---------------------------------------------------------------- runtime configs + inputs


def runtime_configs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The live module configs (detector, depth, lanes with method yolop)."""
    from perception.lanes.base import load_lane_config

    lanes = load_lane_config(cfg["lanes_config"])
    lanes["method"] = "yolop"
    return {
        "detector": load_detector_config(cfg["detector_config"]),
        "depth": load_depth_config(cfg["depth_config"]),
        "lanes": lanes,
    }


def midas_cfg(rc: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return rc["depth"]["models"]["midas_small"]


class NoSession:
    """Stand-in session so ``YolopLaneDetector`` can be built for its preprocessing only."""

    def __init__(self, input_name: str = "images") -> None:
        self.input_name = input_name

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=self.input_name, shape=[1, 3, 640, 640])]

    def run(self, *_: Any) -> list:
        raise RuntimeError("NoSession only supports preprocessing")


def input_hw(model: str, rc: dict[str, dict[str, Any]]) -> tuple[int, int]:
    """Static model input (H, W) as used live."""
    from perception.detection.processing import as_hw

    if model == "yolo":
        return as_hw(rc["detector"]["imgsz"])
    if model == "midas_small":
        h, w = midas_cfg(rc)["input_hw"]
        return int(h), int(w)
    if model == "yolop":
        w, h = rc["lanes"]["yolop"]["input_size"]
        return int(h), int(w)
    raise KeyError(model)


def blob_fn(model: str, rc: dict[str, dict[str, Any]]) -> Callable[[np.ndarray], np.ndarray]:
    """BGR frame -> the exact 1x3xHxW float32 tensor the live module feeds the model."""
    if model == "yolo":
        from perception.detection.processing import letterbox, to_blob

        imgsz = rc["detector"]["imgsz"]
        return lambda frame: to_blob(letterbox(frame, imgsz)[0])
    if model == "midas_small":
        from perception.depth.estimator import preprocess

        mcfg = midas_cfg(rc)
        return lambda frame: preprocess(frame, mcfg)
    if model == "yolop":
        from perception.lanes.yolop import YolopLaneDetector

        det = YolopLaneDetector(rc["lanes"], session=NoSession())
        return lambda frame: det.preprocess(frame)[0].copy()
    raise KeyError(model)


def fixture_frames(cfg: dict[str, Any]) -> list[np.ndarray]:
    """``parity.frames`` BGR frames: the BDD fixture images, then frames of the KITTI clip."""
    p = cfg["parity"]
    n = int(p["frames"])
    frames = []
    for img_path in sorted(resolve_path(p["fixture_images"]).glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is not None:
            frames.append(img)
    cap = cv2.VideoCapture(str(resolve_path(p["fixture_clip"])))
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < n:
        raise RuntimeError(f"only {len(frames)} fixture frames available, need {n}")
    return frames[:n]


# ---------------------------------------------------------------- BDD val subset


def find_val_root(cfg: dict[str, Any]) -> Path:
    """bdd_yolo root of the data-prep kernel output (configs/data_bdd.yaml path, else search)."""
    data = yaml.safe_load(resolve_path(cfg["data_config"]).read_text(encoding="utf-8"))
    root = Path(data["path"])
    if (root / "images" / "val").is_dir():
        return root
    from scripts.data.check_data_yaml import find_dataset_root

    return find_dataset_root(Path(cfg["data_search"]))


def spread(items: list[Any], k: int) -> list[Any]:
    """``k`` evenly spaced items (all of them if there are fewer), order kept."""
    if k >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def stage_val_subset(cfg: dict[str, Any], n: int, dest: Path) -> list[tuple[Path, Path]]:
    """Copy ``n`` evenly spaced BDD val images + YOLO labels to local disk (the mount is slow).

    Returns (image, label) pairs in a fixed order, so every backend sees the same frames.
    """
    root = find_val_root(cfg)
    images = sorted((root / "images" / "val").glob("*.jpg"))
    pairs = []
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    for img in spread(images, n):
        lbl = root / "labels" / "val" / f"{img.stem}.txt"
        d_img, d_lbl = dest / "images" / img.name, dest / "labels" / lbl.name
        if not d_img.exists():
            shutil.copyfile(img, d_img)
        if lbl.exists() and not d_lbl.exists():
            shutil.copyfile(lbl, d_lbl)
        elif not d_lbl.exists():
            d_lbl.write_text("")
        pairs.append((d_img, d_lbl))
    return pairs


def local_val_pairs(dest: Path) -> list[tuple[Path, Path]]:
    """Pairs staged earlier by ``stage_val_subset`` (sorted, same order every call)."""
    imgs = sorted((dest / "images").glob("*.jpg"))
    return [(p, dest / "labels" / f"{p.stem}.txt") for p in imgs]


# ---------------------------------------------------------------- environment


def package_versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def nvidia_smi() -> dict[str, str]:
    """GPU name, driver and the CUDA version the driver supports."""
    info: dict[str, str] = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
        name, driver, mem = (s.strip() for s in out.split(","))
        info.update({"gpu": name, "driver": driver, "gpu_memory": mem})
        head = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        for tok in head.split("|"):
            if "CUDA Version" in tok:
                info["cuda_driver"] = tok.split("CUDA Version:")[1].strip()
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        pass
    return info


class VramSampler:
    """Peak GPU memory in use on device 0 while it runs (NVML, sampled every ``interval_s``).

    The benchmark runs one model + backend per process with the GPU otherwise idle, so the
    peak minus the reading at start-up is that process's footprint, CUDA context included.
    """

    def __init__(self, interval_s: float = 0.01) -> None:
        self.ok = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.ok = True
        except Exception as exc:  # noqa: BLE001 - VRAM is reported as missing, never fatal
            print(f"NVML unavailable ({type(exc).__name__}: {exc}); no VRAM numbers", flush=True)
            return
        self.base = self.used()
        self.peak = self.base
        self._stop = threading.Event()
        self._interval = interval_s
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def used(self) -> int:
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._h).used)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used())
            time.sleep(self._interval)

    def result(self) -> dict[str, float] | None:
        if not self.ok:
            return None
        self.peak = max(self.peak, self.used())
        return {
            "base_mb": self.base / 2**20,
            "peak_mb": self.peak / 2**20,
            "process_peak_mb": (self.peak - self.base) / 2**20,
        }

    def stop(self) -> dict[str, float] | None:
        if self.ok:
            self._stop.set()
            self._thread.join(timeout=1)
        return self.result()
