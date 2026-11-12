"""Common lane-detector interface and config-driven factory.

Every backend turns a BGR frame into an image-space lane binary and hands it to the shared
`LaneFitter` + `LaneSmoother`, so all backends return the same `LaneResult` (ARCHITECTURE.md 3.4).
Switch backends with `method:` in configs/lanes.yaml.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import yaml

from perception.lanes.fitting import LaneFitter, ego_mask_from_polylines
from perception.lanes.smoothing import LaneSmoother
from perception.types import LaneResult

# method name -> (module, class). Imported lazily so heavy backends (torch for YOLOP) only load
# when selected.
_BACKENDS: dict[str, tuple[str, str]] = {
    "classical": ("perception.lanes.classical", "ClassicalLaneDetector"),
    "yolop": ("perception.lanes.yolop", "YolopLaneDetector"),
}


def load_lane_config(path: str | Path) -> dict:
    """Read a configs/lanes*.yaml file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_lane_detector(cfg: dict) -> LaneDetector:
    """Create the backend named by `cfg["method"]` (classical | yolop)."""
    method = cfg.get("method", "classical")
    if method not in _BACKENDS:
        raise ValueError(f"Unknown lane method {method!r}; expected one of {sorted(_BACKENDS)}")
    module_name, class_name = _BACKENDS[method]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(cfg)


class LaneDetector(ABC):
    """Base class: `detect(frame_bgr) -> LaneResult`, plus `reset()` between unrelated frames.

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
        self._frame_idx = 0
        self._last: LaneResult | None = None

    def detect(self, frame_bgr: np.ndarray) -> LaneResult:
        """Lane boundaries for one HxWx3 uint8 BGR frame, in image coordinates."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError(
                f"expected HxWx3 uint8 BGR frame, got shape {frame_bgr.shape} {frame_bgr.dtype}"
            )
        run = self._last is None or self._frame_idx % self.every_n == 0
        self._frame_idx += 1
        if run:
            self._last = self._infer(frame_bgr)
        assert self._last is not None
        return self._last

    def reset(self) -> None:
        """Forget temporal state (call between unrelated images or clips)."""
        self._frame_idx = 0
        self._last = None
        self.smoother.reset()
        self.last_debug = {}

    @abstractmethod
    def lane_binary(self, frame_bgr: np.ndarray) -> np.ndarray:
        """HxW bool mask of lane-marking pixels in image coordinates."""

    def _infer(self, frame_bgr: np.ndarray) -> LaneResult:
        return self._lanes_from_binary(self.lane_binary(frame_bgr))

    def _lanes_from_binary(self, binary: np.ndarray) -> LaneResult:
        h, w = binary.shape
        fit = self.fitter.fit(binary)
        left, right, c_l, c_r = self.smoother.update(
            fit.left, fit.right, fit.conf_left, fit.conf_right, y_ref=h - 1.0, width=w
        )
        mask = None
        if self.make_ego_mask and left is not None and right is not None:
            mask = ego_mask_from_polylines(left, right, (h, w), fit.y_top)
        self.last_debug = {"binary": binary, "y_top": fit.y_top, **fit.debug}
        return LaneResult(left, right, mask, float(0.5 * (c_l + c_r)))
