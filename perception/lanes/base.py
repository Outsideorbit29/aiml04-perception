"""Common lane-detector interface and config-driven factory.

Every backend turns a BGR frame into an image-space lane binary and hands it to the shared
`LaneFitter` + `LaneSmoother`, so all backends return the same `LaneResult` (ARCHITECTURE.md 3.4).
Switch backends with `method:` in configs/lanes.yaml.

Two calling modes:
- Video (default): `detect(frame)` keeps temporal state (EMA smoothing, `every_n_frames` reuse).
  State is reset automatically when the frame size changes.
- Independent images (dataset eval): `detect(frame, stateless=True)` always runs the backend,
  skips smoothing and the result cache, and leaves the video state untouched. No `reset()` needed.

Debug data (lane binary, fitted rows, synthesized side) of the last backend run goes to
`last_debug` for video calls and to `last_debug_stateless` for stateless calls, so stateless
calls never overwrite the debug data a video overlay is drawing from.

Single-side fallback (`single_side_fallback: true` in the config): when only one boundary is
found, the other is synthesized by offsetting the found one by `fallback_lane_width_frac` of the
bird's-eye width, so the ego mask can still be built. The synthesized side is NOT returned in
`LaneResult.left/right`; those always hold measured boundaries only. So a result with an
`ego_mask` but one side `None` means "mask built from one measured side". Its confidence is
the mean of the found side's confidence and `fallback_conf_mult` times it, i.e. at most
(1 + fallback_conf_mult) / 2 = 0.75 of a perfect side with the default 0.5. So a confident
one-sided result can score higher than a weak two-sided one: `confidence` is not an ordering
of "both sides measured" over "one side measured"; check `left`/`right` for that. The
synthesized polyline is available for overlays as `last_debug["synthesized_line"]` (side name in
`last_debug["synthesized_side"]`; `last_debug_stateless` for stateless calls).
"""

from __future__ import annotations

import copy
import importlib
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import yaml

from perception.lanes.fitting import LaneFitter, ego_mask_from_polylines, offset_in_birdseye
from perception.lanes.smoothing import LaneSmoother
from perception.types import LaneResult, Polyline

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
        self.fallback = bool(cfg.get("single_side_fallback", False))
        if self.fallback:
            self.fallback_width_frac = float(cfg["fallback_lane_width_frac"])
            self.fallback_conf_mult = float(cfg["fallback_conf_mult"])
            lo, hi = (float(v) for v in cfg["fitting"]["lane_width_frac"])
            if not lo <= self.fallback_width_frac <= hi:
                raise ValueError(
                    f"fallback_lane_width_frac {self.fallback_width_frac} is outside "
                    f"fitting.lane_width_frac [{lo}, {hi}]"
                )
            if not 0.0 <= self.fallback_conf_mult <= 1.0:
                raise ValueError(f"fallback_conf_mult {self.fallback_conf_mult} not in [0, 1]")
        self.last_debug: dict = {}  # last video-mode (`detect(frame)`) backend run
        self.last_debug_stateless: dict = {}  # last `detect(frame, stateless=True)` run
        self.last_timing_ms: dict[str, float] = {}  # stage times of the last backend run
        self._frame_idx = 0
        self._last: LaneResult | None = None
        self._frame_hw: tuple[int, int] | None = None

    def detect(self, frame_bgr: np.ndarray, stateless: bool = False) -> LaneResult:
        """Lane boundaries for one HxWx3 uint8 BGR frame, in image coordinates.

        `stateless=True` treats the frame as an independent image: the backend always runs, no
        smoothing is applied, and the temporal state used by video calls (including
        `last_debug`) is neither read nor changed; its debug data goes to
        `last_debug_stateless`. Use it for dataset evaluation.
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
        syn_side: str | None = None
        syn_line: Polyline | None = None
        if self.make_ego_mask and left is not None and right is not None:
            mask = ego_mask_from_polylines(left, right, (h, w), fit.y_top, fit.y_bottom)
        elif self.make_ego_mask and self.fallback and (left is None) != (right is None):
            syn_side, syn_line, mask = self._single_side_mask(
                left, right, fit.y_top, fit.y_bottom, (h, w)
            )
            if mask is not None:  # synthesized side: lower confidence, never in left/right
                if syn_side == "right":
                    c_r = self.fallback_conf_mult * c_l
                else:
                    c_l = self.fallback_conf_mult * c_r
            else:
                syn_side, syn_line = None, None
        debug = {
            "binary": binary,
            "y_top": fit.y_top,
            "y_bottom": fit.y_bottom,
            "synthesized_side": syn_side,
            "synthesized_line": syn_line,
            **fit.debug,
        }
        if smooth:  # video call
            self.last_debug = debug
        else:
            self.last_debug_stateless = debug
        return LaneResult(left, right, mask, float(0.5 * (c_l + c_r)))

    def _single_side_mask(
        self,
        left: Polyline | None,
        right: Polyline | None,
        y_top: float,
        y_bottom: float,
        shape_hw: tuple[int, int],
    ) -> tuple[str, Polyline | None, np.ndarray | None]:
        """(synthesized side name, its polyline, ego mask) from the one measured side."""
        warp = self.fitter.warp
        assert warp is not None
        found = left if left is not None else right
        assert found is not None
        side = "right" if left is not None else "left"
        sign = 1.0 if side == "right" else -1.0
        offset = sign * self.fallback_width_frac * (shape_hw[1] - 1)
        n = int(self.cfg["fitting"]["refit_samples"])
        syn = offset_in_birdseye(found, warp, y_top, y_bottom, offset, n)
        if syn is None:
            return side, None, None
        pair = (found, syn) if side == "right" else (syn, found)
        return side, syn, ego_mask_from_polylines(*pair, shape_hw, y_top, y_bottom)
