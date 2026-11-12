"""Bird's-eye warp, sliding-window search and 2nd-order polyfit (shared by all lane backends).

Any backend that can produce an image-space lane binary (classical thresholds, YOLOP lane mask)
calls `LaneFitter.fit` and gets image-space `Polyline`s, so every backend returns the same
`LaneResult`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from perception.lanes.threshold import frac_to_px, roi_mask
from perception.types import Polyline


class BirdsEyeWarp:
    """Perspective transform between the camera image and a bird's-eye view of the same size."""

    def __init__(self, src_frac: list, dst_frac: list, image_hw: tuple[int, int]) -> None:
        h, w = image_hw
        self.image_hw = (h, w)
        self.src = frac_to_px(src_frac, w, h)
        self.dst = frac_to_px(dst_frac, w, h)
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, binary: np.ndarray) -> np.ndarray:
        """Warp an HxW bool mask to the bird's-eye view (nearest neighbour keeps it binary)."""
        h, w = self.image_hw
        out = cv2.warpPerspective(
            binary.astype(np.uint8), self.M, (w, h), flags=cv2.INTER_NEAREST
        )
        return out > 0

    def to_image(self, pts_xy: np.ndarray) -> np.ndarray:
        """Map (N, 2) bird's-eye points back to image coordinates."""
        pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.Minv).reshape(-1, 2)


def find_lane_bases(binary: np.ndarray, histogram_frac: float) -> tuple[int | None, int | None]:
    """Column-histogram peaks of the bottom part of a bird's-eye mask, one per image half."""
    h, w = binary.shape
    start = int(h * (1.0 - histogram_frac))
    hist = binary[start:].sum(axis=0)
    mid = w // 2
    left = int(np.argmax(hist[:mid])) if hist[:mid].max() > 0 else None
    right = mid + int(np.argmax(hist[mid:])) if hist[mid:].max() > 0 else None
    return left, right


def sliding_window(
    binary: np.ndarray, base_x: int, n_windows: int, margin: int, min_pix: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Follow one lane line upward from `base_x`.

    Returns (xs, ys, hits): the lane pixels collected and the number of windows that held at least
    `min_pix` pixels (those windows re-centre on the mean x of their pixels).
    """
    h, _ = binary.shape
    nz_y, nz_x = np.nonzero(binary)
    win_h = h / n_windows
    x_cur = base_x
    keep: list[np.ndarray] = []
    hits = 0
    for i in range(n_windows):
        y_hi = int(round(h - i * win_h))
        y_lo = int(round(h - (i + 1) * win_h))
        sel = (nz_y >= y_lo) & (nz_y < y_hi) & (nz_x >= x_cur - margin) & (nz_x < x_cur + margin)
        idx = np.flatnonzero(sel)
        keep.append(idx)
        if idx.size >= min_pix:
            x_cur = int(round(float(nz_x[idx].mean())))
            hits += 1
    idx = np.concatenate(keep) if keep else np.empty(0, dtype=np.int64)
    return nz_x[idx], nz_y[idx], hits


def fit_poly(xs: np.ndarray, ys: np.ndarray, min_points: int) -> np.ndarray | None:
    """Least-squares x = a*y^2 + b*y + c. Returns (a, b, c) or None if there is too little data."""
    if xs.size < max(min_points, 3) or np.unique(ys).size < 3:
        return None
    return np.polyfit(ys.astype(np.float64), xs.astype(np.float64), 2)


def ego_mask_from_polylines(
    left: Polyline, right: Polyline, shape_hw: tuple[int, int], y_top: float
) -> np.ndarray | None:
    """Bool HxW mask of the region between the two boundaries, from `y_top` to the image bottom."""
    h, w = shape_hw
    ys = np.arange(max(int(math.ceil(y_top)), 0), h, dtype=np.float64)
    if ys.size < 2:
        return None
    xl = np.asarray(left.x_at(ys), dtype=np.float64)
    xr = np.asarray(right.x_at(ys), dtype=np.float64)
    valid = xr > xl
    if valid.sum() < 2:
        return None
    ys = ys[valid]
    xl, xr = np.clip(xl[valid], -w, 2 * w), np.clip(xr[valid], -w, 2 * w)  # keep int32-safe
    poly = np.concatenate([np.stack([xl, ys], axis=1), np.stack([xr, ys], axis=1)[::-1]])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask.astype(bool)


@dataclass
class LaneFit:
    """Per-frame fit before temporal smoothing."""

    left: Polyline | None
    right: Polyline | None
    conf_left: float
    conf_right: float
    y_top: float  # image row where the bird's-eye view starts (top of the measured lane region)
    debug: dict = field(default_factory=dict)


class LaneFitter:
    """ROI -> bird's-eye warp -> sliding window -> polyfit -> image-space polylines.

    `cfg` is the `fitting` section of configs/lanes.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._warp: BirdsEyeWarp | None = None
        self._roi: np.ndarray | None = None

    def _prepare(self, shape_hw: tuple[int, int]) -> None:
        if self._warp is None or self._warp.image_hw != shape_hw:
            wcfg = self.cfg["warp"]
            self._warp = BirdsEyeWarp(wcfg["src"], wcfg["dst"], shape_hw)
            self._roi = roi_mask(shape_hw, self.cfg["roi"]["polygon"])

    @property
    def warp(self) -> BirdsEyeWarp | None:
        return self._warp

    def fit(self, binary: np.ndarray) -> LaneFit:
        h, w = binary.shape
        self._prepare((h, w))
        assert self._warp is not None and self._roi is not None
        sw = self.cfg["sliding_window"]
        n_windows = int(sw["n_windows"])
        margin = max(1, int(round(float(sw["margin_frac"]) * w)))
        win_area = 2 * margin * h / n_windows
        min_pix = max(1, int(math.ceil(float(sw["min_pix_frac"]) * win_area)))
        min_fit = max(3, int(math.ceil(float(self.cfg["min_fit_pix_frac"]) * h * w)))
        min_hits = float(self.cfg["min_window_hits_frac"]) * n_windows

        warped = self._warp.warp(binary.astype(bool) & self._roi)
        bases = find_lane_bases(warped, float(sw["histogram_frac"]))

        coeffs_w: list[np.ndarray | None] = []
        confs: list[float] = []
        counts: list[int] = []
        for base in bases:
            if base is None:
                coeffs_w.append(None)
                confs.append(0.0)
                counts.append(0)
                continue
            xs, ys, hits = sliding_window(warped, base, n_windows, margin, min_pix)
            coeffs = fit_poly(xs, ys, min_fit) if hits >= min_hits else None
            coeffs_w.append(coeffs)
            confs.append(hits / n_windows if coeffs is not None else 0.0)
            counts.append(int(xs.size))

        # Plausibility: the gap between the boundaries must look like a lane at top and bottom.
        if coeffs_w[0] is not None and coeffs_w[1] is not None:
            lo, hi = (float(v) * w for v in self.cfg["lane_width_frac"])
            rows = np.array([0.0, h - 1.0])
            gap = np.polyval(coeffs_w[1], rows) - np.polyval(coeffs_w[0], rows)
            if np.any(gap < lo) or np.any(gap > hi):
                weaker = 0 if (confs[0], counts[0]) < (confs[1], counts[1]) else 1
                coeffs_w[weaker] = None
                confs[weaker] = 0.0

        n_samples = int(self.cfg["refit_samples"])
        ys_w = np.linspace(0.0, h - 1.0, n_samples)
        polys: list[Polyline | None] = []
        for i, coeffs in enumerate(coeffs_w):
            if coeffs is None:
                polys.append(None)
                continue
            pts = self._warp.to_image(np.stack([np.polyval(coeffs, ys_w), ys_w], axis=1))
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            img_coeffs = fit_poly(pts[:, 0], pts[:, 1], 3)
            if img_coeffs is None or not np.all(np.isfinite(img_coeffs)):
                polys.append(None)
                confs[i] = 0.0
                continue
            polys.append(Polyline(img_coeffs))

        y_top = float(self._warp.src[:, 1].min())
        debug = {"warped": warped, "bases": bases, "warped_coeffs": coeffs_w}
        return LaneFit(polys[0], polys[1], confs[0], confs[1], y_top, debug)
