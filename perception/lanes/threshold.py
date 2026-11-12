"""Colour and gradient thresholds for the classical lane pipeline (CPU, OpenCV only)."""

from __future__ import annotations

import cv2
import numpy as np


def frac_to_px(points_frac: list | np.ndarray, width: int, height: int) -> np.ndarray:
    """Map (x/(W-1), y/(H-1)) fractions to pixel coordinates, float32 array of shape (N, 2)."""
    pts = np.asarray(points_frac, dtype=np.float32).reshape(-1, 2)
    return pts * np.array([width - 1, height - 1], dtype=np.float32)


def roi_mask(shape_hw: tuple[int, int], polygon_frac: list | np.ndarray) -> np.ndarray:
    """Boolean HxW mask that is True inside the ROI polygon given in image fractions."""
    h, w = shape_hw
    pts = np.round(frac_to_px(polygon_frac, w, h)).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _in_range(channel: np.ndarray, bounds: list | tuple) -> np.ndarray:
    lo, hi = bounds
    return (channel >= lo) & (channel <= hi)


def threshold_binary(
    frame_bgr: np.ndarray, cfg: dict, region: np.ndarray | None = None
) -> np.ndarray:
    """Lane-pixel candidates: Sobel-x on lightness OR saturation OR lightness, as an HxW bool mask.

    `cfg` is the `classical.threshold` section of configs/lanes.yaml. `region` (HxW bool, usually
    the ROI) sets where the Sobel normalisation peak is measured, so a strong edge in the sky or
    on the hood cannot scale the road gradients down. The saturation (yellow) test also needs a
    lightness of at least `s_min_lightness`, which rejects dark, noisy, saturated pixels.
    """
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    light, sat = hls[:, :, 1], hls[:, :, 2]

    sobel = np.abs(cv2.Sobel(light, cv2.CV_64F, 1, 0, ksize=int(cfg["sobel_kernel"])))
    inside = sobel[region] if region is not None and region.any() else sobel
    peak = float(inside.max()) if inside.size else 0.0
    if peak > 0:
        sobel_scaled = np.clip(255.0 * sobel / peak, 0, 255).astype(np.uint8)
    else:
        sobel_scaled = np.zeros_like(light)

    grad = _in_range(sobel_scaled, cfg["sobel_x"])
    yellow = _in_range(sat, cfg["s_channel"]) & (light >= int(cfg["s_min_lightness"]))
    white = _in_range(light, cfg["l_channel"])
    return grad | yellow | white
