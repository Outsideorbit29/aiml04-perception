"""Detector-box calibration of the flat-ground geometric distance (M5.6 decision 1).

The geometric distance uses the box bottom row as the ground-contact row. A detector's box
bottom does not sit exactly where the distance ground truth puts it: in CARLA the fine-tuned
YOLO11s box bottoms sit higher than the projected 3D-box bottoms, so distances on detections read
~10 % long (docs/EXPERIMENTS.md, M5.6). ``BottomCalibration`` corrects that per camera and
detector with values from config (``geometry.calibration``, configs/depth*.yaml):

    v' = v_bottom + offset_px * (H / H_cfg) + offset_box_frac * box_h      (row correction)
    Z' = distance_scale * Z(v')                                            (scale correction)

``H / H_cfg`` rescales the pixel offset from the camera.yaml resolution to the frame size. The
identity (0, 0, 1) is the default, so cameras without a fitted calibration are unchanged.

The fit helpers are pure numpy and are used by the sim eval (sim/eval_frames.py) on a fit split
of the recorded frames; the fitted values are then copied into the config and checked on the
held-out split. Fitting and scoring on the same frames is never reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.depth.camera import CameraModel

CALIBRATION_KEYS = ("offset_px", "offset_box_frac", "distance_scale")


@dataclass(frozen=True)
class BottomCalibration:
    """Correction of the detection-based geometric distance (module docstring)."""

    offset_px: float = 0.0  # + = box bottom moved down (closer), at camera.yaml resolution
    offset_box_frac: float = 0.0  # + this x box height (px of the frame)
    distance_scale: float = 1.0  # multiplies the distance of the corrected row

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> BottomCalibration:
        d = d or {}
        unknown = set(d) - set(CALIBRATION_KEYS)
        if unknown:
            raise KeyError(f"geometry.calibration has unknown keys {sorted(unknown)}")
        out = cls(**{k: float(d[k]) for k in CALIBRATION_KEYS if k in d})
        for k, v in vars(out).items():
            if not math.isfinite(v):
                raise ValueError(f"geometry.calibration.{k} = {v} must be finite")
        if out.distance_scale <= 0:
            raise ValueError("geometry.calibration.distance_scale must be > 0")
        return out

    @property
    def identity(self) -> bool:
        return self.offset_px == 0 and self.offset_box_frac == 0 and self.distance_scale == 1

    def corrected_row(self, y1: float, y2: float, row_scale: float = 1.0) -> float:
        """Calibrated ground-contact row of a box ``[y1, y2]``; ``row_scale`` = frame height /
        configured camera height (the pixel offset is defined at the configured size)."""
        return y2 + self.offset_px * row_scale + self.offset_box_frac * (y2 - y1)

    def distance(
        self,
        cam: CameraModel,
        y1: float,
        y2: float,
        min_angle_rad: float = 0.0,
        row_scale: float = 1.0,
    ) -> tuple[float, float] | None:
        """(Z, std) of the calibrated contact row, or None above the ``min_angle`` horizon."""
        est = cam.ground_distance(self.corrected_row(y1, y2, row_scale), min_angle_rad)
        if est is None:
            return None
        z = est[0] * self.distance_scale
        return z, cam.distance_std(z)


def box_geometric_distance(
    cam: CameraModel,
    y1: float,
    y2: float,
    calibration: BottomCalibration | None = None,
    min_angle_rad: float = 0.0,
    max_distance_m: float = math.inf,
    bottom_margin_px: float = 0.0,
    row_scale: float = 1.0,
) -> tuple[float, float] | None:
    """Flat-ground (Z, std) of a box with rows ``[y1, y2]`` in a frame of ``cam``'s size, or
    None when the box is cut by the image bottom (``y2`` within ``bottom_margin_px``: contact not
    visible, checked on the raw row), the contact row is less than ``min_angle_rad`` below the
    horizontal, or Z exceeds ``max_distance_m``. ``calibration`` None or identity = raw row."""
    if y2 > cam.height - 1 - bottom_margin_px:
        return None
    if calibration is None or calibration.identity:
        est = cam.ground_distance(y2, min_angle_rad)
    else:
        est = calibration.distance(cam, y1, y2, min_angle_rad, row_scale)
    if est is None or est[0] > max_distance_m:
        return None
    return est


def fit_distance_scale(z_pred: np.ndarray, z_gt: np.ndarray) -> float:
    """Robust multiplicative scale ``k`` with ``k * z_pred ~ z_gt``: exp(median log ratio)."""
    z_pred, z_gt = np.asarray(z_pred, float).ravel(), np.asarray(z_gt, float).ravel()
    ok = np.isfinite(z_pred) & np.isfinite(z_gt) & (z_pred > 0) & (z_gt > 0)
    if not ok.any():
        raise ValueError("no valid (prediction, GT) pairs to fit a distance scale")
    return float(np.exp(np.median(np.log(z_gt[ok] / z_pred[ok]))))


def fit_row_offset(
    v_bottom: np.ndarray,
    box_h: np.ndarray,
    v_target: np.ndarray,
    with_box_h: bool,
    huber_k: float = 1.5,
    iters: int = 10,
) -> tuple[float, float]:
    """Robust fit of ``v_target - v_bottom = a + b * box_h`` (``b = 0`` if not ``with_box_h``)
    by Huber IRLS; ``v_target`` = the row that would give the GT distance
    (``CameraModel.row_of_distance``). Pixel-row residuals are close to Gaussian in inverse
    depth, which is how the geometric error behaves. Returns ``(a, b)`` in frame pixels."""
    v, h, t = (np.asarray(x, float).ravel() for x in (v_bottom, box_h, v_target))
    ok = np.isfinite(v) & np.isfinite(h) & np.isfinite(t)
    if ok.sum() < (2 if with_box_h else 1):
        raise ValueError("too few valid rows to fit a bottom-row offset")
    r = t[ok] - v[ok]
    a_mat = np.stack([np.ones(r.size), h[ok]], 1) if with_box_h else np.ones((r.size, 1))
    w = np.ones(r.size)
    coef = np.zeros(a_mat.shape[1])
    for _ in range(max(1, iters)):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(a_mat * sw[:, None], r * sw, rcond=None)
        res = r - a_mat @ coef
        sigma = 1.4826 * float(np.median(np.abs(res - np.median(res))))
        if not sigma > 0:
            break
        u = np.abs(res) / (huber_k * sigma)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
    a = float(coef[0])
    b = float(coef[1]) if with_box_h else 0.0
    return a, b
