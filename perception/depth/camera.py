"""Flat-ground geometric distance from camera intrinsics, mount height and pitch (M4.2).

    cam = load_camera("configs/camera.yaml", "kitti")
    z, std = cam.ground_distance(v_bottom)          # metres, or None above the horizon

The model and conventions are documented in configs/camera.yaml: pitch > 0 tilts the camera
down, Z is the forward distance to the ground-contact point (camera-frame depth at zero pitch).
Pure numpy, no torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMERA_KEYS = ("width", "height", "fx", "fy", "cx", "cy", "height_m", "pitch_deg")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float
    pitch_rad: float
    sigma_v_px: float = 0.0
    sigma_pitch_rad: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraModel:
        missing = [k for k in CAMERA_KEYS if k not in d]
        if missing:
            raise KeyError(f"camera config is missing {missing}")
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            height_m=float(d["height_m"]),
            pitch_rad=math.radians(float(d["pitch_deg"])),
            sigma_v_px=float(d.get("sigma_v_px", 0.0)),
            sigma_pitch_rad=math.radians(float(d.get("sigma_pitch_deg", 0.0))),
        )

    def with_intrinsics(self, P: np.ndarray, hw: tuple[int, int]) -> CameraModel:
        """Copy with fx, fy, cx, cy from a 3x4 projection matrix (e.g. KITTI P2) and size hw."""
        return replace(
            self,
            width=int(hw[1]),
            height=int(hw[0]),
            fx=float(P[0, 0]),
            fy=float(P[1, 1]),
            cx=float(P[0, 2]),
            cy=float(P[1, 2]),
        )

    def scaled_to(self, hw: tuple[int, int]) -> CameraModel:
        """Copy for a frame of size hw (same field of view, intrinsics rescaled linearly)."""
        h, w = int(hw[0]), int(hw[1])
        if (h, w) == (self.height, self.width):
            return self
        sx, sy = w / self.width, h / self.height
        return replace(
            self,
            width=w,
            height=h,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            sigma_v_px=self.sigma_v_px * sy,
        )

    @property
    def horizon_row(self) -> float:
        """Image row of the horizon (flat ground at infinity)."""
        return self.cy - self.fy * math.tan(self.pitch_rad)

    def ground_angle(self, v: float) -> float:
        """Angle below the horizontal (rad) of the ray through image row v."""
        return self.pitch_rad + math.atan((v - self.cy) / self.fy)

    def ground_distance(self, v: float, min_angle_rad: float = 0.0) -> tuple[float, float] | None:
        """(Z, std) in metres for a ground-contact point at image row v, or None if the ray does
        not hit the ground at least ``min_angle_rad`` below the horizontal."""
        alpha = self.ground_angle(v)
        if alpha <= max(min_angle_rad, 0.0):
            return None
        z = self.height_m / math.tan(alpha)
        return z, self.distance_std(z)

    def distance_std(self, z: float) -> float:
        """1-sigma error of a flat-ground distance z from row noise and pitch noise."""
        return z * z * self.inv_depth_std()

    def inv_depth_std(self) -> float:
        """1-sigma error of 1/Z for a flat-ground distance (1/m), the same at every distance."""
        return math.hypot(self.sigma_v_px / self.fy, self.sigma_pitch_rad) / self.height_m

    def row_of_distance(self, z: float) -> float:
        """Inverse of ground_distance: image row of the ground point at forward distance z."""
        return self.cy + self.fy * math.tan(math.atan2(self.height_m, z) - self.pitch_rad)


def load_camera(path: str | Path, name: str) -> CameraModel:
    """Camera ``name`` (kitti | carla | bdd) from configs/camera.yaml."""
    cfg = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8")) or {}
    if name not in cfg:
        raise KeyError(f"camera {name!r} not in {path}; available: {sorted(cfg)}")
    return CameraModel.from_dict(cfg[name])
