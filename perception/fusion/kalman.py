"""Per-track constant-velocity Kalman filter on distance: closing speed and TTC (M4.3).

State x = [d, v] with d the distance (m) and v = dd/dt (m/s); closing speed = -v (positive when
the object gets closer). dt comes from the frame stamps (KITTI 10 fps, CARLA 20 Hz), and the
process noise is white acceleration with std ``accel_std`` (m/s^2).

``FilterBank`` keeps one filter per track id and tolerates the tracker's behaviour:
- ids appear a few frames after the first detection, or disappear for a few frames after
  ``Tracker.reset()``: a filter is created on the first measurement of an id;
- an id that has not been measured for more than ``max_gap_s`` restarts from scratch, and filters
  older than ``max_age_s`` are pruned on every ``prune()``;
- id switches (the id jumps to another object) show up as a class change or as an innovation
  outside the ``gate_chi2`` gate; both restart the filter instead of blending two objects;
- a stamp that does not advance (dt <= 0) restarts the filter.
TTC is reported only after ``min_updates_for_ttc`` measurements and when the closing speed is
above ``min_closing_speed_mps``; otherwise it is None. Pure numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

REQUIRED_KEYS = (
    "accel_std",
    "init_velocity_std",
    "gate_chi2",
    "max_gap_s",
    "max_age_s",
    "min_updates_for_ttc",
    "min_closing_speed_mps",
)


@dataclass
class FilterOutput:
    distance_m: float
    distance_std: float
    closing_speed_mps: float  # positive = getting closer
    ttc_s: float | None
    restarted: bool  # True if this call created or re-created the filter


class DistanceKalman:
    """1D constant-velocity Kalman filter for one object."""

    def __init__(self, cfg: dict[str, Any], z: float, r_std: float, stamp: float, cls: int):
        self.cfg = cfg
        self.x = np.array([z, 0.0])
        self.P = np.diag([r_std * r_std, float(cfg["init_velocity_std"]) ** 2])
        self.stamp = stamp  # time of the state (advanced by predict)
        self.last_meas = stamp  # time of the last measurement
        self.cls = cls
        self.updates = 1

    def predict(self, stamp: float) -> None:
        dt = stamp - self.stamp
        if dt <= 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        q = float(self.cfg["accel_std"]) ** 2
        Q = q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.stamp = stamp

    def nis(self, z: float, r_std: float) -> float:
        """Normalised innovation squared of a measurement (after ``predict``)."""
        s = self.P[0, 0] + r_std * r_std
        return float((z - self.x[0]) ** 2 / s)

    def update(self, z: float, r_std: float) -> None:
        s = self.P[0, 0] + r_std * r_std
        k = self.P[:, 0] / s
        self.x = self.x + k * (z - self.x[0])
        self.P = self.P - np.outer(k, self.P[0, :])
        self.P = 0.5 * (self.P + self.P.T)
        self.last_meas = self.stamp
        self.updates += 1

    def output(self, restarted: bool) -> FilterOutput:
        d = float(self.x[0])
        closing = -float(self.x[1])
        ttc = None
        if (
            self.updates >= int(self.cfg["min_updates_for_ttc"])
            and closing > float(self.cfg["min_closing_speed_mps"])
            and d > 0
        ):
            ttc = d / closing
        return FilterOutput(d, math.sqrt(max(self.P[0, 0], 0.0)), closing, ttc, restarted)


class FilterBank:
    """One ``DistanceKalman`` per track id. Call ``prune(stamp)`` once per frame."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in cfg]
        if missing:
            raise KeyError(f"kalman config is missing {missing}")
        self.cfg = cfg
        self.filters: dict[int, DistanceKalman] = {}
        self.restarts = 0  # gate / class / time restarts of existing ids (diagnostics)

    def reset(self) -> None:
        self.filters.clear()
        self.restarts = 0

    def __contains__(self, track_id: int) -> bool:
        return track_id in self.filters

    def update(self, track_id: int, cls: int, stamp: float, z: float, r_std: float) -> FilterOutput:
        """Fuse measurement z (std r_std) for ``track_id`` at ``stamp``."""
        f = self.filters.get(track_id)
        if f is not None and self._must_restart(f, cls, stamp, z, r_std):
            self.restarts += 1
            f = None
        if f is None:
            f = DistanceKalman(self.cfg, z, r_std, stamp, cls)
            self.filters[track_id] = f
            return f.output(restarted=True)
        f.update(z, r_std)
        return f.output(restarted=False)

    def predict_only(self, track_id: int, stamp: float) -> FilterOutput | None:
        """Prediction for a track without a measurement this frame (None if unknown/stale)."""
        f = self.filters.get(track_id)
        if f is None or stamp < f.stamp or stamp - f.last_meas > float(self.cfg["max_gap_s"]):
            return None
        f.predict(stamp)
        return f.output(restarted=False)

    def prune(self, stamp: float) -> None:
        """Drop filters not updated for more than ``max_age_s`` (or stamped in the future)."""
        max_age = float(self.cfg["max_age_s"])
        stale = [i for i, f in self.filters.items() if not 0 <= stamp - f.last_meas <= max_age]
        for i in stale:
            del self.filters[i]

    def _must_restart(
        self, f: DistanceKalman, cls: int, stamp: float, z: float, r_std: float
    ) -> bool:
        gap = stamp - f.last_meas
        if cls != f.cls or stamp < f.stamp or gap <= 0 or gap > float(self.cfg["max_gap_s"]):
            return True
        f.predict(stamp)
        return f.nis(z, r_std) > float(self.cfg["gate_chi2"])
