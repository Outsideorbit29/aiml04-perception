"""Shared data contracts for the perception stack (ARCHITECTURE.md section 4).

Every module consumes and produces these types. Changing a field means updating
ARCHITECTURE.md and every consumer in the same change.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """One detected object in image coordinates."""

    bbox_xyxy: np.ndarray  # shape (4,), float32: x1, y1, x2, y2 in pixels
    score: float
    cls: int
    cls_name: str

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Image point where the object touches the road (used for distance and lane checks)."""
        x1, _, x2, y2 = self.bbox_xyxy
        return float((x1 + x2) / 2.0), float(y2)


@dataclass
class Track:
    """A detection with a persistent identity across frames."""

    track_id: int
    det: Detection
    age: int = 0
    history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))


@dataclass
class Polyline:
    """Lane boundary as x = a*y^2 + b*y + c in image coordinates."""

    coeffs: np.ndarray  # shape (3,): a, b, c

    def x_at(self, y: float | np.ndarray) -> float | np.ndarray:
        a, b, c = self.coeffs
        return a * np.square(y) + b * y + c


@dataclass
class LaneResult:
    left: Polyline | None
    right: Polyline | None
    ego_mask: np.ndarray | None  # HxW bool mask of the ego lane, if available
    confidence: float

    def contains(self, x: float, y: float) -> bool:
        """True if image point (x, y) lies between the left and right boundaries."""
        if self.ego_mask is not None:
            h, w = self.ego_mask.shape
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < h and 0 <= xi < w:
                return bool(self.ego_mask[yi, xi])
        if self.left is None or self.right is None:
            return False
        return float(self.left.x_at(y)) <= x <= float(self.right.x_at(y))


@dataclass
class ObjectState:
    track: Track
    distance_m: float
    distance_std: float
    closing_speed_mps: float  # positive = getting closer
    ttc_s: float | None  # time to collision; None if not closing
    in_ego_lane: bool


@dataclass
class PerceptionFrame:
    stamp: float  # seconds
    frame_id: int
    objects: list[ObjectState]
    lanes: LaneResult
    timings_ms: dict[str, float] = field(default_factory=dict)

    def lead_object(self) -> ObjectState | None:
        """Closest object in the ego lane, which the controller reacts to."""
        in_lane = [o for o in self.objects if o.in_ego_lane]
        return min(in_lane, key=lambda o: o.distance_m) if in_lane else None
