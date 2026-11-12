"""Shared data contracts for the perception stack (ARCHITECTURE.md section 4).

Every module consumes and produces these types. Changing a field means updating
ARCHITECTURE.md and every consumer in the same change.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Rows used to fit the mask edge lines when extending the ego mask past its measured rows.
_EDGE_FIT_ROWS = 10


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
        """True if image point (x, y) lies between the left and right boundaries.

        The mask only covers the rows the lane fit measured. Inside those rows it is used
        directly. Outside them (e.g. the bottom-centre of a close lead vehicle below the
        band) the two measured polylines decide; with only one measured side, the mask's
        left/right edges are extended linearly from its nearest end, which follows the
        perspective of straight lane boundaries.
        """
        mask = self.ego_mask
        rows = np.flatnonzero(mask.any(axis=1)) if mask is not None else np.empty(0, int)
        if rows.size:
            h, w = mask.shape
            xi, yi = int(round(x)), int(round(y))
            if rows[0] <= yi <= rows[-1] and 0 <= xi < w:
                return bool(mask[yi, xi])
        if self.left is not None and self.right is not None:
            return float(self.left.x_at(y)) <= x <= float(self.right.x_at(y))
        if rows.size < 2:
            return False
        end = rows[-1] if y > rows[-1] else rows[0]
        near = (
            rows[max(0, rows.size - 1 - _EDGE_FIT_ROWS)]
            if end == rows[-1]
            else rows[min(rows.size - 1, _EDGE_FIT_ROWS)]
        )
        if near == end:
            return False
        lo = self._edge_line(mask, end, near, 0, y)
        hi = self._edge_line(mask, end, near, -1, y)
        return lo <= x <= hi

    @staticmethod
    def _edge_line(mask: np.ndarray, r0: int, r1: int, side: int, y: float) -> float:
        """x of the mask's left (side=0) or right (side=-1) edge, extended linearly to row y."""
        x0 = float(np.flatnonzero(mask[r0])[side])
        x1 = float(np.flatnonzero(mask[r1])[side])
        return x0 + (x1 - x0) * (y - r0) / (r1 - r0)


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
