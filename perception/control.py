"""Longitudinal AEB / ACC decision logic (M5.4), ROS-free and CARLA-free.

The ROS2 ``control_node`` is a thin wrapper around :class:`LongitudinalController`: it turns
each ``/perception/objects`` message into :class:`LeadCandidate` objects, calls ``update`` with
the ego speed from odometry and publishes the returned :class:`ControlDecision`. Lateral control
stays with the CARLA Traffic Manager autopilot; a decision with ``override=False`` hands
throttle/brake back to the autopilot.

Rules (ARCHITECTURE.md 3.7, thresholds in ``control`` of configs/carla.yaml), evaluated on the
*confirmed* lead only:

1. Lead persistence. An object becomes a lead candidate only after it has been ``in_ego_lane``
   for ``lead_persist_frames`` consecutive frames under the same track id (M4.6 found
   one-frame false leads, e.g. a crossing car). The confirmed lead is the closest such object
   within ``max_lead_distance_m``.
2. AEB: ``ttc < aeb_ttc_s`` or ``distance < aeb_distance_m`` -> full brake, and the controller
   latches (``HOLD``).
3. HOLD: after an AEB the brake stays at ``hold_brake`` while a confirmed lead is closer than
   ``hold_release_distance_m``, and for ``lead_release_frames`` frames after the lead is lost
   (a detection dropout at standstill must not let the car lurch forward).
4. ACC: ``aeb_ttc_s <= ttc < acc_ttc_s`` -> brake rising linearly from ``acc_min_brake`` (at
   ``acc_ttc_s``) to ``acc_max_brake`` (at ``aeb_ttc_s``), no throttle.
5. Gap: closing on a lead inside ``gap_min_m + speed * gap_time_s`` -> brake proportional to
   the gap deficit (``gap_brake_gain``, capped at ``acc_max_brake``), no throttle.
6. Otherwise ``AUTOPILOT`` (no override).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

MODES = ("AUTOPILOT", "ACC", "GAP", "AEB", "HOLD")

REQUIRED_KEYS = (
    "lead_persist_frames",
    "lead_release_frames",
    "max_lead_distance_m",
    "aeb_ttc_s",
    "aeb_distance_m",
    "acc_ttc_s",
    "acc_min_brake",
    "acc_max_brake",
    "hold_brake",
    "hold_release_distance_m",
    "gap_min_m",
    "gap_time_s",
    "gap_brake_gain",
)


@dataclass
class LeadCandidate:
    """One tracked object as the controller sees it (from ObjectState or a ROS message)."""

    track_id: int
    distance_m: float
    ttc_s: float | None
    closing_speed_mps: float
    in_ego_lane: bool


@dataclass
class ControlDecision:
    override: bool  # False = Traffic Manager autopilot keeps throttle/brake
    throttle: float
    brake: float
    mode: str  # one of MODES
    lead: LeadCandidate | None = None
    reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def candidates_from_frame(frame: Any) -> list[LeadCandidate]:
    """``perception.types.PerceptionFrame`` -> candidates (offline replay and tests)."""
    return [
        LeadCandidate(
            track_id=int(o.track.track_id),
            distance_m=float(o.distance_m),
            ttc_s=None if o.ttc_s is None else float(o.ttc_s),
            closing_speed_mps=float(o.closing_speed_mps),
            in_ego_lane=bool(o.in_ego_lane),
        )
        for o in frame.objects
    ]


class LongitudinalController:
    """Stateful AEB/ACC rules; call ``update`` once per perception frame."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in cfg]
        if missing:
            raise KeyError(f"control config misses {missing}")
        self.cfg = dict(cfg)
        if int(cfg["lead_persist_frames"]) < 1:
            raise ValueError("lead_persist_frames must be >= 1")
        if not float(cfg["aeb_ttc_s"]) < float(cfg["acc_ttc_s"]):
            raise ValueError("aeb_ttc_s must be < acc_ttc_s")
        self.reset()

    def reset(self) -> None:
        self._streak: dict[int, int] = {}  # track id -> consecutive in-ego-lane frames
        self._latched = False
        self._frames_without_lead = 0

    @property
    def latched(self) -> bool:
        return self._latched

    def confirmed_lead(self, objects: list[LeadCandidate]) -> LeadCandidate | None:
        """Update the persistence counters and return the closest persistent in-lane object."""
        n = int(self.cfg["lead_persist_frames"])
        max_d = float(self.cfg["max_lead_distance_m"])
        streak: dict[int, int] = {}
        for o in objects:
            if o.in_ego_lane and math.isfinite(o.distance_m):
                streak[o.track_id] = self._streak.get(o.track_id, 0) + 1
        self._streak = streak  # ids absent or out of lane this frame start again from 0
        ok = [
            o
            for o in objects
            if o.in_ego_lane
            and math.isfinite(o.distance_m)
            and streak.get(o.track_id, 0) >= n
            and o.distance_m <= max_d
        ]
        return min(ok, key=lambda o: o.distance_m) if ok else None

    def update(self, objects: list[LeadCandidate], ego_speed_mps: float) -> ControlDecision:
        c = self.cfg
        lead = self.confirmed_lead(objects)
        if lead is None:
            self._frames_without_lead += 1
            if self._latched and self._frames_without_lead <= int(c["lead_release_frames"]):
                return self._brake("HOLD", float(c["hold_brake"]), None, "lead lost, holding")
            self._latched = False
            return ControlDecision(False, 0.0, 0.0, "AUTOPILOT", None, "no lead")
        self._frames_without_lead = 0
        d = float(lead.distance_m)
        ttc = lead.ttc_s
        aeb_ttc, acc_ttc = float(c["aeb_ttc_s"]), float(c["acc_ttc_s"])
        if (ttc is not None and ttc < aeb_ttc) or d < float(c["aeb_distance_m"]):
            self._latched = True
            why = f"ttc {ttc:.2f}s" if ttc is not None and ttc < aeb_ttc else f"dist {d:.1f}m"
            return self._brake("AEB", 1.0, lead, why)
        if self._latched:
            if d < float(c["hold_release_distance_m"]):
                return self._brake("HOLD", float(c["hold_brake"]), lead, f"dist {d:.1f}m")
            self._latched = False
        if ttc is not None and ttc < acc_ttc:
            frac = (acc_ttc - ttc) / (acc_ttc - aeb_ttc)  # 0 at acc_ttc_s, 1 at aeb_ttc_s
            lo, hi = float(c["acc_min_brake"]), float(c["acc_max_brake"])
            return self._brake("ACC", lo + (hi - lo) * frac, lead, f"ttc {ttc:.2f}s")
        gap = float(c["gap_min_m"]) + max(0.0, float(ego_speed_mps)) * float(c["gap_time_s"])
        if d < gap and lead.closing_speed_mps > 0.0:
            b = min(float(c["acc_max_brake"]), float(c["gap_brake_gain"]) * (gap - d) / gap)
            return self._brake("GAP", b, lead, f"gap {d:.1f}<{gap:.1f}m")
        return ControlDecision(False, 0.0, 0.0, "AUTOPILOT", lead, "lead far")

    @staticmethod
    def _brake(mode: str, brake: float, lead: LeadCandidate | None, reason: str) -> ControlDecision:
        return ControlDecision(True, 0.0, float(min(1.0, max(0.0, brake))), mode, lead, reason)
