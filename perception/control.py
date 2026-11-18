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
4. ACC: entered when ``ttc < acc_ttc_s``. Brake rises linearly from ``acc_min_brake`` (at
   ``acc_ttc_s``) to ``acc_max_brake`` (at ``aeb_ttc_s``), no throttle. Inside the hysteresis
   band (``acc_ttc_s <= ttc < acc_exit_ttc_s``, or no TTC) the car coasts (override, throttle 0,
   brake 0 unless the gap rule asks for more) instead of handing back to the autopilot.
5. Gap: entered when closing on a lead inside ``gap_min_m + speed * gap_time_s``. Brake is
   proportional to the gap deficit (``gap_brake_gain``, capped at ``acc_max_brake``), no
   throttle. GAP escalates to ACC at once when the ACC entry rule fires; ACC never steps down
   to GAP (one engaged state, fewer mode flips).
6. Leaving ACC/GAP (hysteresis + dwell). Back to ``AUTOPILOT`` only when, for
   ``exit_confirm_frames`` consecutive frames, the TTC is clear (none or
   ``>= acc_exit_ttc_s``), the gap is clear (``d >= desired gap + gap_exit_margin_m`` or the lead
   opens faster than ``gap_exit_opening_mps``), the standstill guard is off and the lead is not
   *tracked* (rule 7), and the mode has lasted ``min_engaged_frames``. A lead dropout keeps the
   mode and the last brake for ``lead_lost_exit_frames`` frames.
7. Lead tracking (no early exit). A lead within ``follow_range_m`` that is stationary or closing
   (not opening faster than ``gap_exit_opening_mps``) keeps ACC/GAP engaged: the autopilot
   would accelerate towards it at cruise speed and ACC would re-engage (the 2026-11-19 Session B
   ACC -> AUTOPILOT -> ACC/GAP stop-and-go).
8. Approach (controlled creep). Engaged, with no brake requested or still applied, the ego
   slower than ``approach_speed_mps`` and the lead at least ``approach_margin_m`` beyond the
   desired gap, the controller creeps: throttle ``approach_throttle * (1 - v /
   approach_speed_mps)``. The ego therefore never exceeds the approach speed under ACC/GAP and
   closes up to the gap, where the gap brake and the standstill guard stop it.
9. Standstill guard (soft HOLD). In ACC/GAP, below ``standstill_speed_mps`` with the lead closer
   than ``standstill_hold_distance_m``, the brake is at least ``standstill_brake``, there is no
   throttle and the mode cannot exit. Beyond that distance (but inside ``follow_range_m``) a
   stopped ego creeps closer (rule 8) instead.
10. The ACC/GAP brake changes by at most ``brake_rate_up`` / ``brake_rate_down`` per frame. AEB
    (full brake) and HOLD are applied at once; entry into AEB has no dwell or hysteresis.
11. Otherwise ``AUTOPILOT`` (no override).
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
    "acc_exit_ttc_s",
    "gap_exit_margin_m",
    "gap_exit_opening_mps",
    "min_engaged_frames",
    "exit_confirm_frames",
    "lead_lost_exit_frames",
    "standstill_speed_mps",
    "standstill_hold_distance_m",
    "standstill_brake",
    "follow_range_m",
    "approach_speed_mps",
    "approach_throttle",
    "approach_margin_m",
    "brake_rate_up",
    "brake_rate_down",
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
        if float(cfg["acc_exit_ttc_s"]) < float(cfg["acc_ttc_s"]):
            raise ValueError("acc_exit_ttc_s must be >= acc_ttc_s")
        if float(cfg["brake_rate_up"]) <= 0 or float(cfg["brake_rate_down"]) <= 0:
            raise ValueError("brake_rate_up and brake_rate_down must be > 0")
        if float(cfg["standstill_hold_distance_m"]) < float(cfg["gap_min_m"]):
            raise ValueError("standstill_hold_distance_m must be >= gap_min_m (no creep into gap)")
        thr, v_app = float(cfg["approach_throttle"]), float(cfg["approach_speed_mps"])
        if not 0.0 <= thr <= 1.0 or v_app < 0.0:
            raise ValueError("approach_throttle must be in [0, 1] and approach_speed_mps >= 0")
        self.reset()

    def reset(self) -> None:
        self._streak: dict[int, int] = {}  # track id -> consecutive in-ego-lane frames
        self._latched = False
        self._frames_without_lead = 0
        self._mode = "AUTOPILOT"  # engaged state of the ACC/GAP rules (AEB/HOLD use the latch)
        self._mode_frames = 0  # frames spent in self._mode
        self._clear_frames = 0  # consecutive frames the ACC/GAP exit condition held
        self._prev_brake = 0.0  # last published brake (rate limiter)

    @property
    def mode(self) -> str:
        """Engaged ACC/GAP state (``AUTOPILOT`` when not engaged or while AEB/HOLD act)."""
        return self._mode

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
                return self._fixed("HOLD", float(c["hold_brake"]), None, "lead lost, holding")
            self._latched = False
            if self._mode != "AUTOPILOT":
                self._mode_frames += 1
                if self._frames_without_lead <= int(c["lead_lost_exit_frames"]):
                    return self._fixed(self._mode, self._prev_brake, None, "lead lost, keeping")
            return self._autopilot(None, "no lead")
        self._frames_without_lead = 0
        d = float(lead.distance_m)
        ttc = lead.ttc_s
        v = max(0.0, float(ego_speed_mps))
        aeb_ttc, acc_ttc = float(c["aeb_ttc_s"]), float(c["acc_ttc_s"])
        # AEB first, without any dwell or hysteresis
        if (ttc is not None and ttc < aeb_ttc) or d < float(c["aeb_distance_m"]):
            self._latched = True
            self._set_mode("AUTOPILOT")  # the latch takes over; ACC/GAP restart after release
            why = f"ttc {ttc:.2f}s" if ttc is not None and ttc < aeb_ttc else f"dist {d:.1f}m"
            return self._fixed("AEB", 1.0, lead, why)
        if self._latched:
            if d < float(c["hold_release_distance_m"]):
                return self._fixed("HOLD", float(c["hold_brake"]), lead, f"dist {d:.1f}m")
            self._latched = False

        acc_on = ttc is not None and ttc < acc_ttc
        acc_brake = 0.0
        if acc_on:
            frac = (acc_ttc - ttc) / (acc_ttc - aeb_ttc)  # 0 at acc_ttc_s, 1 at aeb_ttc_s
            lo, hi = float(c["acc_min_brake"]), float(c["acc_max_brake"])
            acc_brake = lo + (hi - lo) * frac
        gap = float(c["gap_min_m"]) + v * float(c["gap_time_s"])
        gap_on = d < gap and lead.closing_speed_mps > 0.0
        gap_brake = 0.0
        if gap_on:
            gap_brake = min(float(c["acc_max_brake"]), float(c["gap_brake_gain"]) * (gap - d) / gap)

        # entries are immediate; GAP -> ACC is an escalation, ACC never steps down to GAP
        if acc_on and self._mode != "ACC":
            self._set_mode("ACC")
        elif gap_on and self._mode == "AUTOPILOT":
            self._set_mode("GAP")
        if self._mode == "AUTOPILOT":
            return self._autopilot(lead, "lead far")
        self._mode_frames += 1

        standstill = v < float(c["standstill_speed_mps"]) and d < float(
            c["standstill_hold_distance_m"]
        )
        opening = lead.closing_speed_mps < -float(c["gap_exit_opening_mps"])
        tracked = d <= float(c["follow_range_m"]) and not opening
        ttc_clear = ttc is None or ttc >= float(c["acc_exit_ttc_s"])
        gap_clear = d >= gap + float(c["gap_exit_margin_m"]) or opening
        clear = ttc_clear and gap_clear and not standstill and not tracked
        self._clear_frames = self._clear_frames + 1 if clear else 0
        if self._clear_frames >= int(c["exit_confirm_frames"]) and self._mode_frames >= int(
            c["min_engaged_frames"]
        ):
            return self._autopilot(lead, "lead clear")

        target = max(acc_brake, gap_brake)
        if acc_on:
            why = f"ttc {ttc:.2f}s"
        elif gap_on:
            why = f"gap {d:.1f}<{gap:.1f}m"
        else:
            why = f"coast, d {d:.1f}m"
        if standstill and target < float(c["standstill_brake"]):
            target = float(c["standstill_brake"])
            why = f"standstill hold, d {d:.1f}m"
        out = self._ramped(self._mode, target, lead, why)
        v_app = float(c["approach_speed_mps"])
        if (
            out.brake == 0.0
            and not standstill
            and v < v_app
            and d >= gap + float(c["approach_margin_m"])
        ):
            out.throttle = float(c["approach_throttle"]) * (1.0 - v / v_app)
            out.reason = f"approach, d {d:.1f}m"
        return out

    def _set_mode(self, mode: str) -> None:
        if mode != self._mode:
            self._mode = mode
            self._mode_frames = 0
            self._clear_frames = 0

    def _autopilot(self, lead: LeadCandidate | None, reason: str) -> ControlDecision:
        self._set_mode("AUTOPILOT")
        self._prev_brake = 0.0
        return ControlDecision(False, 0.0, 0.0, "AUTOPILOT", lead, reason)

    def _fixed(
        self, mode: str, brake: float, lead: LeadCandidate | None, reason: str
    ) -> ControlDecision:
        """Brake applied as is (AEB, HOLD, the ACC/GAP dropout hold)."""
        out = self._brake(mode, brake, lead, reason)
        self._prev_brake = out.brake
        return out

    def _ramped(
        self, mode: str, target: float, lead: LeadCandidate | None, reason: str
    ) -> ControlDecision:
        """ACC/GAP brake moved towards ``target`` by at most the per-frame rate limits."""
        prev = self._prev_brake
        up, down = float(self.cfg["brake_rate_up"]), float(self.cfg["brake_rate_down"])
        brake = min(target, prev + up) if target >= prev else max(target, prev - down)
        out = self._brake(mode, brake, lead, reason)
        out.extra["target_brake"] = float(target)
        self._prev_brake = out.brake
        return out

    @staticmethod
    def _brake(mode: str, brake: float, lead: LeadCandidate | None, reason: str) -> ControlDecision:
        return ControlDecision(True, 0.0, float(min(1.0, max(0.0, brake))), mode, lead, reason)
