"""Cut-in: an NPC waits in the adjacent lane ``distance_m`` ahead (``lateral_offset_m`` left of
the ego lane centre). When the ego front is ``trigger_distance_m`` from its rear it drives off
at ``speed_mps``, moves into the ego lane over ``cut_in_duration_s`` and then brakes to a stop
at ``decel_mps2``. The NPC is moved kinematically (physics off), so the run is repeatable;
Town01/02 have one lane per direction, so the "adjacent lane" is the oncoming one, scripted
in the ego's direction."""

from __future__ import annotations

import math

from sim.scenarios.common import LaneFollower, Scenario, front_distance, spawn, waypoint_ahead


def cut_in_profile(
    t_since: float, v0: float, duration_s: float, offset_m: float, decel: float
) -> tuple[float, float]:
    """(speed m/s, lateral offset m) ``t_since`` seconds after the trigger. The offset goes
    smoothly (cosine) from ``offset_m`` to 0 over ``duration_s``; afterwards the speed falls
    at ``decel`` to 0."""
    if t_since < duration_s:
        frac = 0.5 - 0.5 * math.cos(math.pi * t_since / duration_s)
        return v0, offset_m * (1.0 - frac)
    return max(0.0, v0 - decel * (t_since - duration_s)), 0.0


class CutIn(Scenario):
    name = "cut_in"

    def setup(self) -> None:
        wp = waypoint_ahead(self.world, self.ego, float(self.cfg["distance_m"]))
        self.path = LaneFollower(wp)
        self.offset = -abs(float(self.cfg["lateral_offset_m"]))  # left of the ego lane
        npc = spawn(self.world, self.cfg["blueprint"], self.path.transform(self.offset, 0.3))
        npc.set_simulate_physics(False)
        npc.set_transform(self.path.transform(self.offset, 0.0))
        self.actors.append(npc)
        self.hazard = self.npc = npc
        self.t_trigger: float | None = None

    def step(self, t: float, dt: float) -> None:
        c = self.cfg
        if self.t_trigger is None:
            rear = self.npc.get_location()
            gap = front_distance(self.ego, rear) - self.npc.bounding_box.extent.x
            if gap > float(c["trigger_distance_m"]):
                return
            self.t_trigger = t
        v, lat = cut_in_profile(
            t - self.t_trigger, float(c["speed_mps"]), float(c["cut_in_duration_s"]),
            self.offset, float(c["decel_mps2"]),
        )
        self.path.advance(v * dt)
        self.npc.set_transform(self.path.transform(lat, 0.0))
