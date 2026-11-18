"""Shared scenario machinery (M5.5): base class, lane-relative placement, kinematic movers and
the episode monitor (min gap, collision, stop detection). ``carla`` is imported lazily, so the
pure parts (``EpisodeMonitor``, ``crossing_state``) are unit-tested in CI.

Scenarios run as a second CARLA client next to carla_bridge_node, which owns synchronous mode
and ticks the world; a scenario only reacts to ticks (``world.wait_for_tick``). Every actor a
scenario spawns goes into ``self.actors`` and is destroyed by sim/scenarios/run.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sim.carla_utils import longitudinal_gap


# ---------------------------------------------------------------------------- pure logic
@dataclass
class EpisodeMonitor:
    """Tracks one scenario episode from per-tick measurements (no CARLA objects).

    ``update(t, speed, gap)``: sim time (s), ego forward speed (m/s), bumper gap to the hazard
    when it is in the ego path (None otherwise). The episode is ``stopped`` once the ego, after
    having reached ``arm_speed_mps``, stays below ``stop_speed_mps`` for ``stop_hold_s``.
    ``update`` then keeps returning None for ``post_stop_s`` more seconds (demo footage of the
    settled car; ``stop_gap_m`` is the gap at the stop, ``final_gap_m`` the gap at the end). A
    collision in that tail still ends the episode as ``collision``.
    """

    arm_speed_mps: float
    stop_speed_mps: float
    stop_hold_s: float
    timeout_s: float
    post_stop_s: float = 0.0
    t0: float | None = None
    t: float = 0.0
    frames: int = 0
    armed: bool = False
    max_speed_mps: float = 0.0
    min_gap_m: float | None = None
    stop_gap_m: float | None = None
    collision: bool = False
    collision_with: str | None = None
    collision_speed_mps: float | None = None
    _slow_since: float | None = None
    _last_gap: float | None = None
    _last_speed: float = 0.0
    _stop_t: float | None = None
    post_stop_max_speed_mps: float | None = None
    outcome: str | None = None  # stopped | collision | timeout | aborted
    events: list[str] = field(default_factory=list)

    def update(self, t: float, speed_mps: float, gap_m: float | None) -> str | None:
        if self.t0 is None:
            self.t0 = t
        self.t = t
        self.frames += 1
        self._last_speed = speed_mps
        self.max_speed_mps = max(self.max_speed_mps, speed_mps)
        if gap_m is not None:
            self._last_gap = gap_m
            self.min_gap_m = gap_m if self.min_gap_m is None else min(self.min_gap_m, gap_m)
        if not self.armed and speed_mps >= self.arm_speed_mps:
            self.armed = True
            self.events.append(f"armed at t={t - self.t0:.2f}s")
        if self.outcome is not None:
            return self._finished(t, speed_mps)
        if self.armed and speed_mps < self.stop_speed_mps:
            if self._slow_since is None:
                self._slow_since = t
            if t - self._slow_since >= self.stop_hold_s:
                self.stop_gap_m = self._last_gap
                self.outcome = "stopped"
                self._stop_t = t
                self.events.append(f"stopped at t={t - self.t0:.2f}s")
                return self._finished(t, speed_mps)
        else:
            self._slow_since = None
        if self.outcome is None and t - self.t0 >= self.timeout_s:
            self.outcome = "timeout"
        return self.outcome

    def _finished(self, t: float, speed_mps: float) -> str | None:
        """The outcome, or None while the post-stop tail of a ``stopped`` episode runs."""
        if self.outcome == "stopped" and self._stop_t is not None:
            prev = self.post_stop_max_speed_mps
            self.post_stop_max_speed_mps = speed_mps if prev is None else max(prev, speed_mps)
            if t - self._stop_t < self.post_stop_s:
                return None
        return self.outcome

    def on_collision(self, other: str) -> None:
        if not self.collision:
            self.collision = True
            self.collision_with = other
            self.collision_speed_mps = self._last_speed
            self.events.append(f"collision with {other} at {self._last_speed:.1f} m/s")
        self.outcome = "collision"

    def result(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome or "aborted",
            "success": self.outcome == "stopped" and not self.collision,
            "collision": self.collision,
            "collision_with": self.collision_with,
            "collision_speed_mps": _r(self.collision_speed_mps),
            "min_gap_m": _r(self.min_gap_m),
            "stop_gap_m": _r(self.stop_gap_m),
            "final_gap_m": _r(self._last_gap),
            "post_stop_s": _r(self.post_stop_s),
            "post_stop_max_speed_mps": _r(self.post_stop_max_speed_mps),
            "max_speed_mps": _r(self.max_speed_mps),
            "sim_duration_s": _r(self.t - self.t0) if self.t0 is not None else 0.0,
            "frames": self.frames,
            "events": list(self.events),
        }


def _r(v: float | None, nd: int = 3) -> float | None:
    return None if v is None else round(float(v), nd)


def crossing_state(
    state: str, lateral_m: float, t: float, t_state: float, pause_s: float, end_offset_m: float
) -> str:
    """Pedestrian crossing state machine: walk_in -> pause (at the lane centre) -> walk_out ->
    done. ``lateral_m`` is the walker's offset right of the lane centre (decreasing while it
    crosses to the left)."""
    if state == "walk_in" and lateral_m <= 0.0:
        return "pause" if pause_s > 0 else "walk_out"
    if state == "pause" and t - t_state >= pause_s:
        return "walk_out"
    if state == "walk_out" and lateral_m <= end_offset_m:
        return "done"
    return state


# ---------------------------------------------------------------------------- carla helpers
def lane_waypoint(world: Any, location: Any) -> Any:
    import carla

    return world.get_map().get_waypoint(
        location, project_to_road=True, lane_type=carla.LaneType.Driving
    )


def waypoint_ahead(world: Any, ego: Any, distance_m: float) -> Any:
    """Waypoint ``distance_m`` ahead of the ego along its lane (TM route may differ at
    junctions; the bridge picks a spawn point with a long straight road)."""
    wp = lane_waypoint(world, ego.get_location())
    nxt = wp.next(float(distance_m))
    if not nxt:
        raise RuntimeError(f"no lane {distance_m} m ahead of the ego")
    return nxt[0]


def offset_transform(tf: Any, lateral_m: float, z_m: float = 0.0, yaw_deg: float = 0.0) -> Any:
    """``tf`` moved ``lateral_m`` to its right (negative = left) and ``z_m`` up."""
    import carla

    r = tf.get_right_vector()
    loc = carla.Location(
        x=tf.location.x + r.x * lateral_m, y=tf.location.y + r.y * lateral_m,
        z=tf.location.z + z_m,
    )
    rot = carla.Rotation(pitch=0.0, yaw=tf.rotation.yaw + yaw_deg, roll=0.0)
    return carla.Transform(loc, rot)


def spawn(world: Any, blueprint: str, tf: Any, role: str = "scenario") -> Any:
    """Spawn ``blueprint`` at ``tf``, lifting it a little if the spot collides with the road."""
    import carla

    bp = world.get_blueprint_library().find(blueprint)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role)
    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")
    for dz in (0.3, 0.8, 1.5):
        t = carla.Transform(
            carla.Location(tf.location.x, tf.location.y, tf.location.z + dz), tf.rotation
        )
        actor = world.try_spawn_actor(bp, t)
        if actor is not None:
            return actor
    raise RuntimeError(f"could not spawn {blueprint} at {tf.location}")


def lateral_offset(tf: Any, location: Any) -> float:
    """Signed offset of ``location`` to the right of ``tf`` (m)."""
    r = tf.get_right_vector()
    return (location.x - tf.location.x) * r.x + (location.y - tf.location.y) * r.y


def front_distance(ego: Any, location: Any) -> float:
    """Distance from the ego front bumper to ``location`` along the ego heading (m)."""
    tf = ego.get_transform()
    f = tf.get_forward_vector()
    lon = (location.x - tf.location.x) * f.x + (location.y - tf.location.y) * f.y
    return lon - ego.bounding_box.extent.x


def ego_speed(ego: Any) -> float:
    v = ego.get_velocity()
    yaw = math.radians(ego.get_transform().rotation.yaw)
    return v.x * math.cos(yaw) + v.y * math.sin(yaw)


def actor_gap(ego: Any, actor: Any, margin_m: float) -> float | None:
    """Bumper gap from the ego to ``actor`` if it is in the ego path (sim.carla_utils)."""
    et, at = ego.get_transform(), actor.get_transform()
    ee, ae = ego.bounding_box.extent, actor.bounding_box.extent
    return longitudinal_gap(
        (et.location.x, et.location.y), et.rotation.yaw, ee.x, ee.y,
        (at.location.x, at.location.y), ae.x, ae.y, margin_m,
    )


class LaneFollower:
    """Kinematic path along a lane: ``advance(ds)`` then ``transform(lateral)``."""

    def __init__(self, wp: Any) -> None:
        self.wp = wp

    def advance(self, ds: float) -> None:
        if ds > 1e-3:
            nxt = self.wp.next(ds)
            if nxt:
                self.wp = nxt[0]

    def transform(self, lateral_m: float, z_m: float) -> Any:
        return offset_transform(self.wp.transform, lateral_m, z_m)


# ---------------------------------------------------------------------------- base class
class Scenario:
    """One scripted hazard. Subclasses implement ``setup`` and optionally ``step``."""

    name = "base"

    def __init__(self, world: Any, ego: Any, cfg: dict[str, Any]) -> None:
        self.world = world
        self.ego = ego
        self.cfg = cfg
        self.actors: list[Any] = []
        self.hazard: Any = None  # actor whose gap to the ego is measured

    def setup(self) -> None:
        raise NotImplementedError

    def step(self, t: float, dt: float) -> None:  # noqa: B027 - optional hook
        """Called once per sim tick with sim time ``t`` and step ``dt`` (seconds)."""

    def cleanup(self) -> None:  # noqa: B027 - optional hook
        """Undo world changes other than actors (weather); actors are destroyed by run.py."""

    def describe(self) -> dict[str, Any]:
        return {"scenario": self.name, **{k: v for k, v in self.cfg.items() if k != "base"}}
