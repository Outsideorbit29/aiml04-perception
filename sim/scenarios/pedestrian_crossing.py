"""Pedestrian crossing: a walker waits ``start_offset_m`` right of the ego lane centre,
``distance_m`` ahead. When the ego front is ``trigger_distance_m`` from the crossing point it
walks left across the lane, stands in the lane centre for ``pause_in_lane_s`` and walks on to
``end_offset_m``."""

from __future__ import annotations

from sim.scenarios.common import (
    Scenario,
    crossing_state,
    front_distance,
    lateral_offset,
    offset_transform,
    spawn,
    waypoint_ahead,
)


class PedestrianCrossing(Scenario):
    name = "pedestrian_crossing"

    def setup(self) -> None:
        wp = waypoint_ahead(self.world, self.ego, float(self.cfg["distance_m"]))
        self.cross_tf = wp.transform
        start = offset_transform(wp.transform, float(self.cfg["start_offset_m"]), 1.0, -90.0)
        self.walker = spawn(self.world, self.cfg["blueprint"], start)
        self.actors.append(self.walker)
        self.hazard = self.walker
        self.state = "wait"
        self.t_state = 0.0

    def _walk(self, speed: float) -> None:
        import carla

        r = self.cross_tf.get_right_vector()
        d = carla.Vector3D(-r.x, -r.y, 0.0)  # towards the left side of the lane
        self.walker.apply_control(carla.WalkerControl(direction=d, speed=float(speed)))

    def step(self, t: float, dt: float) -> None:
        c = self.cfg
        if self.state == "wait":
            if front_distance(self.ego, self.cross_tf.location) <= float(c["trigger_distance_m"]):
                self.state, self.t_state = "walk_in", t
                self._walk(float(c["speed_mps"]))
            return
        lat = lateral_offset(self.cross_tf, self.walker.get_location())
        new = crossing_state(
            self.state, lat, t, self.t_state, float(c["pause_in_lane_s"]),
            float(c["end_offset_m"]),
        )
        if new != self.state:
            self.state, self.t_state = new, t
            self._walk(float(c["speed_mps"]) if new == "walk_out" else 0.0)
