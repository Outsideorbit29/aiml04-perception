"""Stopped car: a parked vehicle (hand brake on) in the ego lane ``distance_m`` ahead."""

from __future__ import annotations

from sim.scenarios.common import Scenario, spawn, waypoint_ahead


class StoppedCar(Scenario):
    name = "stopped_car"

    def setup(self) -> None:
        import carla

        wp = waypoint_ahead(self.world, self.ego, float(self.cfg["distance_m"]))
        npc = spawn(self.world, self.cfg["blueprint"], wp.transform)
        self.actors.append(npc)
        npc.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        self.hazard = npc
