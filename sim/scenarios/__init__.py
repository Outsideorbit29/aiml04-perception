"""Scripted demo scenarios (M5.5). Run one next to the ROS2 stack with
``python3 -m sim.scenarios.run --scenario <name>`` (stack.launch.py scenario:=<name> does it).

Importing this package never imports ``carla``; the scenario classes import it lazily.
"""

from __future__ import annotations

from typing import Any


def registry() -> dict[str, Any]:
    from sim.scenarios.cut_in import CutIn
    from sim.scenarios.night_rain import NightRain
    from sim.scenarios.pedestrian_crossing import PedestrianCrossing
    from sim.scenarios.stopped_car import StoppedCar

    return {c.name: c for c in (StoppedCar, PedestrianCrossing, CutIn, NightRain)}


def scenario_config(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """``scenarios.<name>`` from configs/carla.yaml, merged over ``scenarios.<base>`` if set."""
    all_cfg = cfg["scenarios"]
    if name not in all_cfg:
        raise KeyError(f"scenario {name!r} not in configs/carla.yaml; have {sorted(all_cfg)}")
    own = dict(all_cfg[name])
    base = own.pop("base", None)
    return {**scenario_config(cfg, base), **own} if base else own
