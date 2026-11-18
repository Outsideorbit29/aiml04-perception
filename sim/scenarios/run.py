"""Run one demo scenario next to the ROS2 stack (M5.5) and log min gap + collision.

    # WSL2, repo root, with CARLA (Windows) and the stack running (the launch file does this):
    python3 -m sim.scenarios.run --scenario stopped_car [--host localhost] [--port 2000]

The script is a second CARLA client: it waits for the ego (role ``ego.role_name``) spawned by
carla_bridge_node, spawns the scenario's actors relative to it, attaches a collision sensor to
the ego and follows the episode tick by tick (the bridge ticks). The episode ends when the
ego has stopped (``scenario.stop_hold_s`` below ``stop_speed_mps`` after reaching
``arm_speed_mps``), on a collision, or after ``scenario.timeout_s`` sim seconds. One JSON line
(outcome, min gap, collision, ...) is appended to ``scenario.results``; the scenario's actors
are destroyed and the weather restored in ``finally``. Exit code 0 only for a stop without
collision.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.carla_utils import (  # noqa: E402
    connect,
    destroy_actors,
    find_actor_by_role,
    load_carla_config,
    resolve_path,
)
from sim.scenarios import registry, scenario_config  # noqa: E402
from sim.scenarios.common import EpisodeMonitor, actor_gap, ego_speed  # noqa: E402


def append_result(path: str | Path, record: dict[str, Any]) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--config", default="configs/carla.yaml")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--results", help="JSONL (default scenario.results)")
    args = ap.parse_args(argv)

    import carla

    cfg = load_carla_config(args.config)
    s = cfg["scenario"]
    scls = registry()[args.scenario]
    scfg = scenario_config(cfg, args.scenario)
    c = cfg["carla"]
    client = connect(args.host or c["host"], args.port or int(c["port"]), float(c["timeout_s"]))
    world = client.get_world()
    print(f"[scenario] {args.scenario}: waiting for the ego ({cfg['ego']['role_name']})",
          flush=True)
    ego = find_actor_by_role(world, cfg["ego"]["role_name"], float(s["ego_wait_s"]))
    if ego is None:
        print("[scenario] no ego vehicle: is carla_bridge_node running?", flush=True)
        return 2
    world = client.get_world()  # the bridge may have loaded another map meanwhile
    mon = EpisodeMonitor(
        float(s["arm_speed_mps"]), float(s["stop_speed_mps"]), float(s["stop_hold_s"]),
        float(s["timeout_s"]),
    )
    sc = scls(world, ego, scfg)
    collision_sensor = None
    # sim ticks pause while the bridge waits for the first control (model loading)
    idle_s = float(cfg["bridge"]["first_control_timeout_s"]) + 10.0
    try:
        sc.setup()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        collision_sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego)
        collision_sensor.listen(lambda ev: mon.on_collision(ev.other_actor.type_id))
        print(f"[scenario] {args.scenario} set up: {sc.describe()}", flush=True)
        t_prev = None
        last_tick = time.monotonic()
        while True:
            try:
                snap = world.wait_for_tick(2.0)
            except RuntimeError:
                if time.monotonic() - last_tick > idle_s:
                    print("[scenario] no ticks: is the bridge stuck?", flush=True)
                    break
                continue
            last_tick = time.monotonic()
            t = snap.timestamp.elapsed_seconds
            dt = snap.timestamp.delta_seconds if t_prev is None else t - t_prev
            t_prev = t
            sc.step(t, dt)
            gap = actor_gap(ego, sc.hazard, float(s["in_path_margin_m"])) if sc.hazard else None
            if mon.update(t, ego_speed(ego), gap):
                break
    except KeyboardInterrupt:
        print("[scenario] interrupted", flush=True)
    finally:
        sc.cleanup()
        actors = list(sc.actors) + ([collision_sensor] if collision_sensor is not None else [])
        n = destroy_actors(client, actors)
        res = {
            "scenario": args.scenario,
            "town": world.get_map().name.split("/")[-1],
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **mon.result(),
            "config": scfg,
            "control": {k: cfg["control"][k] for k in (
                "lead_persist_frames", "aeb_ttc_s", "aeb_distance_m", "acc_ttc_s")},
            "desired_speed_kmh": cfg["traffic_manager"]["desired_speed_kmh"],
        }
        path = append_result(args.results or s["results"], res)
        print(f"[scenario] destroyed {n} actors; result -> {path}", flush=True)
        print("[scenario] " + json.dumps({k: res[k] for k in (
            "scenario", "outcome", "success", "collision", "min_gap_m", "stop_gap_m",
            "max_speed_mps", "sim_duration_s")}), flush=True)
    return 0 if res["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
