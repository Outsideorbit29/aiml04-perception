"""Record CARLA frames with ground truth for the sim eval / fine-tune on Kaggle (M5.6, Session A).

    # Windows: D:\\CARLA_0.9.15\\CarlaUE4.exe -quality-level=Low -RenderOffScreen -fps=20
    # WSL2 (repo root), ~15 min for 3000 frames:
    python3 sim/record_dataset.py --town Town01 --load-town [--frames 3000] [--out <dir>]
    # then (Windows): kaggle datasets create -p runs/carla_frames   (dataset aiml04-carla-frames)

Synchronous mode at ``carla.fixed_delta_seconds``; the ego drives on the normal Traffic Manager
autopilot (it obeys traffic, unlike the scenario runs) among ``dataset.n_vehicles`` NPC cars and
``dataset.n_walkers`` pedestrians; the weather cycles through ``dataset.weathers``. Every
``every_n_ticks`` ticks it writes:

    images/<town>_<idx>.jpg        ego RGB (800x450, same camera as the live stack)
    labels/<town>_<idx>.txt        YOLO boxes, BDD class ids (configs/data_bdd.yaml)
    gt/<town>_<idx>.json           boxes + GT distance (nearest bbox vertex depth), visibility
    depth/<town>_<idx>.png         GT depth, uint16 centimetres (every depth_every_n_saved-th)
    meta_<town>.json               camera intrinsics, classes, weather per frame range

All actors are destroyed and the world settings restored in ``finally``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.carla_utils import (  # noqa: E402
    SensorQueue,
    bgra_to_bgr,
    camera_blueprint,
    camera_intrinsics,
    camera_transform,
    connect,
    decode_depth,
    destroy_actors,
    ensure_town,
    load_carla_config,
    resolve_path,
    restore_settings,
    set_synchronous,
    spawn_ego,
    weather_preset,
)
from sim.gt_labels import BASE_TYPE_TO_BDD, label_objects, yolo_line  # noqa: E402


def bdd_class_ids() -> dict[str, int]:
    names = yaml.safe_load(resolve_path("configs/data_bdd.yaml").read_text("utf-8"))["names"]
    return {str(v): int(k) for k, v in names.items()}


def spawn_traffic(client: Any, world: Any, tm_port: int, n_vehicles: int, n_walkers: int,
                  rng: random.Random) -> tuple[list, list]:
    """NPC vehicles on autopilot + walkers with AI controllers. Returns (actors, controllers)."""
    import carla

    bp_lib = world.get_blueprint_library()
    vbps = [b for b in bp_lib.filter("vehicle.*")
            if b.has_attribute("base_type") and b.get_attribute("base_type").as_str()]
    points = world.get_map().get_spawn_points()
    rng.shuffle(points)
    cmds = []
    for tf in points[:n_vehicles]:
        bp = rng.choice(vbps)
        bp.set_attribute("role_name", "npc")
        cmds.append(carla.command.SpawnActor(bp, tf).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)))
    actors = [world.get_actor(r.actor_id) for r in client.apply_batch_sync(cmds, True)
              if not r.error]
    wbps = list(bp_lib.filter("walker.pedestrian.*"))
    wcmds = []
    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            bp = rng.choice(wbps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            wcmds.append(carla.command.SpawnActor(bp, carla.Transform(loc)))
    walkers = [r.actor_id for r in client.apply_batch_sync(wcmds, True) if not r.error]
    cbp = bp_lib.find("controller.ai.walker")
    ccmds = [carla.command.SpawnActor(cbp, carla.Transform(), w) for w in walkers]
    ctrl_ids = [r.actor_id for r in client.apply_batch_sync(ccmds, True) if not r.error]
    world.tick()
    controllers = [world.get_actor(i) for i in ctrl_ids]
    world.set_pedestrians_cross_factor(0.2)
    for c in controllers:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(1.4)
    actors += [world.get_actor(i) for i in walkers]
    return [a for a in actors if a is not None], [c for c in controllers if c is not None]


def scene_objects(world: Any, ego: Any, max_dist: float) -> list[dict[str, Any]]:
    """Vehicles and walkers near the ego with BDD class and world bbox vertices."""
    ego_loc = ego.get_location()
    out = []
    for a in world.get_actors():
        if a.id == ego.id:
            continue
        if a.type_id.startswith("vehicle."):
            cls = BASE_TYPE_TO_BDD.get(a.attributes.get("base_type", "car").lower(), "car")
        elif a.type_id.startswith("walker.pedestrian"):
            cls = "person"
        else:
            continue
        if a.get_location().distance(ego_loc) > max_dist + 10.0:
            continue
        verts = a.bounding_box.get_world_vertices(a.get_transform())
        out.append({"id": a.id, "cls_name": cls,
                    "vertices": np.array([[v.x, v.y, v.z] for v in verts])})
    return out


def main(argv: list[str] | None = None) -> int:
    cfg = load_carla_config()
    c, d = cfg["carla"], cfg["dataset"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=c["host"])
    ap.add_argument("--port", type=int, default=int(c["port"]))
    ap.add_argument("--town", default=c["town"])
    ap.add_argument("--load-town", action="store_true")
    ap.add_argument("--frames", type=int, default=int(d["frames"]))
    ap.add_argument("--out", default=d["out_dir"])
    args = ap.parse_args(argv)

    import cv2

    out = resolve_path(args.out)
    for sub in ("images", "labels", "gt", "depth"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cam_cfg = cfg["camera"]
    intr = camera_intrinsics(cam_cfg["width"], cam_cfg["height"], cam_cfg["fov_deg"])
    cls_ids = bdd_class_ids()
    rng = random.Random(int(c["seed"]))

    client = connect(args.host, args.port, float(c["timeout_s"]))
    world = ensure_town(client, args.town, args.load_town)
    town = world.get_map().name.split("/")[-1]
    original = None
    tm = None
    actors: list = []
    controllers: list = []
    weather0 = world.get_weather()
    saved = 0
    weather_log: list[dict[str, Any]] = []
    t0 = time.monotonic()
    try:
        original = set_synchronous(world, float(c["fixed_delta_seconds"]))
        tm = client.get_trafficmanager(int(cfg["traffic_manager"]["port"]))
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(int(c["seed"]))
        ego = spawn_ego(world, cfg["ego"])
        actors.append(ego)
        bp_lib = world.get_blueprint_library()
        rgb = world.spawn_actor(camera_blueprint(bp_lib, cam_cfg, "rgb"),
                                camera_transform(cam_cfg), attach_to=ego)
        dep = world.spawn_actor(camera_blueprint(bp_lib, cam_cfg, "depth"),
                                camera_transform(cam_cfg), attach_to=ego)
        actors += [rgb, dep]
        q_rgb, q_dep = SensorQueue(), SensorQueue()
        rgb.listen(q_rgb)
        dep.listen(q_dep)
        npcs, controllers = spawn_traffic(client, world, tm.get_port(), int(d["n_vehicles"]),
                                          int(d["n_walkers"]), rng)
        actors += npcs
        ego.set_autopilot(True, tm.get_port())
        tm.update_vehicle_lights(ego, True)
        print(f"[record] {town}: {len(npcs)} NPCs, {len(controllers)} walker controllers",
              flush=True)
        weathers = list(d["weathers"])
        tick = 0
        while saved < args.frames:
            wi = (saved // int(d["frames_per_weather"])) % len(weathers)
            if not weather_log or weather_log[-1]["weather"] != weathers[wi]:
                world.set_weather(weather_preset(weathers[wi]))
                weather_log.append({"weather": weathers[wi], "first_index": saved})
            frame = world.tick(float(c["tick_timeout_s"]))
            img = q_rgb.get(frame, float(cfg["bridge"]["sensor_timeout_s"]))
            dimg = q_dep.get(frame, float(cfg["bridge"]["sensor_timeout_s"]))
            tick += 1
            if tick % int(d["every_n_ticks"]):
                continue
            bgr = bgra_to_bgr(img.raw_data, img.height, img.width)
            depth = decode_depth(dimg.raw_data, dimg.height, dimg.width)
            w2c = np.array(img.transform.get_inverse_matrix())
            labels = label_objects(scene_objects(world, ego, float(d["max_box_distance_m"])),
                                   w2c, depth, intr, d)
            name = f"{town}_{saved:05d}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, int(d["jpg_quality"])])
            (out / "labels" / f"{name}.txt").write_text("".join(
                yolo_line(cls_ids[lb["cls_name"]], np.array(lb["bbox_xyxy"]), img.width,
                          img.height) + "\n" for lb in labels), encoding="utf-8")
            (out / "gt" / f"{name}.json").write_text(json.dumps({
                "frame": int(frame), "sim_time_s": float(img.timestamp),
                "weather": weathers[wi], "ego_speed_mps": float(ego.get_velocity().length()),
                "objects": labels}), encoding="utf-8")
            if saved % int(d["depth_every_n_saved"]) == 0:
                cm = np.clip(depth * 100.0, 0, 65535).astype(np.uint16)
                cv2.imwrite(str(out / "depth" / f"{name}.png"), cm)
            saved += 1
            if saved % 200 == 0:
                print(f"[record] {saved}/{args.frames} frames "
                      f"({time.monotonic() - t0:.0f} s, {weathers[wi]})", flush=True)
    finally:
        for ctl in controllers:
            try:
                ctl.stop()
            except Exception:  # noqa: BLE001
                pass
        n = destroy_actors(client, actors + controllers)  # reversed: controllers go first
        if tm is not None:
            tm.set_synchronous_mode(False)
        if original is not None:
            restore_settings(world, original)
        world.set_weather(weather0)
        meta = {
            "town": town, "frames": saved, "camera": cam_cfg,
            "intrinsics": dict(zip(("fx", "fy", "cx", "cy"), intr, strict=True)),
            "classes": cls_ids, "weathers": weather_log, "dataset_cfg": d,
            "distance_definition": "nearest 3D bbox vertex depth along the camera axis (m)",
            "depth_png": "uint16 centimetres",
        }
        (out / f"meta_{town}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[record] saved {saved} frames to {out}; destroyed {n} actors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
