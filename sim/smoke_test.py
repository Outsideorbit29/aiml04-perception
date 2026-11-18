"""M0.8 smoke test: connect to the CARLA server, spawn the ego car + RGB camera, save one frame.

Runs in WSL2 Ubuntu 22.04 (Python 3.10, `pip install carla==0.9.15`), with the CARLA server
already running on Windows:

    # Windows (PowerShell)
    D:\\CARLA_0.9.15\\CarlaUE4.exe -quality-level=Low -RenderOffScreen -fps=20
    # WSL2 (repo root)
    python3 sim/smoke_test.py --out runs/smoke/frame.png [--town Town01] [--host localhost]

No windows are opened. The world is switched to synchronous mode only for the test; the
original settings are restored and every spawned actor is destroyed in ``finally``, also on
Ctrl+C or an error, so the server is left clean.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # `python3 sim/smoke_test.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.carla_utils import (  # noqa: E402
    SensorQueue,
    bgra_to_bgr,
    camera_blueprint,
    camera_transform,
    connect,
    destroy_actors,
    ensure_town,
    load_carla_config,
    restore_settings,
    set_synchronous,
    spawn_ego,
)


def main(argv: list[str] | None = None) -> int:
    cfg = load_carla_config()
    c = cfg["carla"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=c["host"])
    ap.add_argument("--port", type=int, default=int(c["port"]))
    ap.add_argument("--timeout", type=float, default=float(c["timeout_s"]), help="RPC seconds")
    ap.add_argument("--town", default=None, help="load this map first if another is loaded")
    ap.add_argument("--ticks", type=int, default=10, help="sim steps before the saved frame")
    ap.add_argument("--out", default="runs/smoke/frame.png")
    args = ap.parse_args(argv)

    import cv2

    t0 = time.monotonic()
    client = connect(args.host, args.port, args.timeout)
    print(f"[smoke] connected to {args.host}:{args.port} server "
          f"{client.get_server_version()} client {client.get_client_version()}", flush=True)
    world = ensure_town(client, args.town, load=args.town is not None)
    print(f"[smoke] map {world.get_map().name}", flush=True)

    actors: list = []
    original = None
    try:
        original = set_synchronous(world, float(c["fixed_delta_seconds"]))
        ego = spawn_ego(world, cfg["ego"])
        actors.append(ego)
        bp = camera_blueprint(world.get_blueprint_library(), cfg["camera"], "rgb")
        cam = world.spawn_actor(bp, camera_transform(cfg["camera"]), attach_to=ego)
        actors.append(cam)
        q = SensorQueue()
        cam.listen(q)
        img = None
        for _ in range(max(1, args.ticks)):
            frame = world.tick(float(c["tick_timeout_s"]))
            img = q.get(frame, float(cfg["bridge"]["sensor_timeout_s"]))
        bgr = bgra_to_bgr(img.raw_data, img.height, img.width)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), bgr):
            raise RuntimeError(f"cannot write {out}")
        print(f"[smoke] saved {out} ({img.width}x{img.height}, frame {img.frame}, "
              f"mean {bgr.mean():.1f}) in {time.monotonic() - t0:.1f} s", flush=True)
    finally:
        n = destroy_actors(client, actors)
        if original is not None:
            restore_settings(world, original)
        print(f"[smoke] destroyed {n} actors, world settings restored", flush=True)
    print("SMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
