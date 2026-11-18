"""CARLA client helpers shared by sim/smoke_test.py, sim/scenarios/, sim/record_dataset.py and
the ROS2 carla_bridge_node. ROS-free.

The module imports without the ``carla`` package (CI has none): the pure helpers at the top
(depth decoding, frame conversions, geometry) are unit-tested in tests/test_sim_utils.py, and
every function that needs ``carla`` imports it lazily.

CARLA uses a left-handed frame (x forward, y right, z up, yaw clockwise seen from above). ROS
uses a right-handed one (REP 103: x forward, y left, z up), so y, roll and yaw change sign.
"""

from __future__ import annotations

import math
import queue
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPTH_FAR_M = 1000.0  # CARLA depth camera range encoded in 24 bits


# ---------------------------------------------------------------------------- config
def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def load_carla_config(path: str | Path = "configs/carla.yaml") -> dict[str, Any]:
    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------- pure helpers
def bgra_to_bgr(raw: bytes | np.ndarray, height: int, width: int) -> np.ndarray:
    """``carla.Image.raw_data`` (BGRA uint8) -> contiguous HxWx3 BGR array (a copy)."""
    a = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    a = a.reshape(height, width, 4)
    return np.ascontiguousarray(a[:, :, :3])


def decode_depth(raw: bytes | np.ndarray, height: int, width: int) -> np.ndarray:
    """CARLA depth camera BGRA -> float32 metres: (R + 256 G + 65536 B) / (2^24 - 1) * 1000."""
    a = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw
    a = a.reshape(height, width, 4).astype(np.float32)
    norm = (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) / (256.0**3 - 1.0)
    return (norm * DEPTH_FAR_M).astype(np.float32)


def camera_intrinsics(width: int, height: int, fov_deg: float) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) of a CARLA pinhole camera (square pixels, centred principal point)."""
    f = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return f, f, width / 2.0, height / 2.0


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Radians (ZYX order) -> quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def carla_to_ros_pose(
    x: float, y: float, z: float, roll_deg: float, pitch_deg: float, yaw_deg: float
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """CARLA location (m) + rotation (deg) -> ROS position and quaternion (x, y, z, w).

    Same convention as carla_ros_bridge: y, pitch and yaw change sign, roll keeps it. CARLA
    yaw > 0 turns right (ROS: left is positive); CARLA pitch > 0 is nose up, which in ROS is a
    negative rotation about +y (right-hand rule)."""
    q = euler_to_quaternion(
        math.radians(roll_deg), -math.radians(pitch_deg), -math.radians(yaw_deg)
    )
    return (x, -y, z), q


def forward_speed(vx: float, vy: float, yaw_deg: float) -> float:
    """Signed speed along the vehicle heading (m/s, CARLA world velocity + yaw)."""
    yaw = math.radians(yaw_deg)
    return vx * math.cos(yaw) + vy * math.sin(yaw)


def longitudinal_gap(
    ego_xy: tuple[float, float],
    ego_yaw_deg: float,
    ego_half_len: float,
    ego_half_wid: float,
    obj_xy: tuple[float, float],
    obj_half_len: float,
    obj_half_wid: float,
    margin_m: float = 0.5,
) -> float | None:
    """Bumper-to-bumper gap (m) to an object in the ego path, else None.

    The object counts as in the path when its centre's lateral offset in the ego frame is
    below ``ego_half_wid + obj_half_wid + margin_m`` and it is ahead of the ego centre. The
    object's extent along the ego heading is taken as ``max(obj_half_len, obj_half_wid)``
    (orientation-agnostic, so a crossing pedestrian or a car at an angle is not
    under-estimated). A negative value means the boxes overlap longitudinally.
    """
    yaw = math.radians(ego_yaw_deg)
    dx, dy = obj_xy[0] - ego_xy[0], obj_xy[1] - ego_xy[1]
    lon = dx * math.cos(yaw) + dy * math.sin(yaw)
    lat = -dx * math.sin(yaw) + dy * math.cos(yaw)
    if lon <= 0.0 or abs(lat) > ego_half_wid + obj_half_wid + margin_m:
        return None
    return lon - ego_half_len - max(obj_half_len, obj_half_wid)


def heading_change_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute difference between two headings, in degrees."""
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


class ControlWaiter:
    """When may the bridge tick again? After the control for the current frame arrives, or after
    a timeout (``first_timeout_s`` until the first control ever arrives: model loading).

    Controls are matched to frames by their header stamp (sim time in nanoseconds), which
    control_node copies from the /perception/objects message it reacts to."""

    def __init__(self, timeout_s: float, first_timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        self.first_timeout_s = float(first_timeout_s)
        self.seen_any = False
        self.expected_ns: int | None = None
        self.got_expected = False
        self.timeouts = 0

    def expect(self, stamp_ns: int) -> float:
        """Start waiting for frame ``stamp_ns``; returns the timeout to use (seconds)."""
        self.expected_ns = int(stamp_ns)
        self.got_expected = False
        return self.timeout_s if self.seen_any else self.first_timeout_s

    def offer(self, stamp_ns: int) -> bool:
        """A control arrived; True if it is for the expected (or a newer) frame."""
        self.seen_any = True
        if self.expected_ns is not None and int(stamp_ns) >= self.expected_ns:
            self.got_expected = True
        return self.got_expected

    def timed_out(self) -> None:
        self.timeouts += 1


# ---------------------------------------------------------------------------- carla helpers
def connect(host: str, port: int, timeout_s: float) -> Any:
    import carla  # lazy: not installed in CI

    client = carla.Client(host, int(port))
    client.set_timeout(float(timeout_s))
    client.get_server_version()  # fails fast with a clear error if the server is unreachable
    return client


def ensure_town(client: Any, town: str | None, load: bool) -> Any:
    """Return the world; load ``town`` first if ``load`` and another map is active."""
    world = client.get_world()
    current = world.get_map().name.split("/")[-1]
    if town and load and current != town:
        print(f"[carla] loading {town} (was {current})", flush=True)
        world = client.load_world(town)
    elif town and current != town:
        print(f"[carla] WARNING map is {current}, config wants {town} (load_town false)",
              flush=True)
    return world


def set_synchronous(world: Any, fixed_delta_s: float) -> Any:
    """Switch to synchronous mode; returns the previous settings for ``restore_settings``."""
    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = float(fixed_delta_s)
    s.no_rendering_mode = False  # cameras need rendering
    world.apply_settings(s)
    return original


def restore_settings(world: Any, original: Any) -> None:
    try:
        s = world.get_settings()
        s.synchronous_mode = original.synchronous_mode if original is not None else False
        s.fixed_delta_seconds = original.fixed_delta_seconds if original is not None else None
        world.apply_settings(s)
    except Exception as e:  # noqa: BLE001 - shutdown path: report, never raise
        print(f"[carla] could not restore world settings: {e}", flush=True)


def destroy_actors(client: Any, actors: list[Any]) -> int:
    """Stop sensors and destroy actors (reverse spawn order). Returns how many were destroyed."""
    import carla

    alive = [a for a in reversed(actors) if a is not None]
    for a in alive:
        if hasattr(a, "is_listening") and getattr(a, "is_listening", False):
            try:
                a.stop()
            except Exception:  # noqa: BLE001
                pass
    if not alive:
        return 0
    try:
        res = client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in alive], False)
        return sum(1 for r in res if not r.error)
    except Exception as e:  # noqa: BLE001
        print(f"[carla] batch destroy failed ({e}), destroying one by one", flush=True)
        n = 0
        for a in alive:
            try:
                n += int(bool(a.destroy()))
            except Exception:  # noqa: BLE001
                pass
        return n


def straight_run_m(
    carla_map: Any, location: Any, step_m: float = 2.0, max_len_m: float = 200.0,
    max_heading_deg: float = 10.0,
) -> float:
    """Length of straight lane ahead of ``location`` before a junction or a bend."""
    wp = carla_map.get_waypoint(location)
    if wp is None:
        return 0.0
    yaw0 = wp.transform.rotation.yaw
    run = 0.0
    while run < max_len_m:
        nxt = wp.next(step_m)
        if not nxt:
            break
        wp = nxt[0]
        if wp.is_junction or heading_change_deg(wp.transform.rotation.yaw, yaw0) > max_heading_deg:
            break
        run += step_m
    return run


def pick_spawn_point(carla_map: Any, spec: Any, min_straight_m: float) -> Any:
    """Spawn transform: index ``spec`` or, for ``"auto"``, the longest straight run ahead."""
    points = carla_map.get_spawn_points()
    if not points:
        raise RuntimeError("map has no spawn points")
    if str(spec) != "auto":
        return points[int(spec) % len(points)]
    runs = [straight_run_m(carla_map, p.location) for p in points]
    best = int(np.argmax(runs))
    if runs[best] < min_straight_m:
        print(f"[carla] WARNING longest straight run is {runs[best]:.0f} m", flush=True)
    print(f"[carla] spawn point {best}: {runs[best]:.0f} m straight ahead", flush=True)
    return points[best]


def camera_blueprint(bp_lib: Any, cam_cfg: dict[str, Any], kind: str = "rgb") -> Any:
    bp = bp_lib.find("sensor.camera.rgb" if kind == "rgb" else "sensor.camera.depth")
    bp.set_attribute("image_size_x", str(int(cam_cfg["width"])))
    bp.set_attribute("image_size_y", str(int(cam_cfg["height"])))
    bp.set_attribute("fov", str(float(cam_cfg["fov_deg"])))
    bp.set_attribute("sensor_tick", str(float(cam_cfg.get("sensor_tick", 0.0))))
    return bp


def camera_transform(cam_cfg: dict[str, Any]) -> Any:
    import carla

    return carla.Transform(
        carla.Location(x=float(cam_cfg["x"]), y=float(cam_cfg["y"]), z=float(cam_cfg["z"])),
        carla.Rotation(pitch=float(cam_cfg.get("pitch_deg", 0.0))),
    )


def spawn_ego(world: Any, ego_cfg: dict[str, Any], seed: int = 0) -> Any:
    """Spawn the ego vehicle (role ``ego_cfg.role_name``) at the configured spawn point."""
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find(ego_cfg["blueprint"])
    bp.set_attribute("role_name", str(ego_cfg["role_name"]))
    if bp.has_attribute("color"):
        bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
    carla_map = world.get_map()
    tf = pick_spawn_point(carla_map, ego_cfg.get("spawn_point", "auto"),
                          float(ego_cfg.get("min_straight_m", 0.0)))
    ego = world.try_spawn_actor(bp, tf)
    if ego is None:  # spawn point blocked (orphan from an earlier run?): try the others
        for p in carla_map.get_spawn_points():
            ego = world.try_spawn_actor(bp, p)
            if ego is not None:
                print("[carla] WARNING configured spawn point blocked, used another", flush=True)
                break
    if ego is None:
        raise RuntimeError("could not spawn the ego vehicle")
    return ego


def find_actor_by_role(world: Any, role: str, timeout_s: float = 0.0) -> Any | None:
    """First vehicle with ``role_name == role``; polls up to ``timeout_s`` seconds."""
    t_end = time.monotonic() + float(timeout_s)
    while True:
        for a in world.get_actors().filter("vehicle.*"):
            if a.attributes.get("role_name") == role:
                return a
        if time.monotonic() >= t_end:
            return None
        time.sleep(0.5)


def weather_preset(name: str) -> Any:
    import carla

    w = getattr(carla.WeatherParameters, name, None)
    if w is None:
        raise ValueError(f"unknown carla.WeatherParameters preset {name!r}")
    return w


class SensorQueue:
    """Thread-safe buffer for a sensor's callback; ``get(frame)`` returns that frame's data."""

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()

    def __call__(self, data: Any) -> None:
        self._q.put(data)

    def get(self, frame: int, timeout_s: float) -> Any:
        t_end = time.monotonic() + float(timeout_s)
        while True:
            left = t_end - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no sensor data for frame {frame}")
            try:
                data = self._q.get(timeout=left)
            except queue.Empty:
                raise TimeoutError(f"no sensor data for frame {frame}") from None
            if data.frame == frame:
                return data
            if data.frame > frame:
                raise RuntimeError(f"sensor skipped frame {frame} (got {data.frame})")
