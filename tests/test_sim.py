"""sim/ (M5): CARLA-free helpers, scenario logic and config consistency. No carla package, no
simulator: everything here runs in CI."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from perception.pipeline import load_pipeline_config, stage_configs
from sim.carla_utils import (
    ControlWaiter,
    SensorQueue,
    bgra_to_bgr,
    camera_intrinsics,
    carla_to_ros_pose,
    decode_depth,
    euler_to_quaternion,
    forward_speed,
    heading_change_deg,
    load_carla_config,
    longitudinal_gap,
)
from sim.scenarios import registry, scenario_config
from sim.scenarios.common import EpisodeMonitor, crossing_state
from sim.scenarios.cut_in import cut_in_profile

CFG = load_carla_config("configs/carla.yaml")


# ---------------------------------------------------------------------------- images
def test_bgra_to_bgr_drops_alpha():
    raw = np.zeros((2, 3, 4), np.uint8)
    raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3] = 10, 20, 30, 255
    out = bgra_to_bgr(raw.tobytes(), 2, 3)
    assert out.shape == (2, 3, 3) and out.flags["C_CONTIGUOUS"]
    assert out[0, 0].tolist() == [10, 20, 30]


def test_decode_depth_matches_carla_formula():
    raw = np.zeros((1, 3, 4), np.uint8)
    raw[0, 1, 2] = 255  # R only: 255 / (2^24 - 1) * 1000
    raw[0, 2, :3] = 255  # all 24 bits -> far plane 1000 m
    d = decode_depth(raw.tobytes(), 1, 3)
    assert d.dtype == np.float32
    assert d[0, 0] == 0.0
    assert d[0, 1] == pytest.approx(255 / (256**3 - 1) * 1000, rel=1e-5)
    assert d[0, 2] == pytest.approx(1000.0)
    # 10 m: n = 10 / 1000 * (2^24 - 1) -> B, G, R bytes
    n = round(10 / 1000 * (256**3 - 1))
    raw2 = np.array([[[n // 65536, (n // 256) % 256, n % 256, 255]]], np.uint8)
    assert decode_depth(raw2.tobytes(), 1, 1)[0, 0] == pytest.approx(10.0, abs=1e-3)


# ---------------------------------------------------------------------------- geometry
def test_camera_config_matches_camera_model():
    cam = CFG["camera"]
    fx, fy, cx, cy = camera_intrinsics(cam["width"], cam["height"], cam["fov_deg"])
    model = yaml.safe_load(open("configs/camera.yaml", encoding="utf-8"))["carla"]
    assert (model["width"], model["height"]) == (cam["width"], cam["height"])
    assert fx == pytest.approx(model["fx"]) and fy == pytest.approx(model["fy"])
    assert (cx, cy) == (model["cx"], model["cy"])
    assert cam["z"] == pytest.approx(model["height_m"])
    assert cam["pitch_deg"] == pytest.approx(model["pitch_deg"])
    assert (cam["width"], cam["height"]) == (800, 450)  # ARCHITECTURE.md 3.7 load budget


def test_quaternion_identity_and_yaw():
    assert euler_to_quaternion(0, 0, 0) == pytest.approx((0, 0, 0, 1))
    q = euler_to_quaternion(0, 0, math.pi / 2)
    assert q == pytest.approx((0, 0, math.sqrt(0.5), math.sqrt(0.5)))


def test_carla_to_ros_pose_flips_y_and_yaw():
    pos, q = carla_to_ros_pose(1.0, 2.0, 3.0, 0.0, 0.0, 90.0)  # CARLA yaw 90 = facing +y_carla
    assert pos == (1.0, -2.0, 3.0)
    assert q == pytest.approx(euler_to_quaternion(0, 0, -math.pi / 2))
    _, qp = carla_to_ros_pose(0, 0, 0, 0.0, 10.0, 0.0)  # nose up -> negative ROS pitch
    assert qp[1] < 0


def test_forward_speed():
    assert forward_speed(5.0, 0.0, 0.0) == pytest.approx(5.0)
    assert forward_speed(0.0, 5.0, 90.0) == pytest.approx(5.0)
    assert forward_speed(-5.0, 0.0, 0.0) == pytest.approx(-5.0)


def test_longitudinal_gap():
    # ego at origin facing +x (half length 2.4), car 20 m ahead (half length 2.0)
    assert longitudinal_gap((0, 0), 0.0, 2.4, 1.0, (20, 0), 2.0, 0.9) == pytest.approx(15.6)
    # adjacent lane (3.5 m to the side): not in path
    assert longitudinal_gap((0, 0), 0.0, 2.4, 1.0, (20, 3.5), 2.0, 0.9) is None
    # behind: not in path
    assert longitudinal_gap((0, 0), 0.0, 2.4, 1.0, (-10, 0), 2.0, 0.9) is None
    # heading 90 deg (CARLA +y), walker ahead; extent along heading = max(half len, half width)
    g = longitudinal_gap((0, 0), 90.0, 2.4, 1.0, (0.5, 10), 0.2, 0.3)
    assert g == pytest.approx(10 - 2.4 - 0.3)
    assert longitudinal_gap((0, 0), 0.0, 2.4, 1.0, (3, 0), 2.0, 0.9) < 0  # overlap


def test_heading_change_wraps():
    assert heading_change_deg(179.0, -179.0) == pytest.approx(2.0)
    assert heading_change_deg(10.0, 350.0) == pytest.approx(20.0)


# ---------------------------------------------------------------------------- bridge sync
def test_control_waiter_first_timeout_then_regular():
    w = ControlWaiter(0.5, 120.0)
    assert w.expect(1_000) == 120.0  # models still loading
    assert not w.offer(500)  # stale control for an older frame
    assert w.offer(1_000) and w.got_expected
    assert w.expect(2_000) == 0.5
    assert not w.got_expected
    w.timed_out()
    assert w.timeouts == 1


class _Data:
    def __init__(self, frame: int) -> None:
        self.frame = frame


def test_sensor_queue_returns_matching_frame():
    q = SensorQueue()
    for f in (3, 4, 5):
        q(_Data(f))
    assert q.get(5, 0.1).frame == 5  # older frames are dropped
    with pytest.raises(TimeoutError):
        q.get(6, 0.05)
    q(_Data(8))
    with pytest.raises(RuntimeError):
        q.get(7, 0.1)


# ---------------------------------------------------------------------------- scenarios
def test_registry_and_configs():
    reg = registry()
    assert set(reg) == {"stopped_car", "pedestrian_crossing", "cut_in", "night_rain"}
    for name in reg:
        scenario_config(CFG, name)
    nr = scenario_config(CFG, "night_rain")
    assert nr["distance_m"] == CFG["scenarios"]["stopped_car"]["distance_m"]
    assert nr["weather"] and "base" not in nr
    with pytest.raises(KeyError):
        scenario_config(CFG, "nope")
    assert CFG["carla"]["town"] in ("Town01", "Town02")
    assert CFG["carla"]["host"] == "localhost"


def _monitor(**kw) -> EpisodeMonitor:
    s = CFG["scenario"]
    args = dict(
        arm_speed_mps=s["arm_speed_mps"], stop_speed_mps=s["stop_speed_mps"],
        stop_hold_s=s["stop_hold_s"], timeout_s=s["timeout_s"],
    )
    args.update(kw)
    return EpisodeMonitor(**args)


def test_episode_monitor_stop_after_arming():
    m = _monitor(timeout_s=100.0)
    t = 0.0
    assert m.update(t, 0.0, None) is None  # at rest before driving: not a stop
    for _ in range(100):
        t += 0.05
        m.update(t, 0.0, 50.0)
    assert m.outcome is None and not m.armed
    for v, gap in ((4.0, 40.0), (8.0, 20.0), (3.0, 9.0), (0.5, 6.5)):
        t += 0.05
        m.update(t, v, gap)
    assert m.armed and m.outcome is None
    while m.outcome is None:
        t += 0.05
        m.update(t, 0.0, 6.2)
    r = m.result()
    assert r["outcome"] == "stopped" and r["success"] and not r["collision"]
    assert r["min_gap_m"] == pytest.approx(6.2) and r["stop_gap_m"] == pytest.approx(6.2)
    assert r["max_speed_mps"] == pytest.approx(8.0)


def test_episode_monitor_post_stop_tail():
    s = CFG["scenario"]
    assert float(s["post_stop_s"]) > 0.0
    m = _monitor(post_stop_s=float(s["post_stop_s"]), timeout_s=100.0)
    t = 0.0
    m.update(t, 8.0, 20.0)
    while m.stop_gap_m is None:  # stop detection unchanged by the tail
        t += 0.05
        assert m.update(t, 0.0, 10.5) is None
    t_stop = t
    assert m.outcome == "stopped" and m.stop_gap_m == pytest.approx(10.5)
    while m.update(t, 0.0, 10.4) is None:  # the tail keeps the episode running
        t += 0.05
    assert t - t_stop == pytest.approx(float(s["post_stop_s"]), abs=0.06)
    r = m.result()
    assert r["success"] and r["final_gap_m"] == pytest.approx(10.4)
    assert r["stop_gap_m"] == pytest.approx(10.5) and r["post_stop_max_speed_mps"] == 0.0
    # a collision during the tail still fails the run
    m = _monitor(post_stop_s=3.0)
    t = 0.0
    m.update(t, 8.0, 20.0)
    while m.stop_gap_m is None:
        t += 0.05
        m.update(t, 0.0, 10.5)
    m.on_collision("vehicle.audi.a2")
    assert m.update(t + 0.05, 0.0, 0.0) == "collision" and not m.result()["success"]


def test_episode_monitor_collision_and_timeout():
    m = _monitor()
    m.update(0.0, 8.0, 3.0)
    m.on_collision("vehicle.audi.a2")
    r = m.result()
    assert r["outcome"] == "collision" and not r["success"]
    assert r["collision_with"] == "vehicle.audi.a2" and r["collision_speed_mps"] == 8.0
    m2 = _monitor(timeout_s=1.0)
    t = 0.0
    while m2.update(t, 8.0, None) is None:
        t += 0.05
    assert m2.outcome == "timeout" and not m2.result()["success"]


def test_crossing_state_machine():
    assert crossing_state("walk_in", 1.0, 0.0, 0.0, 4.0, -4.0) == "walk_in"
    assert crossing_state("walk_in", -0.01, 1.0, 0.0, 4.0, -4.0) == "pause"
    assert crossing_state("walk_in", -0.01, 1.0, 0.0, 0.0, -4.0) == "walk_out"
    assert crossing_state("pause", 0.0, 4.5, 1.0, 4.0, -4.0) == "pause"
    assert crossing_state("pause", 0.0, 5.0, 1.0, 4.0, -4.0) == "walk_out"
    assert crossing_state("walk_out", -4.1, 6.0, 5.0, 4.0, -4.0) == "done"


def test_cut_in_profile():
    v, lat = cut_in_profile(0.0, 5.0, 1.5, -3.5, 3.0)
    assert (v, lat) == (5.0, pytest.approx(-3.5))
    v, lat = cut_in_profile(0.75, 5.0, 1.5, -3.5, 3.0)
    assert lat == pytest.approx(-1.75)
    v, lat = cut_in_profile(1.5 + 1.0, 5.0, 1.5, -3.5, 3.0)
    assert lat == 0.0 and v == pytest.approx(2.0)
    assert cut_in_profile(10.0, 5.0, 1.5, -3.5, 3.0) == (0.0, 0.0)


# ---------------------------------------------------------------------------- live pipeline
def test_pipeline_carla_config_is_tensorrt_and_carla():
    # M6.4: the live loop runs the laptop TensorRT engines; never PyTorch.
    p = CFG["perception"]
    cfg = load_pipeline_config(p["pipeline_config"])
    sc = stage_configs(cfg, backend=p["backend"], camera_name=p["camera"], fps=p["fps"])
    assert p["backend"] == "tensorrt"
    assert sc["detector"]["backend"] == "tensorrt"
    assert str(sc["detector"]["weights"]).endswith(".engine")
    assert sc["depth"]["estimator"]["backend"] == "tensorrt"
    assert str(sc["lanes"]["yolop"]["weights"]).endswith(".engine")
    assert sc["camera_name"] == "carla"
    assert sc["tracker"]["frame_rate"] == 20
    assert sc["lanes"]["method"] == "yolop"
    # lanes_carla.yaml ROI is used, not the BDD/KITTI one
    carla_roi = yaml.safe_load(open("configs/lanes_carla.yaml", encoding="utf-8"))
    assert sc["lanes"]["fitting"]["roi"] == carla_roi["fitting"]["roi"]
    assert 1.0 / CFG["carla"]["fixed_delta_seconds"] == pytest.approx(p["fps"])
