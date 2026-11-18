"""perception_ros.conversions and message definitions (M5.1, M5.3). No ROS: messages are
replaced by SimpleNamespace objects, and every converter is checked against the field list of
the real .msg file so the two cannot drift apart."""

from __future__ import annotations

import ast
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ros2_ws" / "src" / "perception_ros"
MSG = ROOT / "ros2_ws" / "src" / "perception_msgs" / "msg"
sys.path.insert(0, str(PKG))

from perception_ros import conversions as cv  # noqa: E402
from perception_ros.repo import repo_root  # noqa: E402

from perception.control import ControlDecision, LeadCandidate  # noqa: E402
from perception.pipeline import frame_to_dict  # noqa: E402
from perception.types import (  # noqa: E402
    Detection,
    LaneResult,
    ObjectState,
    PerceptionFrame,
    Polyline,
    Track,
)


def msg_fields(name: str) -> list[str]:
    """Field names of ros2_ws/src/perception_msgs/msg/<name>.msg (header excluded)."""
    out = []
    for line in (MSG / f"{name}.msg").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        typ, field = line.split()[:2]
        assert re.fullmatch(r"[a-z][a-z0-9_]*", field), field  # rosidl naming rule
        if field != "header":
            out.append(field)
    return out


def _frame() -> PerceptionFrame:
    det = Detection(np.array([100, 200, 160, 260], np.float32), 0.87, 0, "car")
    objs = [
        ObjectState(Track(3, det, age=12), 14.5, 0.6, 4.2, 3.45, True),
        ObjectState(Track(9, det, age=2), 31.0, 2.1, -0.4, None, False),
    ]
    lanes = LaneResult(Polyline(np.array([0.001, -0.5, 300.0])), None,
                       np.zeros((4, 4), bool), 0.62)
    return PerceptionFrame(1.25, 25, objs, lanes, {"total_ms": 48.0})


def test_repo_root_found():
    assert repo_root() == ROOT.resolve()


def test_message_names_are_valid_and_complete():
    assert set(msg_fields("TrackedObjectArray")) == {
        "frame_id", "objects", "lead_track_id", "total_ms",
    }
    # TrackedObject carries every ObjectState / Detection field of object_to_dict
    d = frame_to_dict(_frame())["objects"][0]
    assert set(d) <= set(msg_fields("TrackedObject"))


def test_object_array_roundtrip_through_control_candidates():
    d = frame_to_dict(_frame())
    msg = cv.fill_object_array(SimpleNamespace(), d, SimpleNamespace)
    assert set(vars(msg)) == set(msg_fields("TrackedObjectArray"))
    assert set(vars(msg.objects[0])) == set(msg_fields("TrackedObject"))
    assert msg.frame_id == 25 and msg.lead_track_id == 3 and msg.total_ms == 48.0
    o0, o1 = msg.objects
    assert o0.has_ttc and o0.ttc_s == pytest.approx(3.45) and o0.in_ego_lane
    assert not o1.has_ttc and math.isnan(o1.ttc_s)
    assert o0.bbox_xyxy == [100.0, 200.0, 160.0, 260.0]
    cands = cv.candidates_from_msg(msg)
    assert [c.track_id for c in cands] == [3, 9]
    assert cands[0].ttc_s == pytest.approx(3.45) and cands[1].ttc_s is None


def test_candidates_skip_nan_distance():
    o = SimpleNamespace(track_id=1, distance_m=float("nan"), has_ttc=False, ttc_s=float("nan"),
                        closing_speed_mps=0.0, in_ego_lane=True)
    assert cv.candidates_from_msg(SimpleNamespace(objects=[o])) == []


def test_lanes_message():
    d = frame_to_dict(_frame())
    msg = cv.fill_lanes(SimpleNamespace(), d["lanes"], {"y_top": 260.0, "y_bottom": 430.0})
    assert set(vars(msg)) == set(msg_fields("LaneBoundaries"))
    assert msg.has_left and not msg.has_right and msg.has_ego_mask
    assert msg.left_coeffs == pytest.approx([0.001, -0.5, 300.0])
    assert msg.right_coeffs == [0.0, 0.0, 0.0]
    assert msg.y_top == 260.0
    assert math.isnan(cv.fill_lanes(SimpleNamespace(), d["lanes"]).y_top)


def test_control_message():
    dec = ControlDecision(True, 0.0, 1.0, "AEB", LeadCandidate(4, 7.5, 1.2, 6.0, True), "ttc")
    msg = cv.fill_control(SimpleNamespace(), dec, 8.3)
    assert set(vars(msg)) == set(msg_fields("EgoControl"))
    assert msg.override_autopilot and msg.brake == 1.0 and msg.mode == "AEB"
    assert msg.lead_track_id == 4 and msg.lead_ttc_s == 1.2 and msg.ego_speed_mps == 8.3
    none = cv.fill_control(SimpleNamespace(), ControlDecision(False, 0, 0, "AUTOPILOT"), 0.0)
    assert none.lead_track_id == -1 and math.isnan(none.lead_distance_m)


@pytest.mark.parametrize("enc", ["bgr8", "rgb8", "bgra8", "rgba8", "mono8"])
def test_image_to_bgr_encodings(enc):
    bgr = np.random.default_rng(0).integers(0, 255, (5, 7, 3), dtype=np.uint8)
    if enc == "bgr8":
        arr = bgr
    elif enc == "rgb8":
        arr = bgr[:, :, ::-1]
    elif enc == "bgra8":
        arr = np.dstack([bgr, np.full((5, 7), 255, np.uint8)])
    elif enc == "rgba8":
        arr = np.dstack([bgr[:, :, ::-1], np.full((5, 7), 255, np.uint8)])
    else:
        arr = bgr[:, :, :1]
        bgr = np.repeat(arr, 3, axis=2)
    pad = 3  # rows padded beyond width * channels (step > width * ch)
    ch = arr.shape[2]
    rows = np.zeros((5, 7 * ch + pad), np.uint8)
    rows[:, : 7 * ch] = arr.reshape(5, -1)
    msg = SimpleNamespace(encoding=enc, height=5, width=7, step=7 * ch + pad,
                          data=rows.tobytes())
    np.testing.assert_array_equal(cv.image_to_bgr(msg), bgr)


def test_fill_image_roundtrip_bgr_and_depth():
    img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    msg = cv.fill_image(SimpleNamespace(), img, "bgr8")
    assert (msg.height, msg.width, msg.step, msg.is_bigendian) == (4, 6, 18, 0)
    np.testing.assert_array_equal(cv.image_to_bgr(msg), img)
    depth = np.linspace(0.5, 80.0, 24, dtype=np.float32).reshape(4, 6)
    dm = cv.fill_image(SimpleNamespace(), depth, "32FC1")
    assert dm.step == 24
    np.testing.assert_array_equal(cv.image_to_depth(dm), depth)
    with pytest.raises(ValueError):
        cv.fill_image(SimpleNamespace(), img, "mono8")
    with pytest.raises(ValueError):
        cv.image_to_bgr(SimpleNamespace(encoding="16UC1", height=1, width=1, step=2, data=b"00"))


def test_stamps():
    s = SimpleNamespace(sec=0, nanosec=0)
    cv.set_stamp_ns(s, 12_345_000_000_123)
    assert (s.sec, s.nanosec) == (12_345, 123)
    assert cv.stamp_to_ns(s) == 12_345_000_000_123


def test_ros_sources_parse_and_have_entry_points():
    nodes = ["carla_bridge_node", "perception_node", "control_node", "recorder_node"]
    for f in [*(PKG / "perception_ros").glob("*.py"), PKG / "launch" / "stack.launch.py"]:
        ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
    setup = (PKG / "setup.py").read_text(encoding="utf-8")
    for n in nodes:
        assert (PKG / "perception_ros" / f"{n}.py").is_file()
        assert f"{n} = {{PACKAGE}}.{n}:main" in setup
    cmake = (MSG.parent / "CMakeLists.txt").read_text(encoding="utf-8")
    for m in ("TrackedObject", "TrackedObjectArray", "LaneBoundaries", "EgoControl"):
        assert f'"msg/{m}.msg"' in cmake
