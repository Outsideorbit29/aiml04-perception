"""Message <-> numpy / dict conversions. No ROS imports: every function works on any object
with the message's attributes, so tests/test_ros_conversions.py runs without ROS (CI).

Image encodings handled: bgr8, rgb8, bgra8, rgba8, mono8 (to BGR) and 32FC1 (depth, metres).
"""

from __future__ import annotations

import math
from array import array
from typing import Any

import numpy as np

NAN = float("nan")
_CHANNELS = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}


# ---------------------------------------------------------------------------- time
def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def set_stamp_ns(stamp: Any, ns: int) -> None:
    stamp.sec, stamp.nanosec = divmod(int(ns), 1_000_000_000)


def copy_stamp(dst: Any, src: Any) -> None:
    dst.sec, dst.nanosec = int(src.sec), int(src.nanosec)


# ---------------------------------------------------------------------------- images
def image_to_bgr(msg: Any) -> np.ndarray:
    """sensor_msgs/Image (8-bit colour or mono) -> HxWx3 uint8 BGR (a copy)."""
    enc = str(msg.encoding).lower()
    if enc not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    ch = _CHANNELS[enc]
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    rows = buf[: h * step].reshape(h, step)[:, : w * ch]
    img = rows.reshape(h, w, ch) if ch > 1 else rows.reshape(h, w, 1)
    if enc == "bgr8":
        return img.copy()
    if enc == "rgb8":
        return np.ascontiguousarray(img[:, :, ::-1])
    if enc == "bgra8":
        return np.ascontiguousarray(img[:, :, :3])
    if enc == "rgba8":
        return np.ascontiguousarray(img[:, :, 2::-1])
    return np.repeat(img, 3, axis=2)


def image_to_depth(msg: Any) -> np.ndarray:
    """sensor_msgs/Image 32FC1 -> HxW float32 metres (a copy)."""
    if str(msg.encoding) != "32FC1":
        raise ValueError(f"depth image must be 32FC1, got {msg.encoding!r}")
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    dt = np.dtype(np.float32).newbyteorder(">" if int(msg.is_bigendian) else "<")
    rows = np.frombuffer(msg.data, dtype=np.uint8)[: h * step].reshape(h, step)[:, : w * 4]
    return rows.copy().view(dt).reshape(h, w).astype(np.float32)


def fill_image(msg: Any, arr: np.ndarray, encoding: str) -> Any:
    """Write ``arr`` (HxWx3 uint8 for bgr8, HxW float32 for 32FC1) into an Image message."""
    if encoding == "32FC1":
        a = np.ascontiguousarray(arr, dtype="<f4")
        h, w = a.shape
        msg.step = w * 4
    else:
        ch = _CHANNELS[encoding]
        a = np.ascontiguousarray(arr, dtype=np.uint8)
        h, w = a.shape[:2]
        if (a.shape[2] if a.ndim == 3 else 1) != ch:
            raise ValueError(f"{encoding} needs {ch} channels, got shape {a.shape}")
        msg.step = w * ch
    msg.height, msg.width = int(h), int(w)
    msg.encoding = encoding
    msg.is_bigendian = 0
    # array('B') takes rclpy's fast path for uint8[] (a memcpy); bytes or lists are validated
    # element by element (~1M elements per 800x450 frame).
    msg.data = array("B", a.tobytes())
    return msg


# ---------------------------------------------------------------------------- perception
def _f(v: Any) -> float:
    return NAN if v is None else float(v)


def fill_tracked_object(msg: Any, d: dict[str, Any]) -> Any:
    """``perception.pipeline.object_to_dict`` dict -> TrackedObject."""
    msg.track_id = int(d["track_id"])
    msg.cls = int(d["cls"])
    msg.cls_name = str(d["cls_name"])
    msg.score = _f(d["score"])
    msg.bbox_xyxy = [_f(v) for v in d["bbox_xyxy"]]
    msg.age = int(d["age"])
    msg.distance_m = _f(d["distance_m"])
    msg.distance_std = _f(d["distance_std"])
    msg.closing_speed_mps = _f(d["closing_speed_mps"])
    msg.has_ttc = d["ttc_s"] is not None
    msg.ttc_s = _f(d["ttc_s"])
    msg.in_ego_lane = bool(d["in_ego_lane"])
    return msg


def fill_object_array(msg: Any, frame: dict[str, Any], object_factory: Any) -> Any:
    """``perception.pipeline.frame_to_dict`` dict -> TrackedObjectArray (header left alone)."""
    msg.frame_id = int(frame["frame_id"])
    msg.objects = [fill_tracked_object(object_factory(), o) for o in frame["objects"]]
    lead = frame.get("lead_track_id")
    msg.lead_track_id = -1 if lead is None else int(lead)
    msg.total_ms = _f((frame.get("timings_ms") or {}).get("total_ms"))
    return msg


def fill_lanes(msg: Any, lanes: dict[str, Any], lane_debug: dict[str, Any] | None = None) -> Any:
    """``frame_to_dict(...)["lanes"]`` (+ the lane detector's ``last_debug``) -> LaneBoundaries."""
    dbg = lane_debug or {}
    left, right = lanes.get("left"), lanes.get("right")
    msg.has_left = left is not None
    msg.left_coeffs = [_f(v) for v in left] if left is not None else [0.0, 0.0, 0.0]
    msg.has_right = right is not None
    msg.right_coeffs = [_f(v) for v in right] if right is not None else [0.0, 0.0, 0.0]
    msg.confidence = _f(lanes.get("confidence"))
    msg.has_ego_mask = bool(lanes.get("has_ego_mask"))
    msg.y_top = _f(dbg.get("y_top"))
    msg.y_bottom = _f(dbg.get("y_bottom"))
    return msg


def candidates_from_msg(msg: Any) -> list[Any]:
    """TrackedObjectArray -> ``perception.control.LeadCandidate`` list (NaN distance dropped)."""
    from perception.control import LeadCandidate

    out = []
    for o in msg.objects:
        d = float(o.distance_m)
        if not math.isfinite(d):
            continue
        ttc = float(o.ttc_s) if o.has_ttc and math.isfinite(float(o.ttc_s)) else None
        cs = float(o.closing_speed_mps)
        out.append(
            LeadCandidate(
                track_id=int(o.track_id),
                distance_m=d,
                ttc_s=ttc,
                closing_speed_mps=cs if math.isfinite(cs) else 0.0,
                in_ego_lane=bool(o.in_ego_lane),
            )
        )
    return out


def fill_control(msg: Any, decision: Any, ego_speed_mps: float) -> Any:
    """``perception.control.ControlDecision`` -> EgoControl (header left alone)."""
    msg.override_autopilot = bool(decision.override)
    msg.throttle = float(decision.throttle)
    msg.brake = float(decision.brake)
    msg.mode = str(decision.mode)
    msg.reason = str(decision.reason)
    lead = decision.lead
    msg.lead_track_id = -1 if lead is None else int(lead.track_id)
    msg.lead_distance_m = NAN if lead is None else float(lead.distance_m)
    msg.lead_ttc_s = NAN if lead is None or lead.ttc_s is None else float(lead.ttc_s)
    msg.ego_speed_mps = float(ego_speed_mps)
    return msg
