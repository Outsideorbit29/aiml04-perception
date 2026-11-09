import numpy as np

from perception.types import (
    Detection,
    LaneResult,
    ObjectState,
    PerceptionFrame,
    Polyline,
    Track,
)


def make_obj(track_id: int, x: float, dist: float, in_lane: bool) -> ObjectState:
    det = Detection(np.array([x - 10, 100, x + 10, 200], np.float32), 0.9, 0, "car")
    return ObjectState(Track(track_id, det), dist, 0.5, 0.0, None, in_lane)


def test_bottom_center():
    det = Detection(np.array([10, 20, 30, 60], np.float32), 0.8, 2, "car")
    assert det.bottom_center == (20.0, 60.0)


def test_polyline_straight_line():
    line = Polyline(np.array([0.0, 0.0, 100.0]))
    assert line.x_at(50) == 100.0


def test_lane_contains_uses_polylines():
    left, right = Polyline(np.array([0, 0, 100.0])), Polyline(np.array([0, 0, 300.0]))
    lanes = LaneResult(left, right, None, 1.0)
    assert lanes.contains(200, 150)
    assert not lanes.contains(50, 150)


def test_lane_contains_prefers_mask():
    mask = np.zeros((10, 10), bool)
    mask[5, 5] = True
    lanes = LaneResult(None, None, mask, 1.0)
    assert lanes.contains(5, 5)
    assert not lanes.contains(1, 1)


def test_lead_object_is_closest_in_lane():
    lanes = LaneResult(None, None, None, 0.0)
    frame = PerceptionFrame(
        0.0,
        0,
        [make_obj(1, 0, 30, True), make_obj(2, 0, 12, True), make_obj(3, 0, 5, False)],
        lanes,
    )
    assert frame.lead_object().track.track_id == 2


def test_lead_object_none_when_lane_empty():
    frame = PerceptionFrame(0.0, 0, [make_obj(1, 0, 5, False)], LaneResult(None, None, None, 0.0))
    assert frame.lead_object() is None
