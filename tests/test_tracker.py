"""Tracker (ByteTrack via supervision) on synthetic detections, plus the track overlay."""

import numpy as np
import pytest

from perception.detection import load_config
from perception.tracking import Tracker
from perception.tracking.viz import draw_tracks, id_color
from perception.types import Detection, Track

NAMES = {0: "car", 3: "person"}
CFG = load_config("configs/tracker.yaml", frame_rate=10)


def det(x: float, y: float, cls: int = 0, score: float = 0.9, w: float = 40, h: float = 30):
    return Detection(np.array([x, y, x + w, y + h], np.float32), score, cls, NAMES[cls])


def scene(t: int) -> list[Detection]:
    """A car moving right and a person moving left, far apart."""
    return [det(10 + 3 * t, 50), det(250 - 2 * t, 150, cls=3)]


def test_ids_persist_across_frames():
    tr = Tracker(CFG)
    seen = []
    for t in range(15):
        tracks = tr.update(scene(t), None)
        assert len(tracks) == 2
        seen.append({tk.det.cls: tk.track_id for tk in tracks})
    assert all(s == seen[0] for s in seen)
    assert seen[0][0] != seen[0][3]


def test_age_history_and_detection_passthrough():
    tr = Tracker({**CFG, "history_len": 3})
    for t in range(5):
        dets = scene(t)
        tracks = tr.update(dets, np.zeros((240, 320, 3), np.uint8))
    car = next(tk for tk in tracks if tk.det.cls == 0)
    assert isinstance(car, Track)
    assert car.det is dets[0]  # the input Detection is passed through unchanged
    assert car.age == 4
    assert len(car.history) == 3 and car.history.maxlen == 3
    np.testing.assert_allclose(car.history[-1], dets[0].bbox_xyxy)
    np.testing.assert_allclose(car.history[0], scene(2)[0].bbox_xyxy)


def test_short_gap_keeps_id():
    tr = Tracker(CFG)
    for t in range(6):
        first = {tk.det.cls: tk.track_id for tk in tr.update(scene(t), None)}
    for _ in range(2):  # both objects missed for two frames
        assert tr.update([], None) == []
    later = {tk.det.cls: tk.track_id for tk in tr.update(scene(8), None)}
    assert later == first


def test_new_object_gets_new_id():
    tr = Tracker(CFG)
    ids: set[int] = set()
    for t in range(8):
        dets = [det(10 + 3 * t, 50)]
        if t >= 4:
            dets.append(det(200, 10, cls=3))
        tracks = tr.update(dets, None)
        ids |= {tk.track_id for tk in tracks}
    assert len(tracks) == 2
    assert len(ids) == 2


def test_reset_and_config_errors():
    tr = Tracker(CFG)
    tr.update(scene(0), None)
    tr.reset()
    assert tr.frame_idx == 0
    with pytest.raises(ValueError):
        Tracker({**CFG, "type": "sort"})
    with pytest.raises(NotImplementedError):
        Tracker({**CFG, "type": "deepsort"})
    with pytest.raises(KeyError):
        Tracker({"type": "bytetrack"})


def test_draw_tracks_does_not_modify_input():
    frame = np.zeros((120, 240, 3), np.uint8)
    tk = Track(7, det(20, 40))
    tk.history.extend([tk.det.bbox_xyxy - 5, tk.det.bbox_xyxy])
    out = draw_tracks(frame, [tk])
    assert out.shape == frame.shape and out.any() and not frame.any()
    assert id_color(1) != id_color(2)


# ---------- added by qa-reviewer (M2.3-M2.5 gate) ----------
def test_detection_order_does_not_swap_track_association():
    """Track.det must be the detection ByteTrack matched, whatever order the detector emits."""
    tr = Tracker(CFG)
    by_id: dict[int, list[np.ndarray]] = {}
    for t in range(10):
        dets = scene(t)
        if t % 2:
            dets = dets[::-1]
        for tk in tr.update(dets, None):
            by_id.setdefault(tk.track_id, []).append(tk.det.bbox_xyxy)
            np.testing.assert_allclose(tk.history[-1], tk.det.bbox_xyxy)
    assert len(by_id) == 2
    for boxes in by_id.values():  # each id follows one smoothly moving object
        steps = np.abs(np.diff(np.stack(boxes)[:, 0]))
        assert steps.max() <= 3.0 + 1e-6


def test_low_score_detection_extends_but_never_starts_a_track():
    """ByteTrack second pass: 0.1 < score < activation threshold keeps an existing id alive."""
    tr = Tracker(CFG)
    for t in range(5):
        (tid,) = [tk.track_id for tk in tr.update([det(10 + 3 * t, 50)], None)]
    low = det(10 + 3 * 5, 50, score=0.15)
    tracks = tr.update([low], None)
    assert [tk.track_id for tk in tracks] == [tid]
    assert tracks[0].det is low and tracks[0].det.score == pytest.approx(0.15)
    assert tracks[0].age == 5

    fresh = Tracker(CFG)
    for _ in range(5):
        assert fresh.update([det(200, 100, cls=3, score=0.15)], None) == []
