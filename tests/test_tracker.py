"""Tracker (ByteTrack via supervision) on synthetic detections, plus the track overlay."""

from types import SimpleNamespace

import numpy as np
import pytest

from perception.detection import load_config
from perception.tracking import Tracker
from perception.tracking.viz import draw_tracks, id_color
from perception.types import Detection, Track

NAMES = {0: "car", 3: "person"}
# Wrapper-mechanics tests pin the tunables they depend on (ids from the first frame, max_lost 10),
# so retuning configs/tracker.yaml cannot silently change what they check. The shipped defaults
# are tested separately at the end of this file (SHIPPED).
CFG = load_config(
    "configs/tracker.yaml", frame_rate=10, lost_track_buffer=30, minimum_consecutive_frames=1
)


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


# ---------- state pruning follows ByteTrack, not the last output (qa-reviewer M2 item 2) ----------
class _Out:
    """Minimal stand-in for ``sv.Detections`` returned by ``update_with_detections``."""

    def __init__(self, ids: list[int]) -> None:
        self.tracker_id = np.array(ids, int)
        self.data = {"idx": np.arange(len(ids))}

    def __len__(self) -> int:
        return len(self.tracker_id)


class ScriptedByteTrack:
    """Outputs id 1 on ``out`` frames and lists it as live on ``live`` frames.

    Frames that are live but not output mimic supervision updating a track whose Kalman box
    overlaps the detection by IoU <= 0.5: ByteTrack keeps it, but it is not returned.
    """

    def __init__(self, out: set[int], live: set[int], expose: bool = True) -> None:
        self.out, self.live, self.expose, self.f = out, live, expose, 0
        if expose:
            self.tracked_tracks: list = []
            self.lost_tracks: list = []

    def update_with_detections(self, _dets):
        self.f += 1
        if self.expose:
            alive = self.f in self.live
            self.tracked_tracks = [SimpleNamespace(external_track_id=1)] if alive else []
        return _Out([1] if self.f in self.out else [])


def run_scripted(bt: ScriptedByteTrack, frames: int) -> list[list[Track]]:
    tr = Tracker(CFG)  # frame_rate 10, lost_track_buffer 30 -> max_lost 10
    assert tr.max_lost == 10
    tr._bt = bt
    return [tr.update([det(10, 50)], None) for _ in range(frames)]


def test_state_kept_while_bytetrack_holds_the_id_even_without_output():
    out = {1, 2, 3, 30}
    seq = run_scripted(ScriptedByteTrack(out, live=set(range(1, 31))), 30)
    (back,) = seq[29]  # 26 frames without output (> 2 * max_lost), but ByteTrack kept it
    assert back.track_id == 1 and back.age == 29 and len(back.history) == 4


def test_state_dropped_once_bytetrack_drops_the_id():
    seq = run_scripted(ScriptedByteTrack({1, 2, 3, 6}, live={1, 2, 3}), 6)
    (back,) = seq[5]  # gone from ByteTrack at frame 4: any later id 1 is a new track
    assert back.age == 0 and len(back.history) == 1


def test_state_fallback_keeps_two_max_lost_when_internals_hidden():
    (kept,) = run_scripted(ScriptedByteTrack({1, 16}, set(), expose=False), 16)[15]
    assert kept.age == 15 and len(kept.history) == 2  # 15 frames > max_lost, < 2 * max_lost
    (gone,) = run_scripted(ScriptedByteTrack({1, 25}, set(), expose=False), 25)[24]
    assert gone.age == 0


def test_installed_supervision_exposes_live_ids():
    """Guards the internals _live_ids reads; if this fails the 2*max_lost fallback is in use."""
    tr = Tracker(CFG)
    for t in range(3):
        ids = {tk.track_id for tk in tr.update(scene(t), None)}
    live = tr._live_ids()
    assert live is not None and ids <= live


# ---------- shipped defaults (qa-reviewer, M2.6 gate: minimum_consecutive_frames 2, buffer 60) ----
SHIPPED = load_config("configs/tracker.yaml", frame_rate=10)


def _first_output(tracks_per_frame: list[list[Track]], cls: int) -> int | None:
    return next(
        (t for t, tr in enumerate(tracks_per_frame) if any(k.det.cls == cls for k in tr)), None
    )


def test_shipped_new_tracks_appear_on_their_third_matched_frame():
    """consecutive 2: an id is assigned on the 3rd consecutive high-score match, for objects seen
    on frame 1 and for objects born later (consecutive 1 shows late-born ones on the 2nd)."""
    k = int(SHIPPED["minimum_consecutive_frames"])
    tr = Tracker(SHIPPED)
    out = []
    for t in range(10):
        dets = [det(10 + 3 * t, 50)] + ([det(200, 10, cls=3)] if t >= 4 else [])
        out.append(tr.update(dets, None))
    assert _first_output(out, 0) == k  # frame-1 object: frames 0..k-1 hidden
    assert _first_output(out, 3) == 4 + k  # born at t=4
    first = {tk.det.cls: tk for tk in out[4 + k]}
    assert first[3].age == 0 and len(first[3].history) == 1
    ids = [{tk.det.cls: tk.track_id for tk in tracks} for tracks in out[4 + k :]]
    assert all(i == ids[0] for i in ids) and len(set(ids[0].values())) == 2
    car = next(tk for tk in out[-1] if tk.det.cls == 0)
    assert car.age == 9 - k  # age counts from the first output, not the first detection

    legacy = Tracker(CFG)  # consecutive 1: late-born object shows on its 2nd frame
    out1 = [
        legacy.update([det(10 + 3 * t, 50)] + ([det(200, 10, cls=3)] if t >= 4 else []), None)
        for t in range(8)
    ]
    assert _first_output(out1, 0) == 0 and _first_output(out1, 3) == 5


def test_shipped_two_frame_ghost_never_gets_an_id():
    ghost = det(200, 10, cls=3)
    for cfg, expect_ids in ((SHIPPED, 1), (CFG, 2)):
        tr = Tracker(cfg)
        ids: set[int] = set()
        for t in range(10):
            dets = [det(10 + 3 * t, 50)] + ([ghost] if t in (4, 5) else [])
            ids |= {tk.track_id for tk in tr.update(dets, None)}
        assert len(ids) == expect_ids  # only consecutive 1 lets the ghost through


def test_shipped_short_gap_and_low_score_keep_the_id():
    tr = Tracker(SHIPPED)
    for t in range(6):
        tracks = tr.update([det(10 + 3 * t, 50)], None)
    (tid,) = [tk.track_id for tk in tracks]
    for _ in range(2):
        assert tr.update([], None) == []
    (back,) = tr.update([det(10 + 3 * 8, 50)], None)
    assert back.track_id == tid
    (low,) = tr.update([det(10 + 3 * 9, 50, score=0.15)], None)
    assert low.track_id == tid


@pytest.mark.parametrize("fps", [10, 20])
def test_wrapper_buffer_matches_bytetrack(fps):
    """Tracker.max_lost must equal supervision's max_time_lost (both scale the 30-fps buffer)."""
    tr = Tracker({**SHIPPED, "frame_rate": fps})
    assert tr.max_lost == tr._bt.max_time_lost
    assert tr.max_lost == int(fps / 30 * SHIPPED["lost_track_buffer"])
