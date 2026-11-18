"""perception.control (M5.4): AEB / ACC / gap rules, lead persistence and the HOLD latch."""

from __future__ import annotations

import pytest
import yaml

from perception.control import (
    MODES,
    ControlDecision,
    LeadCandidate,
    LongitudinalController,
    candidates_from_frame,
)
from perception.types import Detection, LaneResult, ObjectState, PerceptionFrame, Track

CFG = yaml.safe_load(open("configs/carla.yaml", encoding="utf-8"))["control"]
N = int(CFG["lead_persist_frames"])


def cand(tid=1, d=30.0, ttc=None, closing=0.0, lane=True) -> LeadCandidate:
    return LeadCandidate(tid, d, ttc, closing, lane)


def run(ctrl: LongitudinalController, objs, frames: int, speed: float = 8.0) -> ControlDecision:
    out = None
    for _ in range(frames):
        out = ctrl.update(list(objs), speed)
    return out


def test_config_matches_architecture_thresholds():
    assert CFG["aeb_ttc_s"] == 2.0 and CFG["aeb_distance_m"] == 8.0 and CFG["acc_ttc_s"] == 4.0
    assert N >= 2  # M4.6: one-frame false leads must not brake


def test_no_objects_is_autopilot():
    d = LongitudinalController(CFG).update([], 8.0)
    assert not d.override and d.mode == "AUTOPILOT" and d.lead is None
    assert d.mode in MODES


def test_single_frame_false_lead_never_brakes():
    ctrl = LongitudinalController(CFG)
    danger = cand(tid=7, d=5.0, ttc=0.5, closing=10.0)
    for _ in range(5):  # the dangerous object flickers: in lane every other frame
        assert not ctrl.update([danger], 8.0).override
        assert not ctrl.update([cand(tid=7, d=5.0, lane=False)], 8.0).override


def test_aeb_fires_after_persistence():
    ctrl = LongitudinalController(CFG)
    danger = cand(d=20.0, ttc=1.5, closing=12.0)
    for _ in range(N - 1):
        assert not ctrl.update([danger], 8.0).override
    d = ctrl.update([danger], 8.0)
    assert d.override and d.mode == "AEB" and d.brake == 1.0 and d.throttle == 0.0
    assert d.lead.track_id == 1


def test_aeb_on_short_distance_without_ttc():
    d = run(LongitudinalController(CFG), [cand(d=6.0, ttc=None)], N)
    assert d.mode == "AEB" and d.brake == 1.0


def test_new_track_id_restarts_persistence():
    ctrl = LongitudinalController(CFG)
    run(ctrl, [cand(tid=1, d=6.0)], N - 1)
    assert not ctrl.update([cand(tid=2, d=6.0)], 8.0).override  # id switch: count from 0


def test_acc_brake_is_proportional_in_ttc_band():
    lo, hi = float(CFG["acc_min_brake"]), float(CFG["acc_max_brake"])
    near = run(LongitudinalController(CFG), [cand(d=30.0, ttc=2.2, closing=13.0)], N)
    far = run(LongitudinalController(CFG), [cand(d=30.0, ttc=3.8, closing=8.0)], N)
    assert near.mode == far.mode == "ACC"
    assert lo <= far.brake < near.brake <= hi
    assert near.throttle == far.throttle == 0.0


def test_gap_rule_only_when_closing():
    gap = float(CFG["gap_min_m"]) + 8.0 * float(CFG["gap_time_s"])  # 18 m at 8 m/s
    d_close = run(LongitudinalController(CFG), [cand(d=gap - 5, ttc=None, closing=0.3)], N)
    d_open = run(LongitudinalController(CFG), [cand(d=gap - 5, ttc=None, closing=-1.0)], N)
    assert d_close.mode == "GAP" and 0 < d_close.brake <= float(CFG["acc_max_brake"])
    assert d_open.mode == "AUTOPILOT" and not d_open.override


def test_hold_latch_keeps_braking_at_standstill_and_on_dropouts():
    ctrl = LongitudinalController(CFG)
    run(ctrl, [cand(d=7.0, ttc=1.0, closing=5.0)], N)
    assert ctrl.latched
    # stopped 9 m behind the obstacle: not AEB any more, but the latch holds the brake
    d = ctrl.update([cand(d=9.0, ttc=None, closing=0.0)], 0.0)
    assert d.mode == "HOLD" and d.brake == float(CFG["hold_brake"])
    # detection dropout: still holding for lead_release_frames
    for _ in range(int(CFG["lead_release_frames"])):
        assert ctrl.update([], 0.0).mode == "HOLD"
    assert ctrl.update([], 0.0).mode == "AUTOPILOT"
    assert not ctrl.latched


def test_hold_releases_when_lead_moves_away():
    ctrl = LongitudinalController(CFG)
    run(ctrl, [cand(d=7.0, ttc=1.0, closing=5.0)], N)
    far = float(CFG["hold_release_distance_m"]) + 5.0
    d = ctrl.update([cand(d=far, ttc=None, closing=-2.0)], 0.0)
    assert d.mode == "AUTOPILOT" and not ctrl.latched


def test_closest_persistent_in_lane_object_is_lead():
    objs = [
        cand(tid=1, d=40.0, ttc=None),
        cand(tid=2, d=25.0, ttc=None),
        cand(tid=3, d=10.0, ttc=None, lane=False),  # not in lane
        cand(tid=4, d=200.0, ttc=None),  # beyond max_lead_distance_m
    ]
    d = run(LongitudinalController(CFG), objs, N)
    assert d.lead is not None and d.lead.track_id == 2


def test_nan_distance_is_ignored():
    d = run(LongitudinalController(CFG), [cand(d=float("nan"), ttc=0.5)], N + 2)
    assert not d.override


def test_config_validation():
    with pytest.raises(KeyError):
        LongitudinalController({"aeb_ttc_s": 2.0})
    with pytest.raises(ValueError):
        LongitudinalController({**CFG, "aeb_ttc_s": 5.0})
    with pytest.raises(ValueError):
        LongitudinalController({**CFG, "lead_persist_frames": 0})


def test_candidates_from_frame():
    import numpy as np

    det = Detection(np.array([1, 2, 3, 4], np.float32), 0.9, 0, "car")
    objs = [
        ObjectState(Track(5, det), 12.0, 0.5, 3.0, 4.0, True),
        ObjectState(Track(6, det), 30.0, 2.0, -1.0, None, False),
    ]
    pf = PerceptionFrame(0.0, 0, objs, LaneResult(None, None, None, 0.0))
    c = candidates_from_frame(pf)
    assert [x.track_id for x in c] == [5, 6]
    assert c[0].ttc_s == 4.0 and c[1].ttc_s is None and c[0].in_ego_lane and not c[1].in_ego_lane
