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
    settle = N + int(1.0 / float(CFG["brake_rate_up"])) + 1  # the rate limiter has caught up
    near = run(LongitudinalController(CFG), [cand(d=30.0, ttc=2.2, closing=13.0)], settle)
    far = run(LongitudinalController(CFG), [cand(d=30.0, ttc=3.8, closing=8.0)], settle)
    assert near.mode == far.mode == "ACC"
    assert lo <= far.brake < near.brake <= hi
    assert near.brake == pytest.approx(near.extra["target_brake"])
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


# --- hysteresis, dwell, standstill guard, brake rate limit (M5.4 follow-up) -------------------

NO_HYSTERESIS = {
    **CFG,
    "acc_exit_ttc_s": CFG["acc_ttc_s"],
    "gap_exit_margin_m": 0.0,
    "gap_exit_opening_mps": 0.0,
    "min_engaged_frames": 0,
    "exit_confirm_frames": 1,
    "lead_lost_exit_frames": 0,
    "standstill_speed_mps": 0.0,
    "brake_rate_up": 1.0,
    "brake_rate_down": 1.0,
}


def transitions(modes: list[str]) -> int:
    return sum(a != b for a, b in zip(modes, modes[1:], strict=False))


def chatter_sequence(frames: int = 240):
    """Like the 2026-11-18 stopped_car log: TTC hovering around acc_ttc_s, the distance
    hovering around the desired gap, closing speed sign noise and short lead dropouts."""
    acc = float(CFG["acc_ttc_s"])
    for i in range(frames):
        v = 8.0 - 6.0 * i / frames  # slowing from 8 to 2 m/s
        gap = float(CFG["gap_min_m"]) + v * float(CFG["gap_time_s"])
        d = gap + (0.3 if i % 2 else -0.3)
        ttc = [acc - 0.1, acc + 0.2, acc + 0.6, None][i % 4]
        closing = 0.3 if i % 3 else -0.1
        objs = [] if i % 30 in (15, 16) else [cand(d=d, ttc=ttc, closing=closing)]
        yield objs, v


def test_chatter_sequence_gives_at_most_two_transitions():
    ctrl = LongitudinalController(CFG)
    modes = [ctrl.update(objs, v).mode for objs, v in chatter_sequence()]
    assert "AEB" not in modes and "HOLD" not in modes
    assert transitions(modes) <= 2, modes
    # the same input without hysteresis/dwell chatters (the bug this guards against)
    raw = LongitudinalController(NO_HYSTERESIS)
    raw_modes = [raw.update(objs, v).mode for objs, v in chatter_sequence()]
    assert transitions(raw_modes) > 10


def test_acc_brake_is_rate_limited():
    ctrl = LongitudinalController(CFG)
    up, down = float(CFG["brake_rate_up"]), float(CFG["brake_rate_down"])
    prev = 0.0
    for objs, v in chatter_sequence():
        b = ctrl.update(objs, v).brake
        assert -down - 1e-9 <= b - prev <= up + 1e-9
        prev = b


def test_aeb_fires_on_the_same_frame_while_in_acc():
    """Hysteresis adds no delay into AEB: first frame with ttc < aeb_ttc_s or d < aeb_distance_m."""
    aeb_ttc = float(CFG["aeb_ttc_s"])
    ttcs = [4.5 - 0.1 * i for i in range(40)]  # passes ACC entry, the coast band, then AEB
    expected = next(i for i, t in enumerate(ttcs) if i >= N - 1 and t < aeb_ttc)
    ctrl = LongitudinalController(CFG)
    modes = [ctrl.update([cand(d=30.0, ttc=t, closing=8.0)], 8.0).mode for t in ttcs]
    assert modes.index("AEB") == expected
    assert "ACC" in modes[:expected]
    # distance trigger from the standstill guard
    ctrl = LongitudinalController(CFG)
    run(ctrl, [cand(d=20.0, ttc=3.5, closing=6.0)], N + 5)
    hold = run(ctrl, [cand(d=10.0, ttc=None, closing=0.0)], 30, speed=0.0)
    assert hold.mode == "ACC" and hold.brake >= float(CFG["standstill_brake"])
    d = ctrl.update([cand(d=7.9, ttc=None, closing=0.0)], 0.0)
    assert d.mode == "AEB" and d.brake == 1.0 and ctrl.latched


def test_hold_semantics_unchanged_after_acc():
    ctrl = LongitudinalController(CFG)
    run(ctrl, [cand(d=25.0, ttc=3.5, closing=7.0)], N + 5)
    assert ctrl.mode == "ACC"
    assert ctrl.update([cand(d=7.0, ttc=1.0, closing=5.0)], 5.0).mode == "AEB"
    d = ctrl.update([cand(d=9.0, ttc=None, closing=0.0)], 0.0)
    assert d.mode == "HOLD" and d.brake == float(CFG["hold_brake"])
    for _ in range(int(CFG["lead_release_frames"])):
        assert ctrl.update([], 0.0).mode == "HOLD"
    assert ctrl.update([], 0.0).mode == "AUTOPILOT" and not ctrl.latched


def test_acc_exit_needs_clear_frames_and_min_dwell():
    ctrl = LongitudinalController(CFG)
    entered = run(ctrl, [cand(d=40.0, ttc=3.9, closing=10.0)], N)
    assert entered.mode == "ACC"
    clear = [cand(d=40.0, ttc=float(CFG["acc_exit_ttc_s"]) + 1.0, closing=7.0)]
    dwell = max(int(CFG["min_engaged_frames"]) - 1, int(CFG["exit_confirm_frames"]))
    for _ in range(dwell - 1):
        assert ctrl.update(clear, 7.0).mode == "ACC"
    assert ctrl.update(clear, 7.0).mode == "AUTOPILOT"
    # inside the hysteresis band ACC coasts (override, no throttle) instead of releasing
    ctrl = LongitudinalController(CFG)
    run(ctrl, [cand(d=40.0, ttc=3.9, closing=10.0)], N)
    band = run(ctrl, [cand(d=40.0, ttc=4.5, closing=9.0)], 100)
    assert band.mode == "ACC" and band.override and band.throttle == 0.0 and band.brake == 0.0


def test_gap_exit_needs_margin():
    ctrl = LongitudinalController(CFG)
    gap = float(CFG["gap_min_m"]) + 8.0 * float(CFG["gap_time_s"])
    assert run(ctrl, [cand(d=gap - 1.0, closing=0.5)], N).mode == "GAP"
    inside = cand(d=gap + float(CFG["gap_exit_margin_m"]) - 0.5, closing=0.0)
    assert run(ctrl, [inside], 100).mode == "GAP"  # beyond gap_m but inside the margin: stays
    outside = cand(d=gap + float(CFG["gap_exit_margin_m"]) + 0.5, closing=0.0)
    assert run(ctrl, [outside], int(CFG["exit_confirm_frames"])).mode == "AUTOPILOT"


def test_gap_escalates_to_acc_but_acc_does_not_step_down():
    ctrl = LongitudinalController(CFG)
    gap = float(CFG["gap_min_m"]) + 8.0 * float(CFG["gap_time_s"])
    assert run(ctrl, [cand(d=gap - 1.0, closing=0.5)], N).mode == "GAP"
    assert ctrl.update([cand(d=gap - 1.0, ttc=3.5, closing=5.0)], 8.0).mode == "ACC"
    assert ctrl.update([cand(d=gap - 1.0, ttc=None, closing=0.5)], 8.0).mode == "ACC"


def test_lead_dropout_keeps_mode_then_releases():
    ctrl = LongitudinalController(CFG)
    b = run(ctrl, [cand(d=25.0, ttc=3.0, closing=8.0)], N + 10).brake
    for _ in range(int(CFG["lead_lost_exit_frames"])):
        d = ctrl.update([], 8.0)
        assert d.mode == "ACC" and d.brake == pytest.approx(b)
    assert ctrl.update([], 8.0).mode == "AUTOPILOT"


def test_standstill_guard_releases_when_lead_moves_away():
    ctrl = LongitudinalController(CFG)
    run(ctrl, [cand(d=20.0, ttc=3.5, closing=6.0)], N + 5)
    assert run(ctrl, [cand(d=10.0, closing=0.0)], 200, speed=0.0).mode == "ACC"  # no creep
    far = cand(d=float(CFG["standstill_hold_distance_m"]) + 5.0, closing=-1.0)
    assert run(ctrl, [far], int(CFG["exit_confirm_frames"]), speed=0.0).mode == "AUTOPILOT"


def test_hysteresis_config_validation():
    with pytest.raises(ValueError):
        LongitudinalController({**CFG, "acc_exit_ttc_s": float(CFG["acc_ttc_s"]) - 0.5})
    with pytest.raises(ValueError):
        LongitudinalController({**CFG, "brake_rate_up": 0.0})
    with pytest.raises(KeyError):
        LongitudinalController({k: v for k, v in CFG.items() if k != "min_engaged_frames"})


def _stopped_car_approach(cfg: dict, seed: int) -> tuple[list[str], float, bool]:
    """Crude point-mass loop: autopilot accelerates to 8.3 m/s, brake 1.0 ~ 9 m/s^2, noisy
    distance/closing speed, 5 % lead dropouts, TTC only when closing > 0.6 m/s."""
    import random

    rng = random.Random(seed)
    ctrl = LongitudinalController(cfg)
    x, v, dt, lead_x, modes = 0.0, 0.0, 0.05, 60.0, []
    for _ in range(int(40 / dt)):
        gap = lead_x - x
        if gap <= 0:
            return modes, gap, True
        objs = []
        if rng.random() > 0.05:
            dm = gap * (1 + rng.gauss(0, 0.04))
            closing = v + rng.gauss(0, 0.3)
            objs = [cand(d=dm, ttc=dm / closing if closing > 0.6 else None, closing=closing)]
        dec = ctrl.update(objs, v)
        modes.append(dec.mode)
        a = -(dec.brake * 9.0 + 0.3) if dec.override else (2.0 if v < 8.3 else 0.0)
        v = max(0.0, v + a * dt)
        x += v * dt
    return modes, lead_x - x, False


@pytest.mark.parametrize("seed", range(5))
def test_closed_loop_stopped_car_has_few_transitions(seed):
    modes, stop_gap, collided = _stopped_car_approach(CFG, seed)
    assert not collided and stop_gap > 0.0
    # gap_min_m (10 m) > aeb_distance_m (8 m): ACC/GAP normally stop the car before AEB is
    # needed; if AEB/HOLD does latch, it must stay latched.
    latch = next((i for i, m in enumerate(modes) if m in ("AEB", "HOLD")), len(modes) - 1)
    assert transitions(modes[: latch + 1]) <= 3
    if modes[latch] in ("AEB", "HOLD"):
        assert set(modes[latch:]) <= {"AEB", "HOLD"}
    else:
        assert stop_gap > CFG["aeb_distance_m"] * 0.9  # smooth stop near the gap target
    raw, _, _ = _stopped_car_approach(NO_HYSTERESIS, seed)
    assert transitions(raw) > transitions(modes)


def test_closed_loop_aeb_backup_with_small_gap():
    # With the old 6 m gap target (inside the 8 m AEB distance) the approach still ends in AEB.
    modes, stop_gap, collided = _stopped_car_approach({**CFG, "gap_min_m": 6.0}, 0)
    assert not collided and stop_gap > 0.0 and "AEB" in modes
