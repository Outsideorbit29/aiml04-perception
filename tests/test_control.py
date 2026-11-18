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
    "follow_range_m": 0.0,
    "approach_speed_mps": 0.0,
    "brake_rate_up": 1.0,
    "brake_rate_down": 1.0,
}
# The controller as it ran in Session B (2026-11-19): no lead tracking, no approach creep.
SESSION_B = {
    **CFG,
    "follow_range_m": 0.0,
    "approach_speed_mps": 0.0,
    "standstill_hold_distance_m": 15.0,
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
    # margin rule in isolation: a lead beyond follow_range_m is not tracked (rule 7)
    ctrl = LongitudinalController({**CFG, "follow_range_m": 0.0})
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


def _stopped_car_approach(
    cfg: dict,
    seed: int,
    lag_s: float = 0.0,
    brake_gain: float = 9.0,
    bias: float = 1.0,
    lead_x: float = 60.0,
    v0: float = 0.0,
    seconds: float = 40.0,
    log: list | None = None,
    ctrl: LongitudinalController | None = None,
) -> tuple[list[str], float, bool]:
    """Crude point-mass loop: autopilot accelerates to 8.3 m/s, brake 1.0 ~ ``brake_gain``
    m/s^2, throttle 1.0 ~ 6 m/s^2, noisy distance (x ``bias``) and closing speed, 5 % lead
    dropouts, TTC only when closing > 0.6 m/s. ``lag_s`` low-pass filters the measured closing
    speed (the Kalman velocity lags when the ego brakes, so ACC over-brakes, as in Session B).
    ``log`` collects (mode, v, true gap, throttle, brake) per frame."""
    import random

    rng = random.Random(seed)
    ctrl = ctrl or LongitudinalController(cfg)
    x, v, dt, modes = 0.0, v0, 0.05, []
    v_meas = v
    for _ in range(int(seconds / dt)):
        gap = lead_x - x
        if gap <= 0:
            return modes, gap, True
        objs = []
        if rng.random() > 0.05:
            dm = gap * bias * (1 + rng.gauss(0, 0.04))
            v_meas += (v - v_meas) * (min(1.0, dt / lag_s) if lag_s > 0 else 1.0)
            closing = v_meas + rng.gauss(0, 0.3)
            objs = [cand(d=dm, ttc=dm / closing if closing > 0.6 else None, closing=closing)]
        dec = ctrl.update(objs, v)
        modes.append(dec.mode)
        if log is not None:
            log.append((dec.mode, v, gap, dec.throttle, dec.brake))
        if dec.override:
            a = dec.throttle * 6.0 - dec.brake * brake_gain - 0.3
        else:
            a = 2.0 if v < 8.3 else 0.0
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


# --- stop-and-go: lead tracking, approach creep, standstill (pre-demo fixes) -----------------

V_APP = float(CFG["approach_speed_mps"])
HOLD_D = float(CFG["standstill_hold_distance_m"])


def _engaged(cfg: dict) -> LongitudinalController:
    ctrl = LongitudinalController(cfg)
    run(ctrl, [cand(d=25.0, ttc=3.5, closing=7.0)], N + 25)
    assert ctrl.mode == "ACC"
    return ctrl


def _slowing_behind_stationary_lead():
    """Session B stopped_car / cut_in: ACC engaged at ~30 m, the ego brakes to a crawl while the
    stationary lead is still ~16 m away; TTC disappears (closing < 0.6 m/s) and the gap is
    beyond desired gap + margin, so the old exit rule handed back to the autopilot."""
    frames = [([cand(d=30.0, ttc=3.8, closing=8.0)], 8.0)] * N
    d = 30.0
    for i in range(60):
        v = 8.0 - 7.8 * i / 59
        d -= v * 0.05 * 0.6  # not exact kinematics; the distance only has to shrink
        frames.append(([cand(d=d, ttc=d / v if v > 0.6 else None, closing=v)], v))
    frames += [([cand(d=d, ttc=None, closing=0.2)], 0.2)] * 60
    return frames


def test_session_b_early_acc_exit_is_reproduced_and_fixed():
    old = LongitudinalController(SESSION_B)
    assert "AUTOPILOT" in [old.update(o, v).mode for o, v in _slowing_behind_stationary_lead()]
    ctrl = LongitudinalController(CFG)
    out = [ctrl.update(o, v) for o, v in _slowing_behind_stationary_lead()]
    modes = [o.mode for o in out]
    first = modes.index("ACC")
    assert set(modes[first:]) == {"ACC"}  # engaged once, no ACC -> AUTOPILOT -> ACC
    # slow and far from the lead: a gentle approach, never cruise throttle
    assert all(o.override and o.throttle <= float(CFG["approach_throttle"]) for o in out[first:])
    assert out[-1].throttle > 0.0 and out[-1].brake == 0.0 and "approach" in out[-1].reason


def test_tracked_lead_releases_when_far_or_opening():
    far = cand(d=float(CFG["follow_range_m"]) + 5.0, ttc=None, closing=0.0)
    out = run(_engaged(CFG), [far], int(CFG["exit_confirm_frames"]), speed=6.0)
    assert out.mode == "AUTOPILOT"
    opening = cand(d=25.0, ttc=None, closing=-1.0)
    out = run(_engaged(CFG), [opening], int(CFG["exit_confirm_frames"]), speed=6.0)
    assert out.mode == "AUTOPILOT"


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("bias", [1.0, 1.13])  # 1.13: night_rain reads the lead ~13 % long
def test_closed_loop_lagging_closing_speed_approaches_monotonically(seed, bias):
    kw = dict(lag_s=1.5, brake_gain=12.0, bias=bias)
    old, _, _ = _stopped_car_approach(SESSION_B, seed, **kw)
    first = next(i for i, m in enumerate(old) if m != "AUTOPILOT")
    assert "AUTOPILOT" in old[first:]  # reproduces the Session B early exit
    log: list = []
    modes, stop_gap, collided = _stopped_car_approach(CFG, seed, log=log, **kw)
    assert not collided
    first = next(i for i, m in enumerate(modes) if m != "AUTOPILOT")
    assert set(modes[first:]) <= {"ACC", "GAP"}  # one engagement, no AEB needed
    assert transitions(modes[first:]) <= 1  # at most GAP -> ACC
    # once the ego is below the approach speed it never speeds up past it (no lurch)
    slow = next(i for i in range(first, len(log)) if log[i][1] < V_APP)
    assert max(r[1] for r in log[slow:]) <= V_APP + 1e-6
    # it closes up to the gap and stops between the AEB distance and the hold distance
    assert log[-1][1] == 0.0 and log[-1][3] == 0.0
    assert float(CFG["aeb_distance_m"]) <= stop_gap * bias <= HOLD_D


def test_standstill_inside_hold_distance_never_throttles():
    ctrl = _engaged(CFG)
    gap_min = float(CFG["gap_min_m"])
    for i in range(200):
        d = gap_min + (HOLD_D - gap_min) * (i % 20) / 20  # anywhere in [gap_min, hold distance)
        out = ctrl.update([cand(d=d, ttc=None, closing=0.0)], 0.0)
        assert out.mode == "ACC" and out.throttle == 0.0
    assert out.brake >= float(CFG["standstill_brake"])


@pytest.mark.parametrize("seed", range(5))
def test_night_rain_standstill_beyond_hold_distance_creeps_up_to_the_gap(seed):
    """Session B night_rain: ACC stopped the car with the lead read at 17.6 m (15.5 m true);
    the old rule released to AUTOPILOT, i.e. cruise throttle towards a stationary car."""
    stopped = [cand(d=17.6, ttc=None, closing=0.0)]
    assert run(_engaged(SESSION_B), stopped, 30, speed=0.0).mode == "AUTOPILOT"
    ctrl = _engaged(CFG)
    out = [ctrl.update(stopped, 0.0) for _ in range(40)]
    assert all(o.mode == "ACC" and o.override for o in out)
    assert 0.0 < out[-1].throttle <= float(CFG["approach_throttle"]) and out[-1].brake == 0.0
    # closed loop from standstill 15.5 m behind the stationary car (read 13 % long)
    log: list = []
    modes, stop_gap, collided = _stopped_car_approach(
        CFG, seed, bias=1.13, lead_x=15.5, v0=0.0, seconds=20.0, log=log, ctrl=_engaged(CFG)
    )
    assert not collided and set(modes) <= {"ACC", "GAP"}
    assert max(r[1] for r in log) <= V_APP + 1e-6  # creep, never cruise
    assert log[-1][1] == 0.0 and stop_gap < 15.5  # it did close up ...
    assert stop_gap * 1.13 >= float(CFG["aeb_distance_m"])  # ... and stopped before AEB range


def test_approach_config_validation():
    with pytest.raises(ValueError):
        LongitudinalController({**CFG, "standstill_hold_distance_m": float(CFG["gap_min_m"]) - 1})
    with pytest.raises(ValueError):
        LongitudinalController({**CFG, "approach_throttle": 1.5})
    with pytest.raises(KeyError):
        LongitudinalController({k: v for k, v in CFG.items() if k != "follow_range_m"})
