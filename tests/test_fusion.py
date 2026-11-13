"""Kalman filter bank and DistanceFusion on synthetic tracks (M4.3, M4.5).

Parameters are pinned here (not read from configs/depth.yaml) so retuning the shipped config
cannot change what these tests check.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.depth import CameraModel
from perception.fusion import (
    DistanceFusion,
    EgoCorridor,
    FilterBank,
    fuse,
    fuse_inverse,
    in_ego_lane,
    lateral_offset,
)
from perception.types import Detection, LaneResult, Polyline, Track

KF = {
    "accel_std": 0.5,
    "init_velocity_std": 10.0,
    "gate_chi2": 16.0,
    "max_gap_s": 1.0,
    "max_age_s": 2.0,
    "min_updates_for_ttc": 3,
    "min_track_s_for_ttc": 0.5,
    "min_closing_speed_mps": 0.5,
    "ttc_speed_sigma_k": 2.0,
    "gate_restart_count": 2,
    "restart_on_class_change": False,
    "reacquire_restart_gap_s": 0.25,
}
CAM = CameraModel(800, 450, 400.0, 400.0, 400.0, 225.0, 1.5, 0.0, 1.0, 0.0)
CFG = {
    "geometry": {"min_angle_deg": 0.5, "bottom_margin_px": 2, "max_distance_m": 80.0},
    "sampling": {"rows_frac": [0.5, 1.0], "cols_frac": [0.25, 0.75], "percentile": 50},
    "anchors": {"min_score": 0.5, "max_distance_m": 40.0, "min_box_h_px": 10},
    "alignment": {
        "min_points_affine": 3,
        "allow_scale_only": True,
        "huber_k": 1.5,
        "reject_k": 3.0,
        "irls_iters": 5,
        "sigma_floor": 1e-4,
        "min_inv_depth": 1e-3,
        "min_spread_ratio": 1.5,
        "no_dof_inflation": 2.0,
    },
    "fusion": {
        "object_classes": ["car", "person"],
        "reuse_alignment_frames": 2,
        "correlation": 0.0,
        "ego_lane": {
            "min_lane_confidence": 0.3,
            "half_width_lanes_m": 1.5,
            "curve_radius_m": 400.0,
            "half_width_no_lanes_m": 1.2,
            "range_no_lanes_m": 30.0,
        },
    },
    "kalman": KF,
}
HW = (450, 800)


# ------------------------------------------------------------------ Kalman


def test_ttc_on_constant_speed_approach():
    bank = FilterBank(KF)
    dt, v = 0.05, 10.0  # 20 Hz, closing at 10 m/s from 50 m
    rng = np.random.default_rng(0)
    out = None
    for k in range(60):
        d = 50.0 - v * k * dt
        out = bank.update(1, 0, k * dt, d + rng.normal(0, 0.2), 0.2)
    d_true = 50.0 - v * 59 * dt
    assert out.distance_m == pytest.approx(d_true, abs=0.3)
    assert out.closing_speed_mps == pytest.approx(v, abs=0.5)
    assert out.ttc_s == pytest.approx(d_true / v, rel=0.1)
    assert out.distance_std < 0.2


def test_ttc_none_when_receding_or_too_new():
    bank = FilterBank(KF)
    first = bank.update(1, 0, 0.0, 20.0, 0.3)
    assert first.restarted and first.ttc_s is None and first.closing_speed_mps == 0.0
    for k in range(1, 20):
        out = bank.update(1, 0, 0.1 * k, 20.0 + 0.5 * k, 0.3)  # moving away at 5 m/s
    assert out.closing_speed_mps < 0 and out.ttc_s is None


def test_id_gap_switch_and_prune():
    bank = FilterBank(KF)
    for k in range(5):
        bank.update(7, 0, 0.1 * k, 30.0 - 0.5 * k, 0.3)
    # short gap (tracker lost it for 3 frames): same filter continues, prediction in between
    assert bank.predict_only(7, 0.6) is not None
    out = bank.update(7, 0, 0.7, 30.0 - 0.5 * 7, 0.3)
    assert not out.restarted
    # id switch to an object 15 m further away: the first jump is gated out (coast), the second
    # consecutive one restarts the filter on the new object
    out = bank.update(7, 0, 0.8, 30.0 - 0.5 * 8 + 15.0, 0.3)
    assert out.coasted and not out.restarted and out.distance_m == pytest.approx(26.0, abs=0.3)
    out = bank.update(7, 0, 0.9, 30.0 - 0.5 * 9 + 15.0, 0.3)
    assert out.restarted and out.distance_m == pytest.approx(40.5)
    assert bank.restarts == 1 and bank.coasts == 1
    # class change restarts at once only when configured
    strict = FilterBank({**KF, "restart_on_class_change": True})
    strict.update(7, 0, 0.0, 40.0, 0.3)
    assert strict.update(7, 3, 0.1, 40.0, 0.3).restarted
    # long gap: restart; unknown id: no prediction; stale ids pruned
    assert bank.update(7, 3, 2.5, 41.0, 0.3).restarted
    assert bank.predict_only(99, 2.5) is None
    bank.update(8, 0, 2.6, 10.0, 0.3)
    bank.prune(4.55)
    assert 7 not in bank and 8 in bank
    # time going backwards (new clip without reset) restarts instead of predicting backwards
    assert bank.update(8, 0, 0.0, 10.0, 0.3).restarted
    bank.reset()
    assert not bank.filters


def test_single_outlier_keeps_closing_speed_and_ttc():
    bank = FilterBank(KF)
    dt, v = 0.05, 8.0
    for k in range(40):
        out = bank.update(1, 0, k * dt, 40.0 - v * k * dt, 0.2)
    before = out
    # one frame where the distance doubles (bad box / MiDaS): coast on the prediction
    out = bank.update(1, 0, 40 * dt, 2.0 * (40.0 - v * 40 * dt), 0.2)
    assert out.coasted and not out.restarted
    assert out.closing_speed_mps == pytest.approx(v, abs=0.5) and out.ttc_s is not None
    out = bank.update(1, 0, 41 * dt, 40.0 - v * 41 * dt, 0.2)  # back to normal: no restart
    assert not out.restarted and not out.coasted
    assert out.ttc_s == pytest.approx((40.0 - v * 41 * dt) / v, rel=0.1)
    assert before.ttc_s is not None and bank.restarts == 0


def test_class_flip_keeps_filter_state():
    """car -> truck -> car on one track (ByteTrack ignores class): closing speed survives."""
    bank = FilterBank(KF)
    dt, v = 0.05, 6.0
    for k in range(40):
        cls = 1 if k in (30, 31, 35) else 0
        out = bank.update(1, cls, k * dt, 30.0 - v * k * dt, 0.2)
        assert not (k > 0 and out.restarted)
    assert out.closing_speed_mps == pytest.approx(v, abs=0.5) and out.ttc_s is not None
    assert bank.restarts == 0


def test_fuse_inverse_variance_in_inverse_depth():
    assert fuse([None, None]) is None and fuse_inverse([None]) is None
    # equal inverse-depth stds: the mean of 1/10 and 1/20, std sqrt(1/2) of either
    k = 0.004
    z, s = fuse_inverse([(1 / 10, k), (1 / 20, k)])
    assert z == pytest.approx(1 / 0.075) and s == pytest.approx(z * z * k / np.sqrt(2))
    # fuse() takes distance estimates with std(Z) = Z^2 * k and gives the same answer
    assert fuse([(10.0, 100 * k), (20.0, 400 * k)]) == pytest.approx((z, s))
    # unequal: weights 1/std^2 in inverse depth
    z, _ = fuse_inverse([(0.1, k), (0.05, 2 * k)])
    assert 1 / z == pytest.approx((4 * 0.1 + 0.05) / 5)
    # a single estimate is returned unchanged
    assert fuse([(10.0, 1.0), None]) == (10.0, 1.0)
    assert fuse([(10.0, 1.0), (np.nan, 1.0), (5.0, 0.0)]) == (10.0, 1.0)


def test_fuse_correlation_inflates_std_only():
    k = 0.004
    est = [(1 / 10, k), (1 / 12, k)]
    z0, s0 = fuse_inverse(est, 0.0)
    z5, s5 = fuse_inverse(est, 0.5)
    z1, s1 = fuse_inverse(est, 1.0)
    assert z0 == z5 == z1  # the fused distance does not depend on the correlation
    assert s0 == pytest.approx(z0 * z0 * k / np.sqrt(2))
    assert s5 == pytest.approx(z0 * z0 * k * np.sqrt(0.75))  # sqrt((1 + rho) / 2)
    assert s1 == pytest.approx(z0 * z0 * k)  # fully correlated: no gain over either
    with pytest.raises(ValueError, match="correlation"):
        fuse_inverse(est, 1.5)
    with pytest.raises(ValueError, match="correlation"):
        DistanceFusion({**CFG, "fusion": {**CFG["fusion"], "correlation": -0.1}}, CAM)


def test_distance_fusion_rejects_zero_sigma_camera():
    flat = CameraModel(800, 450, 400.0, 400.0, 400.0, 225.0, 1.5, 0.0)  # sigmas default to 0
    with pytest.raises(ValueError, match="sigma"):
        DistanceFusion(CFG, flat)


# ------------------------------------------------------------------ DistanceFusion


def box_at(z: float, x_center: float = 400.0, w: float = 60.0, h: float = 50.0) -> np.ndarray:
    v = CAM.row_of_distance(z)
    return np.array([x_center - w / 2, v - h, x_center + w / 2, v], np.float32)


def track(tid: int, z: float, cls_name: str = "car", x: float = 400.0, score: float = 0.9):
    cls = {"car": 0, "person": 3, "traffic light": 7}[cls_name]
    return Track(tid, Detection(box_at(z, x), score, cls, cls_name))


def inv_depth_for(tracks: list[Track], s: float, t: float) -> np.ndarray:
    """Synthetic MiDaS map: each box filled with m such that 1/Z = s m + t."""
    inv = np.zeros(HW, np.float32)
    for tr in sorted(tracks, key=lambda t_: -float(t_.det.bbox_xyxy[3] - 225)):
        x1, y1, x2, y2 = tr.det.bbox_xyxy.astype(int)
        z = CAM.ground_distance(float(tr.det.bbox_xyxy[3]))[0]
        inv[y1 : y2 + 1, x1 : x2 + 1] = (1.0 / z - t) / s
    return inv


def straight_lane() -> LaneResult:
    return LaneResult(
        Polyline(np.array([0.0, 0.0, 300.0])), Polyline(np.array([0.0, 0.0, 500.0])), None, 0.9
    )


def test_geometric_distance_and_in_ego_lane():
    fusion = DistanceFusion(CFG, CAM)
    tracks = [track(1, 12.0, x=400.0), track(2, 20.0, x=650.0), track(3, 15.0, "traffic light")]
    frame = fusion.update(tracks, straight_lane(), 0.0, 0, HW)
    by_id = {o.track.track_id: o for o in frame.objects}
    assert set(by_id) == {1, 2}  # traffic light filtered out by class
    assert by_id[1].distance_m == pytest.approx(12.0, rel=1e-3)
    assert by_id[1].in_ego_lane and not by_id[2].in_ego_lane
    assert frame.lead_object().track.track_id == 1
    assert frame.frame_id == 0 and "fusion_ms" in frame.timings_ms
    assert by_id[1].ttc_s is None  # first measurement


def test_in_ego_lane_without_lanes_uses_the_corridor():
    """No lane result: a car straight ahead is still the lead (corridor only); one 3 m to the
    side or beyond range_no_lanes_m is not."""
    tracks = [track(1, 12.0), track(2, 12.0, x=400 + 3.0 * 400 / 12.0), track(3, 35.0)]
    frame = DistanceFusion(CFG, CAM).update(tracks, None, 0.0, 0, HW)
    by_id = {o.track.track_id: o.in_ego_lane for o in frame.objects}
    assert by_id == {1: True, 2: False, 3: False}
    assert frame.lead_object().track.track_id == 1


def test_midas_alignment_covers_truncated_object():
    """A close truck cut by the image bottom has no geometric distance; aligned MiDaS gives it."""
    fusion = DistanceFusion(CFG, CAM)
    tracks = [track(1, 8.0, x=150), track(2, 15.0, x=400), track(3, 25.0, x=650)]
    near = Track(4, Detection(np.array([300, 330, 500, 449], np.float32), 0.9, 0, "car"))
    s, t = 0.004, 0.01
    inv = inv_depth_for(tracks, s, t)
    inv[400:450, 350:450] = (1.0 / 4.0 - t) / s  # truck 4 m away
    frame = fusion.update([*tracks, near], straight_lane(), 0.0, 0, HW, inv_depth=inv)
    by_id = {o.track.track_id: o for o in frame.objects}
    assert by_id[4].distance_m == pytest.approx(4.0, rel=0.02)
    for tid, z in ((1, 8.0), (2, 15.0), (3, 25.0)):
        assert by_id[tid].distance_m == pytest.approx(z, rel=0.01)
    assert fusion.last_stats["anchors"] == 3 and fusion.last_stats["aligned"] == 1
    ms4 = next(m for m in fusion.last_measurements if m.det is near.det)
    assert ms4.geometric is None and ms4.midas is not None and not ms4.anchor


def test_objects_without_any_estimate_are_dropped_then_predicted():
    fusion = DistanceFusion(CFG, CAM)
    near = Track(4, Detection(np.array([300, 330, 500, 449], np.float32), 0.9, 0, "car"))
    frame = fusion.update([near], None, 0.0, 0, HW)  # no depth map, truncated: nothing
    assert frame.objects == [] and fusion.last_stats["no_estimate"] == 1
    fusion.update([track(5, 10.0)], None, 0.0, 0, HW)
    # same id later truncated without a depth map: Kalman prediction keeps it
    cut = Track(5, Detection(np.array([300, 330, 500, 449], np.float32), 0.9, 0, "car"))
    frame = fusion.update([cut], None, 0.1, 1, HW)
    assert len(frame.objects) == 1 and frame.objects[0].distance_m == pytest.approx(10.0)


def test_leave_one_out_scores_anchors_without_themselves():
    fusion = DistanceFusion(CFG, CAM)
    zs = (6.0, 10.0, 15.0, 22.0, 30.0)
    tracks = [track(i, z, x=80 + 150 * i) for i, z in enumerate(zs)]
    dets = [t.det for t in tracks]
    inv = inv_depth_for(tracks, 0.004, 0.01)
    x1, y1, x2, y2 = dets[2].bbox_xyxy.astype(int)
    inv[y1 : y2 + 1, x1 : x2 + 1] *= 1.3  # MiDaS wrong on object 2 (too close)
    loo, align = fusion.measure(dets, HW, inv, leave_one_out=True)
    assert all(m.anchor for m in loo) and align.n_inliers == 4
    for i in (0, 1, 3, 4):  # each good anchor, scored by a fit on the 4 others
        assert loo[i].midas[0] == pytest.approx(zs[i], rel=0.01)
    # the bad object's own geometric distance cannot pull its MiDaS estimate back to 15 m
    assert abs(loo[2].midas[0] - 15.0) / 15.0 > 0.15
    assert loo[2].geometric[0] == pytest.approx(15.0, rel=1e-3)


def test_reset_and_alignment_reuse():
    fusion = DistanceFusion(CFG, CAM)
    tracks = [track(1, 8.0, x=150), track(2, 15.0, x=400), track(3, 25.0, x=650)]
    inv = inv_depth_for(tracks, 0.004, 0.01)
    fusion.update(tracks, None, 0.0, 0, HW, inv_depth=inv)
    assert fusion.last_alignment is not None
    # next frame: only a truncated object (no anchors) -> last fit is reused
    near = Track(4, Detection(np.array([300, 330, 500, 449], np.float32), 0.9, 0, "car"))
    inv2 = np.full(HW, (1.0 / 4.0 - 0.01) / 0.004, np.float32)
    frame = fusion.update([near], None, 0.1, 1, HW, inv_depth=inv2)
    assert frame.objects[0].distance_m == pytest.approx(4.0, rel=0.01)
    fusion.reset()
    assert fusion.last_alignment is None and not fusion.bank.filters
    assert fusion.update([near], None, 0.2, 2, HW).objects == []


# ------------------------------------------------------------------ QA additions (M4 gate)


def test_empty_frame_with_depth_map_then_track():
    """Tracker returns nothing (e.g. right after Tracker.reset()) while MiDaS ran: no crash, empty
    frame, and an id that shows up later still gets a distance."""
    fusion = DistanceFusion(CFG, CAM)
    inv = np.full(HW, 100.0, np.float32)
    frame = fusion.update([], None, 0.0, 0, HW, inv_depth=inv)
    assert frame.objects == [] and fusion.last_stats["aligned"] == 0
    frame = fusion.update([track(3, 12.0)], None, 0.1, 1, HW, inv_depth=None)
    assert len(frame.objects) == 1 and frame.objects[0].distance_m == pytest.approx(12.0, rel=1e-3)


def test_nan_depth_in_box_is_not_an_anchor_and_distances_stay_finite():
    fusion = DistanceFusion(CFG, CAM)
    tracks = [track(1, 8.0, x=150), track(2, 15.0, x=400), track(3, 25.0, x=650)]
    near = Track(4, Detection(np.array([300, 330, 500, 449], np.float32), 0.9, 0, "car"))
    inv = inv_depth_for(tracks, 0.004, 0.01)
    x1, y1, x2, y2 = tracks[1].det.bbox_xyxy.astype(int)
    inv[y1 : y2 + 1, x1 : x2 + 1] = np.nan  # MiDaS invalid on object 2
    frame = fusion.update([*tracks, near], None, 0.0, 0, HW, inv_depth=inv)
    ms = {m.det.bbox_xyxy.tobytes(): m for m in fusion.last_measurements}
    m2 = ms[tracks[1].det.bbox_xyxy.tobytes()]
    assert not m2.anchor and m2.midas is None and m2.geometric is not None
    assert fusion.last_stats["anchors"] == 2
    assert all(np.isfinite(o.distance_m) and np.isfinite(o.distance_std) for o in frame.objects)
    by_id = {o.track.track_id: o for o in frame.objects}
    assert by_id[2].distance_m == pytest.approx(15.0, rel=1e-3)  # geometric only


def test_leave_one_out_does_not_see_the_scored_anchor():
    """An anchor's leave-one-out MiDaS distance must not change when its own geometric distance
    changes (no leakage of the scored object into its fit)."""
    fusion = DistanceFusion(CFG, CAM)
    zs = (6.0, 10.0, 15.0, 22.0, 30.0)
    tracks = [track(i, z, x=80 + 150 * i) for i, z in enumerate(zs)]
    dets = [t.det for t in tracks]
    inv = inv_depth_for(tracks, 0.004, 0.01)
    x1, y1, x2, y2 = dets[2].bbox_xyxy.astype(int)
    inv[y1 - 20 : y2 + 21, x1 : x2 + 1] = inv[y2, x1]  # MiDaS value of object 2 around its box
    base, _ = fusion.measure(dets, HW, inv, leave_one_out=True)
    moved = [*dets]
    moved[2] = Detection(dets[2].bbox_xyxy + np.array([0, 4, 0, 4], np.float32), 0.9, 0, "car")
    shifted, _ = fusion.measure(moved, HW, inv, leave_one_out=True)
    assert shifted[2].anchor and shifted[2].geometric[0] < base[2].geometric[0] - 0.5
    assert shifted[2].midas[0] == pytest.approx(base[2].midas[0], rel=1e-9)
    # contrast: plain least squares (robust rejection off) without leave-one-out lets the scored
    # anchor's own geometric distance move its MiDaS distance
    plain = {**CFG, "alignment": {**CFG["alignment"], "huber_k": 1e6, "reject_k": 1e6}}
    ls = DistanceFusion(plain, CAM)
    a, _ = ls.measure(dets, HW, inv)
    b, _ = ls.measure(moved, HW, inv)
    assert a[2].midas[0] != pytest.approx(b[2].midas[0], rel=1e-6)


def test_leave_one_out_with_three_and_one_anchors():
    fusion = DistanceFusion(CFG, CAM)
    tracks = [track(1, 8.0, x=150), track(2, 15.0, x=400), track(3, 25.0, x=650)]
    dets = [t.det for t in tracks]
    inv = inv_depth_for(tracks, 0.004, 0.0)  # zero shift: scale-only fits are exact
    meas, align = fusion.measure(dets, HW, inv, leave_one_out=True)
    assert align.mode == "affine" and align.n_anchors == 3
    for ms, z in zip(meas, (8.0, 15.0, 25.0), strict=True):
        assert ms.align_mode == "scale" and ms.align_anchors == 2  # 3 - 1 < min_points_affine
        assert ms.midas[0] == pytest.approx(z, rel=1e-3)
    lone, _ = fusion.measure(dets[:1], HW, inv, leave_one_out=True)
    assert lone[0].anchor and lone[0].midas is None and lone[0].align_mode is None


def test_closing_speed_with_irregular_stamps():
    """dt comes from the stamps: dropped/late frames do not change the speed estimate (m/s)."""
    bank = FilterBank(KF)
    v, t, out = 7.0, 0.0, None
    steps = [0.05, 0.10, 0.05, 0.15] * 12
    for dt in steps:
        out = bank.update(1, 0, t, 45.0 - v * t, 0.1)
        t += dt
    assert out.closing_speed_mps == pytest.approx(v, abs=0.3)
    assert out.ttc_s == pytest.approx(out.distance_m / v, rel=0.05)


def test_fused_distance_is_not_biased_short():
    rng = np.random.default_rng(0)
    k, z = 0.0055, 45.0  # inverse-depth std (1/m) of the KITTI camera; object at 45 m
    out = []
    for _ in range(4000):
        a = 1.0 / (1.0 / z + rng.normal(0, k))
        b = 1.0 / (1.0 / z + rng.normal(0, 2 * k))
        if a > 0 and b > 0:
            out.append(fuse([(a, a * a * k), (b, b * b * 2 * k)])[0])
    assert abs(np.median(out) / z - 1.0) < 0.03


def test_measure_fuses_in_inverse_depth():
    """A far, non-anchor object where aligned MiDaS says 50 m and geometry 45 m, with equal
    inverse-depth stds: the fused distance is the harmonic mean, not pulled to the shorter one."""
    fusion = DistanceFusion(CFG, CAM)
    tracks = [track(1, 8.0, x=100), track(2, 15.0, x=250), track(3, 25.0, x=400)]
    far = track(4, 45.0, x=650)
    s, t = 0.004, 0.01
    inv = inv_depth_for(tracks, s, t)
    x1, y1, x2, y2 = far.det.bbox_xyxy.astype(int)
    inv[y1 : y2 + 1, x1 : x2 + 1] = (1.0 / 50.0 - t) / s
    meas, align = fusion.measure([tr.det for tr in [*tracks, far]], HW, inv)
    ms = meas[3]
    assert not ms.anchor and ms.geometric[0] == pytest.approx(45.0, rel=1e-3)
    assert ms.midas[0] == pytest.approx(50.0, rel=1e-3)
    k = CAM.inv_depth_std()
    assert align.inv_std(k) == pytest.approx(k)  # exact fit: floored at the camera's noise
    z = 1.0 / ((1 / 45.0 + 1 / 50.0) / 2)
    assert ms.fused[0] == pytest.approx(z, rel=1e-3)
    assert ms.fused[1] == pytest.approx(z * z * k / np.sqrt(2), rel=1e-3)
    # correlated errors: same distance, larger std
    corr = DistanceFusion({**CFG, "fusion": {**CFG["fusion"], "correlation": 0.8}}, CAM)
    ms_c = corr.measure([tr.det for tr in [*tracks, far]], HW, inv)[0][3]
    assert ms_c.fused[0] == pytest.approx(ms.fused[0], rel=1e-12)
    assert ms_c.fused[1] == pytest.approx(ms.fused[1] * np.sqrt(1.8), rel=1e-3)


def test_stationary_target_never_gets_short_ttc():
    """A parked car at 10 or 20 m, measured at 20 Hz with realistic noise (white in inverse depth,
    std 0.0055 1/m = KITTI geometric only, the pessimistic case; R from the measured distance as
    in DistanceFusion), must never get TTC < 2 s (that would trigger AEB). Parameters pinned
    here; accel_std is the loose runtime value, which is what exposes the start-up problem."""
    kf = {
        **KF,
        "accel_std": 3.0,
        "min_updates_for_ttc": 3,
        "min_track_s_for_ttc": 0.75,
        "ttc_speed_sigma_k": 2.0,
    }
    old = {**kf, "min_track_s_for_ttc": 0.0, "ttc_speed_sigma_k": 0.0}  # 3 updates only

    def min_ttc(cfg: dict, z: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        bank, k, best = FilterBank(cfg), 0.0055, np.inf
        for i in range(60):  # 3 s at 20 Hz
            zm = 1.0 / (1.0 / z + rng.normal(0, k))
            out = bank.update(1, 0, 0.05 * i, zm, zm * zm * k)
            if out.ttc_s is not None:
                best = min(best, out.ttc_s)
        return best

    for z in (10.0, 20.0):
        new = [min_ttc(kf, z, s) for s in range(200)]
        assert min(new) >= 2.0, (z, min(new))
        # the same seeds with the old count-only gate do raise false alarms
        assert min(min_ttc(old, z, s) for s in range(200)) < 2.0


def test_coasting_never_goes_below_zero():
    """6 m away closing at 8 m/s, then no measurements: the prediction stops at contact (0 m,
    TTC 0) instead of going negative."""
    # reacquire restart off, so the gated measurement at the end coasts (tested separately)
    bank = FilterBank({**KF, "reacquire_restart_gap_s": 10.0})
    dt, v = 0.05, 8.0
    for i in range(21):  # 1 s of measurements, ends at 6 m
        out = bank.update(1, 0, i * dt, 14.0 - v * i * dt, 0.1)
    assert out.distance_m == pytest.approx(6.0, abs=0.1) and out.ttc_s is not None
    for j in range(1, 19):  # coast for 0.9 s (would reach 6 - 7.2 = -1.2 m)
        pred = bank.predict_only(1, 1.0 + j * dt)
        assert pred is not None and pred.distance_m >= 0.0
    assert pred.distance_m == 0.0 and pred.ttc_s == 0.0
    # a gated (outlier) measurement after crossing also coasts at >= 0
    out = bank.update(1, 0, 1.95, 30.0, 0.1)
    assert out.coasted and out.distance_m == 0.0
    # with the default reacquire rule the same late measurement restarts the filter instead
    late = FilterBank(KF)
    for i in range(21):
        late.update(1, 0, i * dt, 14.0 - v * i * dt, 0.1)
    out = late.update(1, 0, 1.95, 30.0, 0.1)
    assert out.restarted and out.distance_m == 30.0


# ------------------------------------------------------------------ in_ego_lane corridor (M4.5)

CORRIDOR = EgoCorridor.from_dict(CFG["fusion"]["ego_lane"])


def project_box(
    x_in: float, x_out: float, z_near: float, length: float = 4.5, height: float = 1.5
) -> np.ndarray:
    """Image box (clipped to the frame) of a kerb-parallel car: lateral x_in..x_out (m, + right),
    forward z_near..z_near+length, on flat ground under CAM (zero pitch)."""
    pts = []
    for x in (x_in, x_out):
        for z in (z_near, z_near + length):
            for y in (CAM.height_m, CAM.height_m - height):  # camera y down: ground, roof
                pts.append((CAM.cx + CAM.fx * x / z, CAM.cy + CAM.fy * y / z))
    u, v = np.array(pts).T
    h, w = HW
    return np.array(
        [max(u.min(), 0), max(v.min(), 0), min(u.max(), w - 1), min(v.max(), h - 1)], np.float32
    )


def one_sided_spilled_lane() -> LaneResult:
    """Left boundary measured at x=300; the synthesized right side spilled to the image edge, so
    the ego mask covers the kerb and the parked cars on the right (rows 250-449)."""
    mask = np.zeros(HW, bool)
    mask[250:, 300:] = True
    return LaneResult(Polyline(np.array([0.0, 0.0, 300.0])), None, mask, 0.6)


def det(box: np.ndarray) -> Detection:
    return Detection(box, 0.9, 0, "car")


def test_lateral_offset_matches_projection_and_pitch():
    assert lateral_offset(400 + 400 * 3.0 / 12.0, 12.0, CAM) == pytest.approx(3.0)
    assert lateral_offset(400 - 40.0, 10.0, CAM) == pytest.approx(-1.0)
    pitched = CameraModel(800, 450, 400.0, 400.0, 400.0, 225.0, 1.5, np.radians(5.0), 1.0, 0.0)
    zc = 10.0 * np.cos(np.radians(5.0)) + 1.5 * np.sin(np.radians(5.0))
    assert lateral_offset(440.0, 10.0, pitched) == pytest.approx(0.1 * zc)


def test_parked_car_box_centre_offset():
    """The box of a car parked 2.0-3.8 m to the right spans its near outer to its far inner
    corner: its centre offset is ~2.2-2.5 m (the configured half-widths sit below that), while
    its inner edge offset (~1.1 m) would wrongly fall inside any useful corridor."""
    box = project_box(2.0, 3.8, 6.0)
    u = float(box[0] + box[2]) / 2
    assert 2.2 < lateral_offset(u, 6.0, CAM) < 2.6
    assert lateral_offset(float(box[0]), 6.0, CAM) < 1.3


@pytest.mark.parametrize("z", [6.0, 15.0])
def test_parked_car_with_spilled_one_sided_mask_is_not_in_lane(z):
    d = det(project_box(2.0, 3.8, z))
    lanes = one_sided_spilled_lane()
    assert lanes.contains(*d.bottom_center)  # the old rule: mask spill -> in lane
    assert not in_ego_lane(d, z, lanes, CAM, CORRIDOR)
    # also with trusted two-sided lanes whose polylines spill over the kerb
    wide = LaneResult(lanes.left, Polyline(np.array([0.0, 0.0, 799.0])), lanes.ego_mask, 0.9)
    assert wide.contains(*d.bottom_center)
    assert not in_ego_lane(d, z, wide, CAM, CORRIDOR)


def test_truncated_parked_car_at_image_bottom_is_not_in_lane():
    """KITTI frame 319 case: a parked car cut by the image bottom (and right edge), distance from
    aligned MiDaS / Kalman. Old rule: the extrapolated one-sided mask made it LEAD."""
    box = project_box(2.0, 3.8, 2.5)
    assert box[3] == HW[0] - 1 and box[2] == HW[1] - 1  # cut at the bottom and the side
    d = det(box)
    lanes = one_sided_spilled_lane()
    assert lanes.contains(*d.bottom_center)
    for z in (2.5, 0.8 * 2.5):  # also with the distance 20 % short
        assert not in_ego_lane(d, z, lanes, CAM, CORRIDOR)
        assert not in_ego_lane(d, z, None, CAM, CORRIDOR)


def two_sided_lane() -> LaneResult:
    """Lane x 300..500 at every row (the 3.5 m lane converges in reality; the polylines decide
    only through contains, the corridor uses metres)."""
    left, right = Polyline(np.array([0.0, 0.0, 300.0])), Polyline(np.array([0.0, 0.0, 500.0]))
    return LaneResult(left, right, None, 0.9)


@pytest.mark.parametrize("z", [6.0, 25.0])
@pytest.mark.parametrize("kind", ["two_sided", "one_sided", "low_conf", "none"])
def test_car_straight_ahead_is_in_lane(z, kind):
    lanes = {
        "two_sided": two_sided_lane(),
        "one_sided": one_sided_spilled_lane(),
        "low_conf": LaneResult(two_sided_lane().left, two_sided_lane().right, None, 0.1),
        "none": None,
    }[kind]
    d = det(project_box(-0.9, 0.9, z))
    assert in_ego_lane(d, z, lanes, CAM, CORRIDOR)
    # a lead car at the edge of its lane (centre 0.85 m) too, also with a 20 % distance error
    edge = det(project_box(-0.05, 1.75, z))
    assert in_ego_lane(edge, 1.2 * z, lanes, CAM, CORRIDOR)
    # end to end through DistanceFusion (geometric distance)
    frame = DistanceFusion(CFG, CAM).update([Track(1, d)], lanes, 0.0, 0, HW)
    assert frame.objects[0].in_ego_lane and frame.lead_object() is frame.objects[0]


def test_truncated_lead_at_image_bottom_is_in_lane():
    """A close car straight ahead cut by the image bottom (no geometric distance): MiDaS gives it
    a distance and it stays the lead with one-sided lanes and with none."""
    fusion = DistanceFusion(CFG, CAM)
    tracks = [track(1, 8.0, x=150), track(2, 15.0, x=600), track(3, 25.0, x=700)]
    near = Track(4, det(project_box(-0.9, 0.9, 2.5)))
    assert near.det.bbox_xyxy[3] == HW[0] - 1
    s, t = 0.004, 0.01
    inv = inv_depth_for(tracks, s, t)
    inv[330:450, 300:500] = (1.0 / 2.5 - t) / s  # lower-central part of the box
    for lanes in (one_sided_spilled_lane(), None):
        fusion.reset()
        frame = fusion.update([*tracks, near], lanes, 0.0, 0, HW, inv_depth=inv)
        lead = frame.lead_object()
        assert lead is not None and lead.track.track_id == 4
        assert lead.distance_m == pytest.approx(2.5, rel=0.02)
        assert {o.track.track_id for o in frame.objects if o.in_ego_lane} == {4}


def lane_around(u: float, cam: CameraModel, half_px: float = 40.0) -> LaneResult:
    """Trusted two-sided lane whose polylines contain column u at every row."""
    return LaneResult(
        Polyline(np.array([0.0, 0.0, u - half_px])),
        Polyline(np.array([0.0, 0.0, u + half_px])),
        None,
        0.9,
    )


def test_two_sided_lanes_curve_allowance():
    """On a curve the lane (and the lead in it) drifts sideways in the camera frame by ~Z^2/2R.
    The trusted-lane corridor grows by Z^2 / (2 curve_radius_m): a lead centred in a lane of
    radius 400 m at 30 m (drift 1.1 m) plus 0.85 m in-lane offset stays in, while a 3.5 m offset
    is vetoed at every range even when the polylines contain it."""
    z = 30.0
    drift = z * z / (2 * CORRIDOR.curve_radius_m)
    u = CAM.cx + CAM.fx * (drift + 0.85) / z
    d = det(np.array([u - 15, 200, u + 15, CAM.row_of_distance(z)], np.float32))
    assert in_ego_lane(d, z, lane_around(u, CAM), CAM, CORRIDOR)
    assert not in_ego_lane(d, z, None, CAM, CORRIDOR)  # no lanes: straight corridor says no
    for zz in (8.0, 15.0, 24.0, 30.0):
        uu = CAM.cx + CAM.fx * 3.5 / zz
        dd = det(np.array([uu - 15, 200, uu + 15, CAM.row_of_distance(zz)], np.float32))
        lanes = lane_around(uu, CAM)
        assert lanes.contains(*dd.bottom_center) and not in_ego_lane(dd, zz, lanes, CAM, CORRIDOR)


# KITTI tracking 0001 regressions (W&B 0gju6sqc, kernel v3): (frame, track, bbox, distance).
KITTI = CameraModel(
    1242, 375, 721.5377, 721.5377, 609.5593, 172.854, 1.65, 0.0, 2.0, float(np.radians(0.5))
)
KITTI_PARKED_TWO_SIDED = [  # both lanes measured (conf 0.36-0.73), YOLOP area covers the cars
    (49, 17, [477, 185, 535, 225], 24.1),
    (50, 17, [469, 189, 531, 232], 21.9),
    (51, 17, [460, 187, 526, 233], 20.2),
    (52, 17, [448, 186, 520, 235], 18.7),
    (53, 17, [436, 186, 515, 239], 17.2),
    (281, 107, [731, 179, 841, 258], 15.0),
    (282, 107, [736, 178, 853, 260], 13.7),
    (283, 107, [742, 173, 869, 261], 12.9),
    (284, 107, [747, 165, 887, 260], 12.3),
    (291, 123, [762, 194, 875, 272], 12.4),
    (292, 122, [731, 186, 816, 249], 15.9),
    (293, 122, [735, 185, 828, 251], 14.9),
    (294, 122, [741, 183, 839, 255], 14.0),
    (295, 122, [746, 183, 854, 260], 13.2),
]


@pytest.mark.parametrize("case", KITTI_PARKED_TWO_SIDED, ids=lambda c: f"f{c[0]}-t{c[1]}")
def test_kitti_parked_cars_with_two_sided_lanes_are_not_in_lane(case):
    _, _, box, z = case
    d = det(np.array(box, np.float32))
    lanes = LaneResult(  # polylines spanning the whole image: contains() alone says yes
        Polyline(np.array([0.0, 0.0, 0.0])), Polyline(np.array([0.0, 0.0, 1241.0])), None, 0.6
    )
    assert lanes.contains(*d.bottom_center)
    assert not in_ego_lane(d, z, lanes, KITTI, CORRIDOR)


@pytest.mark.parametrize(
    "box, z_filtered, z_measured",
    [([838, 191, 1150, 361], 1.2, 6.0), ([790, 187, 1044, 329], 2.7, 6.6)],
    ids=["f313-t124", "f322-t131"],
)
def test_kitti_reacquired_parked_car_with_short_coasted_distance(box, z_filtered, z_measured):
    """Frames 313/322: one-sided lanes, the coasted distance is far too short and shrinks the
    lateral offset to < 1.2 m. The lateral test uses max(filtered, measured) -> ~3.2 m: out."""
    d = det(np.array(box, np.float32))
    lanes = LaneResult(Polyline(np.array([0.0, 0.0, 300.0])), None, None, 0.58)
    assert in_ego_lane(d, z_filtered, lanes, KITTI, CORRIDOR)  # filtered distance alone: in
    assert not in_ego_lane(d, z_filtered, lanes, KITTI, CORRIDOR, z_lateral=z_measured)
    # a car straight ahead keeps its flag whatever the (longer) measured distance
    ahead = det(np.array([500, 191, 720, 361], np.float32))
    assert in_ego_lane(ahead, z_filtered, lanes, KITTI, CORRIDOR, z_lateral=z_measured)


def test_reacquired_track_restarts_instead_of_coasting():
    """KITTI 0001 track 124: measured closing at ~10 m/s (ego passing a parked car) down to 7 m,
    lost by the tracker for 5 frames, re-detected at 6 m. Coasting would output the extrapolated
    ~1 m with TTC ~0.1 s; after a gap > reacquire_restart_gap_s the filter restarts on 6 m."""
    for gap_frames, expect_restart in ((5, True), (0, False)):
        bank = FilterBank(KF)
        for k in range(10):  # 10 Hz, 16 m -> 7 m
            out = bank.update(124, 0, 0.1 * k, 16.0 - 1.0 * k, 0.2)
        t = 0.1 * (9 + gap_frames + 1)
        predicted = 7.0 - 10.0 * (t - 0.9)
        out = bank.update(124, 0, t, 6.0 if gap_frames else predicted + 4.0, 0.2)
        if expect_restart:
            assert out.restarted and not out.coasted
            assert out.distance_m == pytest.approx(6.0) and out.ttc_s is None
        else:  # one outlier on consecutive frames still coasts (closing speed survives)
            assert out.coasted and not out.restarted and out.ttc_s is not None


def test_reacquire_through_fusion_uses_the_measurement():
    """Same through DistanceFusion with geometric distances on the KITTI camera."""
    fusion = DistanceFusion(CFG, KITTI)
    hw = (KITTI.height, KITTI.width)

    def tr(z: float) -> Track:
        v = KITTI.row_of_distance(z)
        return Track(124, det(np.array([900, v - 60, 1100, v], np.float32)))

    for k in range(10):
        fusion.update([tr(16.0 - k)], None, 0.1 * k, k, hw)
    frame = fusion.update([tr(8.0)], None, 1.5, 15, hw)  # 5 frames lost, predicted ~1 m
    o = frame.objects[0]
    assert o.distance_m == pytest.approx(8.0, rel=1e-3) and o.ttc_s is None
    assert not o.in_ego_lane and frame.lead_object() is None


def test_ego_corridor_config_validation():
    bad = {**CFG["fusion"]["ego_lane"], "half_width_no_lanes_m": -1.0}
    with pytest.raises(ValueError, match="half_width_no_lanes_m"):
        EgoCorridor.from_dict(bad)
    missing = {k: v for k, v in CFG["fusion"]["ego_lane"].items() if k != "curve_radius_m"}
    with pytest.raises(KeyError):
        EgoCorridor.from_dict(missing)
    with pytest.raises(ValueError, match="curve_radius_m"):
        EgoCorridor.from_dict({**CFG["fusion"]["ego_lane"], "curve_radius_m": 0.0})


# ------------------------------------------------------------------ QA gate M4.5/M4.6 edge cases


def test_lateral_offset_inverts_a_pitched_projection():
    """Independent check of the pitch handling: project ground points through an explicit
    camera rotation (pitched down by p), then recover X from u and the forward distance Z."""
    p = np.radians(6.0)
    cam = CameraModel(800, 450, 400.0, 400.0, 400.0, 225.0, 1.5, float(p), 1.0, 0.0)
    for x_true, z in ((1.0, 5.0), (-2.5, 12.0), (3.0, 40.0)):
        # level camera frame: x right, y down, z forward; ground point at (X, h, Z)
        yc = cam.height_m * np.cos(p) - z * np.sin(p)
        zc = cam.height_m * np.sin(p) + z * np.cos(p)
        u = cam.cx + cam.fx * x_true / zc
        v = cam.cy + cam.fy * yc / zc
        assert v == pytest.approx(cam.row_of_distance(z))  # same projection as the distance model
        assert lateral_offset(u, z, cam) == pytest.approx(x_true)


def test_in_ego_lane_uses_intrinsics_scaled_to_the_frame():
    """DistanceFusion must rescale the camera to the frame size before the lateral test: on a
    half-resolution frame a centred car is still in the corridor and a car 3 m to the side is
    not (with the unscaled cx = 400 on a 400 px wide frame the centred car would read X ~ -6 m)."""
    hw = (HW[0] // 2, HW[1] // 2)
    centred = Track(1, Detection(box_at(12.0) * 0.5, 0.9, 0, "car"))
    side = Track(2, Detection(box_at(12.0, x_center=400 + 3.0 * 400 / 12.0) * 0.5, 0.9, 0, "car"))
    frame = DistanceFusion(CFG, CAM).update([centred, side], None, 0.0, 0, hw)
    by_id = {o.track.track_id: o for o in frame.objects}
    assert by_id[1].distance_m == pytest.approx(12.0, rel=0.02)
    assert by_id[1].in_ego_lane and not by_id[2].in_ego_lane


def test_reacquire_gap_boundary_at_10hz():
    """10 Hz, stationary target at 20 m, then a +10 m jump (gated). One missed frame (gap 0.2 s
    <= reacquire_restart_gap_s) coasts on the prediction; two missed frames (0.3 s) restart."""
    for missed, restart in ((1, False), (2, True)):
        bank = FilterBank(KF)
        for k in range(10):
            bank.update(1, 0, 0.1 * k, 20.0, 0.2)
        out = bank.update(1, 0, 0.1 * (10 + missed), 30.0, 0.2)
        assert out.restarted is restart and out.coasted is not restart
        assert out.distance_m == pytest.approx(30.0 if restart else 20.0, abs=0.5)
        assert out.distance_m >= 0.0


def test_ttc_gate_honoured_after_reacquire_restart():
    """After a re-acquisition restart the TTC clock starts again: no TTC until the restarted
    filter is min_track_s_for_ttc old with min_updates_for_ttc updates, then TTC comes back
    for a steady approach."""
    bank = FilterBank(KF)
    for k in range(10):  # stationary at 30 m
        bank.update(1, 0, 0.1 * k, 30.0, 0.2)
    t0, v = 1.5, 10.0  # lost for 5 frames, re-detected 10 m closer and approaching at 10 m/s
    ttcs = []
    for k in range(12):
        out = bank.update(1, 0, t0 + 0.1 * k, 20.0 - v * 0.1 * k, 0.2)
        if k == 0:
            assert out.restarted and out.ttc_s is None
        ttcs.append(out.ttc_s)
    young = int(np.ceil(KF["min_track_s_for_ttc"] / 0.1 - 1e-9))
    assert all(t is None for t in ttcs[:young])
    assert ttcs[-1] is not None
    assert ttcs[-1] == pytest.approx((20.0 - v * 1.1) / v, rel=0.25)
