"""Kalman filter bank and DistanceFusion on synthetic tracks (M4.3, M4.5).

Parameters are pinned here (not read from configs/depth.yaml) so retuning the shipped config
cannot change what these tests check.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.depth import CameraModel
from perception.fusion import DistanceFusion, FilterBank, fuse, fuse_inverse
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


def test_in_ego_lane_without_lanes_is_false():
    frame = DistanceFusion(CFG, CAM).update([track(1, 12.0)], None, 0.0, 0, HW)
    assert frame.objects and not frame.objects[0].in_ego_lane
    assert frame.lead_object() is None


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
    bank = FilterBank(KF)
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
