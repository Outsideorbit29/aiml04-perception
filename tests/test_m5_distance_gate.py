"""QA gate tests for the M5.6 sim eval and the CARLA distance fix (geometric_first fusion +
detector-box calibration). Gaps not covered by test_distance_calibration.py / test_sim_eval.py:
the Kalman path receives the calibrated distance, the calibration follows the frame size in
the live call, the geometric_first fallbacks beyond max_distance_m / near the horizon, the
eval's apply_calibration matches the live geometric, the calibration study never fits on its
hold-out split, and the distance-eval GT matching is one-to-one with a hard IoU threshold.
Synthetic boxes only; CPU, no models."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from perception.fusion import DistanceFusion
from perception.types import Detection, Track
from sim.eval_frames import apply_calibration, calibration_study, match_pairs
from tests.test_distance_calibration import CAM, HW, depth_map, det_at, with_fusion

CAL = {"offset_px": -0.74, "offset_box_frac": 0.073}  # shipped configs/depth_carla.yaml values
STUDY = {"models": ["offset_box"], "fit_range_m": [5.0, 60.0], "huber_k": 1.5, "irls_iters": 10}
BINS = [[0, 10], [10, 30], [30, 60]]


def lifted_det(z: float, h: float, cal: dict = CAL, scale: float = 1.0) -> Detection:
    """Box whose bottom is lifted exactly by the calibration (so the calibrated distance is z),
    in a frame ``scale`` times the camera.yaml size."""
    cam = CAM.scaled_to((int(HW[0] * scale), int(HW[1] * scale)))
    hs = h * scale
    v = cam.row_of_distance(z) - (cal["offset_px"] * scale + cal["offset_box_frac"] * hs)
    return Detection(np.array([400 * scale - 30, v - hs, 400 * scale + 30, v], np.float32),
                     0.9, 0, "car")


def test_kalman_path_receives_calibrated_distance():
    """update(): the filter is fed the calibrated geometric distance, so ObjectState.distance_m
    (and so ACC/AEB/TTC downstream) uses the corrected value, not the raw box bottom."""
    fusion = DistanceFusion(with_fusion("geometric_first", CAL), CAM)
    raw = DistanceFusion(with_fusion("geometric_first"), CAM)
    d = lifted_det(12.0, 60.0)
    result = {}
    for name, f in (("cal", fusion), ("raw", raw)):
        out = None
        for k in range(3):
            out = f.update([Track(7, d)], None, 0.05 * k, k, HW)
        assert len(out.objects) == 1
        result[name] = out.objects[0].distance_m
    assert result["cal"] == pytest.approx(12.0, rel=1e-3)
    assert result["raw"] > 12.0 * 1.01  # uncalibrated reads long (the M5.6 finding)
    # the measurement the filter got is the calibrated one
    assert fusion.last_measurements[0].fused[0] == pytest.approx(12.0, rel=1e-6)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_calibration_follows_frame_size_in_live_call(scale):
    """offset_px is defined at camera.yaml resolution; measure() on a resized frame must give
    the same calibrated distance (row_scale wiring in DistanceFusion.geometric)."""
    fusion = DistanceFusion(with_fusion("geometric_first", CAL), CAM)
    hw = (int(HW[0] * scale), int(HW[1] * scale))
    ms, _ = fusion.measure([lifted_det(20.0, 40.0, scale=scale)], hw)
    assert ms[0].geometric[0] == pytest.approx(20.0, rel=1e-4)


def test_geometric_first_falls_back_to_midas_beyond_max_and_near_horizon():
    """Beyond geometry.max_distance_m (and above the min_angle horizon) the geometric estimate
    is invalid, so geometric_first must use the aligned MiDaS distance; near objects keep the
    geometric value even when MiDaS disagrees."""
    fusion = DistanceFusion(with_fusion("geometric_first"), CAM)
    near = [det_at(z, x=100 + 120 * i, h=40) for i, z in enumerate((8.0, 12.0, 18.0, 25.0))]
    far = det_at(95.0, x=700, h=8)  # > 80 m
    horizon = Detection(np.array([600, 215, 640, 225], np.float32), 0.9, 0, "car")  # v = cy
    dets = [*near, far, horizon]
    zs = [8.0, 12.0, 18.0, 25.0, 95.0, 300.0]
    inv = depth_map(dets, zs, s=0.01, t=0.0)
    ms, _ = fusion.measure(dets, HW, inv)
    for m, z in zip(ms[:4], zs[:4], strict=True):
        assert m.fused[0] == pytest.approx(m.geometric[0]) and m.geometric[0] == pytest.approx(z)
    assert ms[4].geometric is None and ms[4].fused is not None
    assert ms[4].fused[0] == pytest.approx(ms[4].midas[0])
    assert ms[5].geometric is None


def test_fallback_thresholds_come_from_config():
    """max_distance_m is read from config (not hard-coded): raising it makes the far box
    geometric again."""
    cfg = with_fusion("geometric_first")
    cfg["geometry"] = {**cfg["geometry"], "max_distance_m": 120.0}
    fusion = DistanceFusion(cfg, CAM)
    m, _ = fusion.measure([det_at(95.0, h=8)], HW)
    assert m[0].geometric is not None and m[0].geometric[0] == pytest.approx(95.0, rel=1e-6)


def test_eval_apply_calibration_matches_live_geometric():
    """The study's apply_calibration and the live DistanceFusion.geometric agree box by box."""
    fusion = DistanceFusion(with_fusion("geometric_first", CAL), CAM)
    raw_fusion = DistanceFusion(with_fusion("geometric_first"), CAM)
    dets = [lifted_det(z, h) for z, h in ((6.0, 140.0), (15.0, 55.0), (40.0, 20.0))]
    dets.append(Detection(np.array([300, 400, 360, 449], np.float32), 0.9, 0, "car"))  # cut
    recs = [{"y1": float(d.bbox_xyxy[1]), "y2": float(d.bbox_xyxy[3]),
             "geometric_raw": (raw_fusion.geometric(d, CAM) or (None,))[0]} for d in dets]
    got = apply_calibration(recs, {**CAL, "distance_scale": 1.0}, CAM, fusion)
    want = [(fusion.geometric(d, CAM) or (None,))[0] for d in dets]
    assert got[:3] == pytest.approx(want[:3]) and got[3] is None and want[3] is None


def _study_records(rng: np.random.Generator, n_frames: int, lift_even: float,
                   lift_odd: float) -> list[dict]:
    recs = []
    for pos in range(n_frames):
        for _ in range(3):
            z = float(rng.uniform(6, 55))
            h = float(rng.uniform(20, 120))
            lift = lift_even if pos % 2 == 0 else lift_odd
            v = CAM.row_of_distance(z) - lift
            raw = CAM.ground_distance(v)
            recs.append({"pos": pos, "gt": z, "y1": v - h, "y2": v,
                         "geometric_raw": raw[0] if raw else None})
    return recs


def test_calibration_study_never_fits_on_holdout():
    """No fit/eval leakage: on the even_odd split, changing only the odd (hold-out) records
    leaves the fitted values unchanged, and only the hold-out score moves."""
    fusion = DistanceFusion(with_fusion("geometric_first"), CAM)
    n = 40
    a = calibration_study(_study_records(np.random.default_rng(0), n, 2.0, 2.0), n, CAM,
                          fusion, STUDY, BINS)
    b = calibration_study(_study_records(np.random.default_rng(0), n, 2.0, 9.0), n, CAM,
                          fusion, STUDY, BINS)
    va, vb = (r["even_odd"]["models"]["offset_box"] for r in (a, b))
    assert va["values"] == pytest.approx(vb["values"])
    assert va["fit"]["all"]["methods"]["geometric_cal"]["absrel"] == pytest.approx(
        vb["fit"]["all"]["methods"]["geometric_cal"]["absrel"])
    hold_a = va["holdout"]["all"]["methods"]["geometric_cal"]["absrel"]
    hold_b = vb["holdout"]["all"]["methods"]["geometric_cal"]["absrel"]
    assert hold_a < 1e-3 < hold_b  # same lift: exact; different lift on hold-out: not


def test_calibration_study_does_not_mutate_records():
    fusion = DistanceFusion(with_fusion("geometric_first"), CAM)
    recs = _study_records(np.random.default_rng(1), 10, 2.0, 2.0)
    before = copy.deepcopy(recs)
    calibration_study(recs, 10, CAM, fusion, STUDY, BINS)
    assert recs == before


def test_match_pairs_one_to_one_and_iou_threshold():
    """Distance-eval matching: each GT and each prediction used at most once; IoU just below
    the threshold never matches; a lower-score duplicate falls to the next free GT only if it
    clears the threshold there."""
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], np.float32)
    dup = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [100, 100, 110, 115]], np.float32)
    pairs = match_pairs(gt, dup, np.array([0.9, 0.8, 0.7]), 0.5)
    assert sorted((g, p) for g, p, _ in pairs) == [(0, 0), (1, 2)]
    assert len({g for g, _, _ in pairs}) == len(pairs) == len({p for _, p, _ in pairs})
    # IoU of [0,0,10,10] vs [0,0,10,w]: 100 / (10 w) for w > 10
    w_below = 100 / (10 * 0.4999)
    below = np.array([[0, 0, w_below, 10]], np.float32)
    assert match_pairs(gt[:1], below, np.ones(1), 0.5) == []
    at = np.array([[0, 0, 20, 10]], np.float32)  # IoU exactly 0.5
    assert len(match_pairs(gt[:1], at, np.ones(1), 0.5)) == 1


def test_identity_calibration_is_bitwise_noop_in_measure():
    """KITTI/BDD default: an identity calibration block gives exactly the same measurements as
    no block at all, in both fusion modes."""
    dets = [det_at(z, x=100 + 150 * i, h=40) for i, z in enumerate((7.0, 16.0, 33.0, 70.0))]
    inv = depth_map(dets, [7.0, 16.0, 33.0, 70.0], s=0.02, t=0.001)
    for mode in (None, "geometric_first"):
        f0 = DistanceFusion(with_fusion(mode), CAM)
        f1 = DistanceFusion(with_fusion(mode, {"offset_px": 0.0, "offset_box_frac": 0.0,
                                               "distance_scale": 1.0}), CAM)
        m0, _ = f0.measure(dets, HW, inv)
        m1, _ = f1.measure(dets, HW, inv)
        assert [m.fused for m in m0] == [m.fused for m in m1]
        assert [m.geometric for m in m0] == [m.geometric for m in m1]

