"""Fusion mode (geometric_first vs inverse_depth_weighted) and the detector-box calibration of
the geometric distance (M5.6 decisions), on synthetic boxes. Pure logic: no models, no Kaggle."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from perception.depth import BottomCalibration, CameraModel, fit_distance_scale, fit_row_offset
from perception.depth.calibration import box_geometric_distance
from perception.depth.estimator import load_depth_config
from perception.fusion import DistanceFusion, fuse_by_mode, fuse_inverse, inverse
from perception.pipeline import load_pipeline_config, stage_configs
from perception.types import Detection
from sim.eval_frames import (
    apply_calibration,
    calibration_study,
    fit_calibration,
    select_split,
    split_masks,
)

REPO = Path(__file__).resolve().parents[1]
CAM =CameraModel(800, 450, 400.0, 400.0, 400.0, 225.0, 1.6, 0.0, 1.5, math.radians(0.3))
HW = (450, 800)
BINS = [[0, 10], [10, 30], [30, 60]]
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
        "correlation": 0.8,
        "ego_lane": {
            "min_lane_confidence": 0.3,
            "half_width_lanes_m": 1.5,
            "curve_radius_m": 400.0,
            "half_width_no_lanes_m": 1.2,
            "range_no_lanes_m": 30.0,
        },
    },
    "kalman": {
        "accel_std": 3.0,
        "init_velocity_std": 10.0,
        "gate_chi2": 16.0,
        "max_gap_s": 1.0,
        "max_age_s": 2.0,
        "min_updates_for_ttc": 3,
        "min_track_s_for_ttc": 0.75,
        "min_closing_speed_mps": 0.5,
        "ttc_speed_sigma_k": 2.0,
        "gate_restart_count": 2,
        "restart_on_class_change": False,
        "reacquire_restart_gap_s": 0.275,
    },
}


def with_fusion(mode: str | None = None, calibration: dict | None = None) -> dict:
    cfg = {**CFG, "fusion": dict(CFG["fusion"]), "geometry": dict(CFG["geometry"])}
    if mode is not None:
        cfg["fusion"]["mode"] = mode
    if calibration is not None:
        cfg["geometry"]["calibration"] = calibration
    return cfg


def det_at(z: float, x: float = 400.0, h: float = 50.0, lift_px: float = 0.0) -> Detection:
    """Car box whose true contact row is at distance z; its bottom is ``lift_px`` higher."""
    v = CAM.row_of_distance(z) - lift_px
    return Detection(np.array([x - 30, v - h, x + 30, v], np.float32), 0.9, 0, "car")


def depth_map(dets: list[Detection], zs: list[float], s: float, t: float) -> np.ndarray:
    """Synthetic MiDaS map with 1/Z = s m + t inside each box (far boxes painted first)."""
    inv = np.zeros(HW, np.float32)
    for d, z in sorted(zip(dets, zs, strict=True), key=lambda p: -p[1]):
        x1, y1, x2, y2 = d.bbox_xyxy.astype(int)
        inv[y1 : y2 + 1, x1 : x2 + 1] = (1.0 / z - t) / s
    return inv


# ------------------------------------------------------------------ fuse_by_mode


def test_fuse_by_mode_geometric_first_and_fallback():
    geo, midas = (1 / 20.0, 0.004), (1 / 25.0, 0.002)  # MiDaS has the smaller std
    z, s = fuse_by_mode(geo, midas, "geometric_first", 0.8)
    assert z == pytest.approx(20.0) and s == pytest.approx(400 * 0.004)
    z, _ = fuse_by_mode(None, midas, "geometric_first", 0.8)
    assert z == pytest.approx(25.0)  # no valid geometry -> MiDaS
    assert fuse_by_mode(None, None, "geometric_first") is None
    weighted = fuse_by_mode(geo, midas, "inverse_depth_weighted", 0.8)
    assert weighted == fuse_inverse([geo, midas], 0.8)
    assert 20.0 < weighted[0] < 25.0
    with pytest.raises(ValueError):
        fuse_by_mode(geo, midas, "average")


def test_inverse_roundtrip():
    assert inverse((20.0, 2.0)) == pytest.approx((0.05, 0.005))
    assert inverse(None) is None and inverse((0.0, 1.0)) is None


def test_default_mode_is_unchanged_weighted_fusion():
    """No ``fusion.mode`` key and the explicit default give identical fused distances (the M4
    KITTI path), and they differ from geometric_first when MiDaS disagrees."""
    zs = [8.0, 15.0, 25.0, 12.0]
    dets = [det_at(z, x) for z, x in zip(zs, (100, 300, 500, 700), strict=True)]
    inv = depth_map(dets, [8.0, 15.0, 25.0, 14.0], 0.004, 0.01)  # MiDaS off on the 4th box
    legacy, _ = DistanceFusion(with_fusion(), CAM).measure(dets, HW, inv)
    explicit, _ = DistanceFusion(with_fusion("inverse_depth_weighted"), CAM).measure(dets, HW, inv)
    geofirst, _ = DistanceFusion(with_fusion("geometric_first"), CAM).measure(dets, HW, inv)
    for a, b in zip(legacy, explicit, strict=True):
        assert a.fused == b.fused
    assert legacy[3].fused[0] > 12.0 + 0.05  # pulled towards MiDaS's 14 m
    assert geofirst[3].fused[0] == pytest.approx(12.0, rel=1e-6)
    for m in geofirst:
        assert m.fused == m.geometric


def test_geometric_first_uses_midas_for_truncated_box():
    zs = [8.0, 15.0, 25.0]
    dets = [det_at(z, x) for z, x in zip(zs, (150, 400, 650), strict=True)]
    near = Detection(np.array([300, 330, 500, 449], np.float32), 0.9, 0, "car")  # cut by bottom
    inv = depth_map(dets, zs, 0.004, 0.01)
    inv[400:450, 350:450] = (1.0 / 4.0 - 0.01) / 0.004
    ms, align = DistanceFusion(with_fusion("geometric_first"), CAM).measure(
        [*dets, near], HW, inv
    )
    assert align is not None and ms[3].geometric is None
    assert ms[3].fused[0] == pytest.approx(4.0, rel=0.02)
    # without a depth map (MiDaS skipped frame) the truncated box has no estimate at all
    ms, _ = DistanceFusion(with_fusion("geometric_first"), CAM).measure([*dets, near], HW)
    assert ms[3].fused is None and ms[0].fused == ms[0].geometric


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        DistanceFusion(with_fusion("median"), CAM)


# ------------------------------------------------------------------ shipped configs


def test_shipped_configs_keep_kitti_default_and_switch_carla():
    base = load_depth_config("configs/depth.yaml")
    assert base["fusion"]["mode"] == "inverse_depth_weighted"
    assert BottomCalibration.from_dict(base["geometry"]["calibration"]).identity
    kitti = stage_configs(load_pipeline_config("configs/pipeline.yaml"), "torch", "kitti")
    assert kitti["depth"]["fusion"]["mode"] == "inverse_depth_weighted"
    assert BottomCalibration.from_dict(kitti["depth"]["geometry"].get("calibration")).identity
    carla = stage_configs(load_pipeline_config("configs/pipeline_carla.yaml"), "torch", "carla")
    assert carla["depth"]["fusion"]["mode"] == "geometric_first"
    assert carla["depth"]["estimator"]["every_n_frames"] == 2  # MiDaS still runs for coverage
    cal = BottomCalibration.from_dict(carla["depth"]["geometry"]["calibration"])
    assert 0.8 <= cal.distance_scale <= 1.2 and abs(cal.offset_px) < 20


# ------------------------------------------------------------------ BottomCalibration


def test_identity_calibration_matches_raw_geometry():
    cal = BottomCalibration.from_dict(None)
    assert cal.identity
    for z in (5.0, 12.0, 40.0):
        v = CAM.row_of_distance(z)
        assert box_geometric_distance(CAM, v - 40, v, cal) == CAM.ground_distance(v)


def test_calibration_corrects_lifted_box_bottom():
    """Box bottoms lifted by a + b*h read long; the matching offset restores the distance."""
    a, b = 1.5, 0.04
    cal = BottomCalibration(offset_px=a, offset_box_frac=b)
    for z, h in ((6.0, 120.0), (15.0, 50.0), (35.0, 22.0)):
        v_true = CAM.row_of_distance(z)
        v = v_true - (a + b * h)
        raw = box_geometric_distance(CAM, v - h, v)
        cor = box_geometric_distance(CAM, v - h, v, cal)
        assert raw[0] > z * 1.02
        assert cor[0] == pytest.approx(z, rel=1e-6)
        assert cor[1] == pytest.approx(CAM.distance_std(z))


def test_calibration_scale_and_row_scale():
    cal = BottomCalibration(distance_scale=0.9)
    v = CAM.row_of_distance(20.0)
    assert box_geometric_distance(CAM, v - 40, v, cal)[0] == pytest.approx(18.0)
    # offset is defined at the configured size: a frame twice as tall doubles it
    off = BottomCalibration(offset_px=2.0)
    assert off.corrected_row(100.0, 300.0, row_scale=2.0) == pytest.approx(304.0)


def test_calibration_keeps_raw_truncation_and_range_rules():
    cal = BottomCalibration(offset_px=-5.0)  # would move a cut box above the margin
    assert box_geometric_distance(CAM, 400, 449, cal, bottom_margin_px=2) is None
    far = CAM.row_of_distance(70.0)
    assert box_geometric_distance(CAM, far - 10, far, BottomCalibration(distance_scale=1.2),
                                  max_distance_m=80.0) is None


def test_calibration_config_validation():
    with pytest.raises(KeyError):
        BottomCalibration.from_dict({"offset": 1.0})
    with pytest.raises(ValueError):
        BottomCalibration.from_dict({"distance_scale": 0.0})
    with pytest.raises(ValueError):
        BottomCalibration.from_dict({"offset_px": float("nan")})


def test_fusion_applies_calibration_to_detections_only_when_asked():
    a = 2.0
    fusion = DistanceFusion(with_fusion("geometric_first", {"offset_px": a}), CAM)
    d = det_at(15.0, lift_px=a)
    cam = CAM.scaled_to(HW)
    assert fusion.geometric(d, cam)[0] == pytest.approx(15.0, rel=1e-6)
    assert fusion.geometric(d, cam, calibrated=False)[0] > 15.5
    ms, _ = fusion.measure([d], HW)
    assert ms[0].fused[0] == pytest.approx(15.0, rel=1e-6)


# ------------------------------------------------------------------ fits


def test_fit_distance_scale_recovers_ratio_robustly():
    rng = np.random.default_rng(0)
    gt = rng.uniform(5, 60, 400)
    pred = gt * 1.1 * np.exp(rng.normal(0, 0.03, gt.size))
    pred[:20] *= 3.0  # outliers
    assert fit_distance_scale(pred, gt) == pytest.approx(1 / 1.1, rel=0.01)
    with pytest.raises(ValueError):
        fit_distance_scale([np.nan], [10.0])


def test_fit_row_offset_recovers_affine_lift():
    rng = np.random.default_rng(1)
    z = rng.uniform(5, 60, 500)
    h = 900.0 / z  # box height shrinks with distance
    v_true = np.array([CAM.row_of_distance(x) for x in z])
    v = v_true - (1.0 + 0.05 * h) + rng.normal(0, 0.3, z.size)
    v[:25] -= 15.0  # outliers (bad boxes)
    a, b = fit_row_offset(v, h, v_true, with_box_h=True)
    assert a == pytest.approx(1.0, abs=0.15) and b == pytest.approx(0.05, abs=0.005)
    a0, b0 = fit_row_offset(v, h, v_true, with_box_h=False)
    assert b0 == 0.0 and a0 > 1.0  # constant model absorbs the mean of the h term


# ------------------------------------------------------------------ sim eval study


def synthetic_records(n_frames: int, a: float, b: float, seed: int = 2) -> list[dict]:
    rng = np.random.default_rng(seed)
    recs = []
    for pos in range(n_frames):
        z = float(rng.uniform(3, 60))
        h = 900.0 / z
        v = CAM.row_of_distance(z) - (a + b * h) + float(rng.normal(0, 0.3))
        raw = box_geometric_distance(CAM, v - h, v, None, math.radians(0.5), 80.0, 2.0)
        recs.append({"pos": pos, "gt": z, "y1": v - h, "y2": v,
                     "geometric_raw": raw[0] if raw else None})
    return recs


def test_split_masks_are_disjoint_and_complete():
    pos = np.arange(10)
    for fit, hold in split_masks(pos, 10).values():
        assert not (fit & hold).any() and (fit | hold).all()
    fit, hold = split_masks(pos, 10)["halves_ab"]
    assert fit[:5].all() and hold[5:].all()


def test_calibration_study_improves_holdout_on_lifted_boxes():
    recs = synthetic_records(600, a=1.0, b=0.05)
    fusion = DistanceFusion(with_fusion(), CAM)
    ccfg = {"models": ["scale", "offset", "offset_box"], "fit_range_m": [5.0, 60.0],
            "huber_k": 1.5, "irls_iters": 10}
    study = calibration_study(recs, 600, CAM, fusion, ccfg, BINS)
    assert set(study) == {"even_odd", "halves_ab", "halves_ba"}
    for split in study.values():
        assert split["n_fit"] + split["n_holdout"] == 600
        for model, e in split["models"].items():
            hold = e["holdout"]["all"]["methods"]
            assert hold["geometric_raw"]["bias"] > 0.02  # lifted boxes read long
            assert hold["geometric_cal"]["absrel"] < hold["geometric_raw"]["absrel"]
            if model == "offset_box":  # the true model: bias gone
                assert abs(hold["geometric_cal"]["bias"]) < 0.01
    vals = study["even_odd"]["models"]["offset_box"]["values"]
    assert vals["offset_px"] == pytest.approx(1.0, abs=0.2)
    assert vals["offset_box_frac"] == pytest.approx(0.05, abs=0.01)
    cz = apply_calibration(recs[:5], vals, CAM, fusion)
    assert len(cz) == 5
    with pytest.raises(ValueError):
        fit_calibration("poly", recs, CAM, [5, 60], 1.5, 10)


# ------------------------------------------------------------------ M5.9 val frames


def test_select_split_keeps_only_the_mix_val_blocks():
    weathers = ["ClearNoon", "CloudyNoon", "WetNoon", "ClearSunset", "ClearNight", "HardRainNoon"]
    meta = {"weathers": [{"weather": w, "first_index": 500 * i} for i, w in enumerate(weathers)]}
    paths = [Path(f"Town01_{i:05d}.jpg") for i in range(3000)]
    scfg = yaml.safe_load((REPO / "configs/yolo_mix.yaml").read_text(encoding="utf-8"))["split"]
    val = select_split(paths, meta, scfg, "val")
    idx = [int(p.stem.split("_")[1]) for p in val]
    assert idx == [i for b in range(6) for i in range(400 + 500 * b, 500 + 500 * b)]
    train = {int(p.stem.split("_")[1]) for p in select_split(paths, meta, scfg, "train")}
    guard = set(range(380, 400)) | set(range(500, 520))
    assert not train & set(idx) and not train & guard
    # positions over the val list: the halves are val blocks 1-3 vs 4-6
    fit, _ = split_masks(np.arange(len(val)), len(val))["halves_ab"]
    first3 = {i for b in range(3) for i in range(400 + 500 * b, 500 + 500 * b)}
    assert {idx[i] for i in np.flatnonzero(fit)} == first3


def test_calibration_study_scores_current_values_without_refit():
    recs = synthetic_records(300, a=1.0, b=0.05)
    fusion = DistanceFusion(with_fusion(), CAM)
    ccfg = {"models": ["offset_box"], "fit_range_m": [5.0, 60.0], "huber_k": 1.5,
            "irls_iters": 10}
    cur = {"offset_px": 1.0, "offset_box_frac": 0.05, "distance_scale": 1.0}
    study = calibration_study(recs, 300, CAM, fusion, ccfg, BINS, current=cur)
    for split in study.values():
        e = split["models"]["current"]
        assert e["values"] == cur
        hold = e["holdout"]["all"]["methods"]
        assert abs(hold["geometric_cal"]["bias"]) < 0.01 < hold["geometric_raw"]["bias"]
