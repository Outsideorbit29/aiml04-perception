"""KITTI helpers and metric code of scripts/eval_depth.py on synthetic data (M4.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from perception.depth import CameraModel
from scripts.eval_depth import (
    KittiObject,
    bin_label,
    draw_gt_overlay,
    draw_pred_overlay,
    error_stats,
    gt_lidar_distance,
    in_box3d,
    lidar_in_image,
    markdown_table,
    parse_calib,
    parse_labels,
    project_rect,
    summarize,
    velo_to_rect,
)

# velodyne (x fwd, y left, z up) -> camera (x right, y down, z fwd), offset like KITTI
TR = np.array([[0, -1, 0, 0.0], [0, 0, -1, -0.08], [1, 0, 0, -0.27]], float)
P2 = np.array([[700.0, 0, 600.0, 45.0], [0, 700.0, 170.0, 0.3], [0, 0, 1.0, 0.005]])
CALIB_TEXT = "\n".join(
    [
        "P0: " + " ".join(["0"] * 12),
        "P2: " + " ".join(f"{v:g}" for v in P2.ravel()),
        "R0_rect: 1 0 0 0 1 0 0 0 1",
        "Tr_velo_to_cam: " + " ".join(f"{v:g}" for v in TR.ravel()),
        "Tr_imu_to_velo: " + " ".join(["0"] * 12),
        "",
    ]
)
HW = (375, 1242)


def test_parse_calib_pads_to_4x4():
    c = parse_calib(CALIB_TEXT)
    assert c["P2"].shape == (3, 4) and c["R0_rect"].shape == (4, 4)
    assert c["Tr_velo_to_cam"].shape == (4, 4)
    np.testing.assert_allclose(c["Tr_velo_to_cam"][3], [0, 0, 0, 1])
    np.testing.assert_allclose(c["R0_rect"], np.eye(4))
    with pytest.raises(KeyError):
        parse_calib("P2: 1 2 3\n")


def test_projection_of_known_point():
    c = parse_calib(CALIB_TEXT)
    p_velo = np.array([[20.0, 2.0, -1.0]])  # 20 m ahead, 2 m left, 1 m down
    rect = velo_to_rect(p_velo, c)
    np.testing.assert_allclose(rect, [[-2.0, 0.92, 19.73]], atol=1e-9)
    uv = project_rect(rect, c["P2"])
    w = 19.73 + 0.005
    expected = [(700 * -2.0 + 600 * 19.73 + 45) / w, (700 * 0.92 + 170 * 19.73 + 0.3) / w]
    np.testing.assert_allclose(uv[0], expected, rtol=1e-12)


def test_lidar_in_image_drops_points_behind_and_outside():
    c = parse_calib(CALIB_TEXT)
    pts = np.array(
        [
            [20.0, 2.0, -1.0],  # in front, in image
            [-20.0, 2.0, -1.0],  # behind: would project into the image if not dropped first
            [0.5, 0.0, 0.0],  # closer than min_depth
            [5.0, -50.0, 0.0],  # far to the right, outside the image
        ]
    )
    uv, rect = lidar_in_image(pts, c, HW, min_depth=1.0)
    assert len(uv) == 1 and rect[0, 2] == pytest.approx(19.73)


def test_parse_labels_skips_dontcare():
    text = (
        "Car 0.00 0 -1.58 587.0 173.3 614.1 200.1 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59\n"
        "DontCare -1 -1 -10 503.9 169.7 590.6 190.1 -1 -1 -1 -1000 -1000 -1000 -10\n"
        "Pedestrian 0.20 1 0.2 700 150 730 230 1.8 0.6 0.8 3.0 1.65 12.0 0.0\n"
    )
    objs = parse_labels(text)
    assert [o.type for o in objs] == ["Car", "Pedestrian"]
    car = objs[0]
    assert car.location[2] == pytest.approx(46.70) and car.dims_hwl[0] == pytest.approx(1.65)
    assert objs[1].occluded == 1 and objs[1].truncated == pytest.approx(0.2)


def car_obj(ry: float = 0.0) -> KittiObject:
    # 4 m long (x), 2 m wide (z), 1.5 m high; bottom centre 2 m right, 1.65 m down, 20 m ahead
    return KittiObject(
        "Car",
        0.0,
        0,
        np.array([0.0, 0.0, 1242.0, 375.0]),
        np.array([1.5, 2.0, 4.0]),
        np.array([2.0, 1.65, 20.0]),
        ry,
    )


def test_in_box3d_rotation_and_ground_clearance():
    pts = np.array(
        [
            [2.0, 1.0, 20.0],  # centre, mid height
            [3.9, 1.0, 20.0],  # along the length (x) inside for ry = 0
            [2.0, 1.0, 21.9],  # along z beyond the half width 1.0 (+0.2 margin)
            [2.0, 1.62, 20.0],  # ground hit just above the bottom (y down)
            [2.0, -0.3, 20.0],  # above the roof (top at y = 0.15, margin 0.2 -> -0.05)
        ]
    )
    m0 = in_box3d(pts, car_obj(0.0), margin=0.2, clearance=0.1)
    assert m0.tolist() == [True, True, False, False, False]
    # rotated by 90 deg the length runs along z
    m90 = in_box3d(pts, car_obj(math.pi / 2), margin=0.2, clearance=0.1)
    assert m90.tolist() == [True, False, True, False, False]


def test_gt_lidar_distance_uses_points_on_the_object():
    obj = car_obj(-math.pi / 2)  # seen from behind: the length runs along camera z
    obj.bbox = np.array([500.0, 100.0, 700.0, 250.0])
    rng = np.random.default_rng(0)
    surf = np.column_stack(
        [rng.uniform(1.2, 2.8, 30), rng.uniform(0.3, 1.4, 30), np.full(30, 18.1)]
    )  # rear surface of the car (box spans z 18-22)
    bg = np.column_stack([rng.uniform(1.2, 2.8, 30), rng.uniform(0.3, 1.4, 30), np.full(30, 40)])
    pts = np.vstack([surf, bg])
    uv = np.tile([600.0, 200.0], (len(pts), 1))  # all inside the 2D box
    gcfg = {"min_points": 5, "percentile": 50, "box3d_margin_m": 0.2, "ground_clearance_m": 0.1}
    d, n = gt_lidar_distance(uv, pts, obj, gcfg)
    assert n == 30 and d == pytest.approx(18.1)
    d, n = gt_lidar_distance(uv[30:], pts[30:], obj, gcfg)
    assert n == 0 and math.isnan(d)


def test_bins_and_error_stats():
    bins = [[0, 10], [10, 30], [30, 60]]
    assert bin_label(5, bins) == "0-10" and bin_label(10, bins) == "10-30"
    assert bin_label(60, bins) == "30-60" and bin_label(61, bins) is None
    st = error_stats(np.array([11.0, 9.0]), np.array([10.0, 10.0]))
    assert st["absrel"] == pytest.approx(0.1) and st["rmse"] == pytest.approx(1.0)
    assert st["bias"] == pytest.approx(0.0)
    assert error_stats(np.array([]), np.array([]))["absrel"] is None


def test_summarize_coverage_and_common_subset():
    recs = [
        {"gt": 5.0, "a": 5.5, "b": None},
        {"gt": 20.0, "a": 22.0, "b": 19.0},
        {"gt": 25.0, "a": None, "b": 26.0},
        {"gt": 80.0, "a": 80.0, "b": 80.0},  # beyond the last bin
        {"gt": None, "a": 1.0, "b": 1.0},  # no GT
    ]
    s = summarize(recs, ["a", "b"], "gt", [[0, 10], [10, 30], [30, 60]])
    assert s["all"]["n_gt"] == 3 and s["10-30"]["n_gt"] == 2
    assert s["10-30"]["methods"]["a"]["n"] == 1 and s["10-30"]["methods"]["a"]["coverage"] == 0.5
    assert s["all"]["n_common"] == 1
    assert s["all"]["common"]["b"]["absrel"] == pytest.approx(0.05)
    md = markdown_table(s, ["a", "b"], "methods")
    assert md.count("\n") == 1 + 2 * 4
    # extra rows (e.g. MiDaS split by fit mode) do not shrink the common subset
    recs[1]["a@scale"] = 22.0
    s2 = summarize(recs, ["a", "b", "a@scale"], "gt", [[0, 10], [10, 30], [30, 60]], ["a", "b"])
    assert s2["all"]["n_common"] == 1 and set(s2["all"]["common"]) == {"a", "b"}
    assert s2["all"]["methods"]["a@scale"]["n"] == 1


def test_overlays_render():
    img = np.zeros((375, 1242, 3), np.uint8)
    rec = {
        "bbox": [500.0, 150.0, 600.0, 220.0],
        "gt_lidar": 18.0,
        "gt_label_z": 20.0,
        "n_lidar": 30,
        "anchor": True,
        "geometric": 17.5,
        "midas_small": None,
    }
    frame = {"records": [rec, {**rec, "gt_lidar": None, "n_lidar": 2}], "skipped": [[1, 2, 30, 40]]}
    dcfg = {
        "max_depth_m": 60.0,
        "point_radius": 1,
        "box_thickness": 2,
        "font_scale": 0.45,
        "text_thickness": 1,
        "inset_height": 50,
        "horizon_color": [0, 255, 255],
        "gt_color": [0, 255, 0],
        "nolidar_color": [0, 0, 255],
        "skipped_color": [128, 128, 128],
    }
    cam = CameraModel(1242, 375, 700.0, 700.0, 600.0, 170.0, 1.65, 0.0)
    uv = np.array([[550.0, 200.0], [10.0, 10.0]])
    gt = draw_gt_overlay(img, uv, np.array([18.0, 50.0]), frame, cam, dcfg)
    assert gt.shape == img.shape and gt.any()
    maps = {"midas_small": np.random.default_rng(0).random((375, 1242)).astype(np.float32)}
    pred = draw_pred_overlay(gt, frame, ["geometric", "midas_small"], maps, dcfg)
    assert pred.shape == (375 + 50, 1242, 3)
