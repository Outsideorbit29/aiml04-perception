"""sim/eval_frames.py (M5.6 sim eval): pure helpers on synthetic data, no models, no Kaggle."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from perception.types import Detection
from sim.eval_frames import (
    METHODS,
    depth_map_metrics,
    draw_snapshot,
    find_meta,
    fit_inverse_affine,
    gt_detections,
    implied_height,
    locate_parts,
    markdown,
    match_pairs,
    mean_rows,
    spread_subset,
    stage_part,
)

REPO = Path(__file__).resolve().parents[1]


def test_find_meta_shallow(tmp_path: Path) -> None:
    root = tmp_path / "datasets" / "owner" / "aiml04-carla-frames"
    root.mkdir(parents=True)
    (root / "meta_Town01.json").write_text("{}")
    assert find_meta(tmp_path, 4) == root / "meta_Town01.json"
    with pytest.raises(FileNotFoundError):
        find_meta(tmp_path, 2)


def test_locate_and_stage_folder_zip_and_nested(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images" / "images").mkdir(parents=True)  # folder extracted inside a folder
    (root / "images" / "images" / "Town01_00000.jpg").write_bytes(b"jpg")
    (root / "images" / "images" / "notes.md").write_text("ignored")
    (root / "labels").mkdir()
    (root / "labels" / "Town01_00000.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    with zipfile.ZipFile(root / "gt.zip", "w") as zf:  # kept as a zip by the upload
        zf.writestr("gt/Town01_00000.json", "{}")
        zf.writestr("gt/", "")
    (root / "depth").mkdir()  # empty folder -> not found
    parts = locate_parts(root)
    assert set(parts) == {"images", "labels", "gt"}
    assert parts["gt"].suffix == ".zip"
    dst = tmp_path / "stage"
    assert stage_part(parts["images"], dst / "images", (".jpg",)) == 1
    assert stage_part(parts["labels"], dst / "labels", (".txt",)) == 1
    assert stage_part(parts["gt"], dst / "gt", (".json",)) == 1
    assert (dst / "images" / "Town01_00000.jpg").read_bytes() == b"jpg"
    assert (dst / "gt" / "Town01_00000.json").read_text() == "{}"


def test_stage_zip_inside_folder(tmp_path: Path) -> None:
    src = tmp_path / "depth"
    src.mkdir()
    with zipfile.ZipFile(src / "depth.zip", "w") as zf:
        zf.writestr("Town01_00000.png", b"png")
        zf.writestr("Town01_00010.png", b"png")
    assert stage_part(src, tmp_path / "out", (".png",)) == 2


def test_spread_subset_spans_weather_blocks() -> None:
    items = list(range(3000))
    sub = spread_subset(items, 30)
    assert len(sub) == 100 and sub[0] == 0 and sub[-1] == 2970
    assert {i // 500 for i in sub} == set(range(6))  # every 500-frame weather block
    assert spread_subset(items, 0) == items


def test_match_pairs_class_agnostic_greedy() -> None:
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
    pred = np.array([[0, 0, 10, 11], [1, 0, 10, 10], [50, 50, 60, 60]], np.float32)
    pairs = match_pairs(gt, pred, np.array([0.5, 0.9, 0.8]), 0.5)
    assert [(g, p) for g, p, _ in pairs] == [(0, 1)]  # higher score wins, one GT per pred
    assert pairs[0][2] == pytest.approx(0.9)
    assert match_pairs(gt, np.zeros((0, 4)), np.zeros(0), 0.5) == []
    assert match_pairs(np.zeros((0, 4)), pred, np.ones(3), 0.5) == []


def test_gt_detections() -> None:
    objs = [{"cls_name": "car", "bbox_xyxy": [1, 2, 3, 4]}, {"cls_name": "person",
                                                             "bbox_xyxy": [5, 6, 7, 8]}]
    dets = gt_detections(objs, {"car": 0, "person": 3})
    assert [d.cls for d in dets] == [0, 3] and dets[1].cls_name == "person"
    assert dets[0].score == 1.0 and dets[0].bbox_xyxy.dtype == np.float32


def test_implied_height_inverts_flat_ground() -> None:
    fy, cy, h = 400.0, 225.0, 1.6
    z = 20.0
    v = cy + fy * h / z  # row of a ground point at 20 m for a 1.6 m camera, zero pitch
    assert implied_height(z, v, cy, fy) == pytest.approx(h)
    assert implied_height(z, cy - 1, cy, fy) is None


def test_fit_inverse_affine_recovers_scale_shift() -> None:
    m = np.linspace(1.0, 10.0, 50)
    z = 1.0 / (0.02 * m + 0.01)
    s, t = fit_inverse_affine(m, z)
    assert s == pytest.approx(0.02) and t == pytest.approx(0.01)
    assert fit_inverse_affine(np.ones(5), z[:5]) is None


def test_depth_map_metrics_perfect_and_masked() -> None:
    rng = np.random.default_rng(0)
    gt = rng.uniform(2.0, 60.0, (45, 80)).astype(np.float32)
    gt[:10] = 1000.0  # sky: outside max_m, ignored
    inv_rel = (1.0 / gt - 0.005) / 0.03  # exact affine relation to GT inverse depth
    r = depth_map_metrics(inv_rel, gt, 1.0, 80.0, stride=2)
    assert r["absrel"] < 1e-5 and r["delta1"] == 1.0 and r["coverage"] == 1.0
    assert r["s"] == pytest.approx(0.03, rel=1e-4)
    noisy = depth_map_metrics(inv_rel * rng.uniform(0.8, 1.2, gt.shape), gt, 1.0, 80.0, 2)
    assert 0.0 < noisy["absrel"] < 1.0
    with pytest.raises(ValueError):
        depth_map_metrics(inv_rel[:5], gt, 1.0, 80.0)


def test_mean_rows() -> None:
    rows = [{"absrel": 0.1, "rmse": 2.0}, {"absrel": 0.3, "rmse": None}]
    out = mean_rows(rows, ("absrel", "rmse", "delta1"))
    assert out == {"frames": 2, "absrel": pytest.approx(0.2), "rmse": 2.0, "delta1": None}


def test_draw_snapshot_shape() -> None:
    img = np.zeros((450, 800, 3), np.uint8)
    objs = [{"cls_name": "car", "bbox_xyxy": [100, 200, 200, 260], "distance_m": 12.3},
            {"cls_name": "person", "bbox_xyxy": [500, 200, 520, 260], "distance_m": 20.0}]
    dets = [Detection(np.array([101, 201, 199, 259], np.float32), 0.9, 0, "car"),
            Detection(np.array([700, 10, 750, 40], np.float32), 0.3, 0, "car")]
    vis = draw_snapshot(img, objs, dets, [(0, 0, 0.9)], [12.0, None], "t")
    assert vis.shape == img.shape and vis.any() and not img.any()


def test_markdown_renders_minimal_results() -> None:
    stats = {"n": 1, "absrel": 0.1, "rmse": 1.0, "bias": -0.05, "medrel": 0.1, "coverage": 1.0}
    dist = {"overall": {"all": {"n_gt": 1, "methods": {m: stats for m in METHODS}}},
            "per_weather": {"ClearNoon": {"all": {"methods": {m: stats for m in METHODS}}}}}
    det = {"images": 2, "all": {"map50": 0.5, "map50_95": 0.3, "precision": 0.6, "recall": 0.4},
           "per_class": [{"class": "car", "precision": 0.6, "recall": 0.4, "ap50": 0.5,
                          "ap50_95": 0.3}],
           "per_weather": {"ClearNoon": {"images": 2, "map50": 0.5, "map50_95": 0.3}}}
    md = markdown({"detection": {"finetuned": det}, "distance": {"det": dist},
                   "midas_gt": {"all": {"frames": 1, "absrel": 0.2, "rmse": 3.0, "delta1": 0.7},
                                "per_weather": {}}})
    assert "| car |" in md and "| fused | all |" in md and "| ClearNoon |" in md
    json.dumps(md)


def test_eval_config_consistent() -> None:
    cfg = yaml.safe_load((REPO / "configs/carla_eval.yaml").read_text(encoding="utf-8"))
    tracker = yaml.safe_load((REPO / "configs/tracker.yaml").read_text(encoding="utf-8"))
    assert cfg["distance"]["det_conf"] == tracker["track_activation_threshold"]
    assert cfg["camera"] == "carla" and cfg["backend"] == "torch"
    bins = cfg["distance"]["bins"]
    assert all(a[1] == b[0] for a, b in zip(bins, bins[1:], strict=False))
