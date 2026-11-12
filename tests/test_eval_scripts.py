"""Pure helpers of scripts/eval_detection.py and scripts/eval_tracking.py (no GPU, no W&B)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from perception.types import Detection
from scripts.data.bdd_yolo import format_yolo_rows, frame_to_yolo
from scripts.data.prepare_bdd import compact
from scripts.eval_detection import (
    Breakdown,
    GroundTruth,
    dets_to_arrays,
    draw_errors,
    gt_from_meta,
    gt_from_yolo,
    latency_stats,
    load_min_box_px,
    make_grid,
    match_image,
    pick_failures,
    size_buckets,
)
from scripts.eval_tracking import frame_paths, track_stats

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parents[1]


def test_match_image_same_class_greedy_by_score():
    gt_boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], np.float32)
    gt_cls = np.array([0, 3])
    pred_boxes = np.array(
        [[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60], [200, 200, 210, 210]], np.float32
    )
    pred_scores = np.array([0.6, 0.9, 0.8, 0.7])
    pred_cls = np.array([0, 0, 0, 0])  # third box has the wrong class
    gt_hit, pred_tp = match_image(gt_boxes, gt_cls, pred_boxes, pred_scores, pred_cls, 0.5)
    assert gt_hit.tolist() == [True, False]
    assert pred_tp.tolist() == [False, True, False, False]  # higher score wins the match


def test_match_image_empty_inputs():
    gt_hit, pred_tp = match_image(np.zeros((0, 4)), np.zeros(0), np.zeros((2, 4)), np.ones(2),
                                  np.zeros(2), 0.5)
    assert gt_hit.size == 0 and pred_tp.tolist() == [False, False]


def test_size_buckets():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 40, 40], [0, 0, 100, 100]], np.float32)
    assert size_buckets(boxes, 1024, 9216).tolist() == ["small", "medium", "large"]


def test_gt_from_yolo_fixture_and_meta():
    label = sorted((FIX / "labels").glob("*.txt"))[0]
    gt = gt_from_yolo(label.read_text(), 320, 180)
    assert len(gt.cls) == sum(1 for line in label.read_text().splitlines() if line.strip())
    assert (gt.boxes[:, 2] > gt.boxes[:, 0]).all() and gt.boxes.max() <= 320 + 1e-3
    assert not gt.occluded.any()
    m = gt_from_meta([[0, 1, 2, 30, 40, 1, 0], [7, 5, 5, 9, 9, 0, 1]], 320, 180, 1.0)
    assert m.cls.tolist() == [0, 7] and m.occluded.tolist() == [True, False]
    assert gt_from_meta([], 320, 180, 1.0).boxes.shape == (0, 4)


def test_breakdown_rows():
    bd = Breakdown()
    bd.add_gt("size", ["small", "small", "large"], [True, False, True])
    bd.add_pred("timeofday", "night", 4, 1)
    rows = {(r["group"], r["key"]): r for r in bd.rows()}
    assert rows[("size", "small")]["recall"] == 0.5 and rows[("size", "small")]["gt"] == 2
    assert rows[("timeofday", "night")]["precision"] == 0.25
    assert rows[("timeofday", "night")]["recall"] is None


def test_pick_failures():
    recs = [
        {"name": "a", "timeofday": "night", "n_gt": 5, "fn": 3, "fp": 1, "fn_small": 0,
         "fn_occluded": 2},
        {"name": "b", "timeofday": "daytime", "n_gt": 9, "fn": 9, "fp": 9, "fn_small": 4,
         "fn_occluded": 0},
        {"name": "c", "timeofday": "night", "n_gt": 1, "fn": 1, "fp": 0, "fn_small": 1,
         "fn_occluded": 0},
    ]
    assert [r["name"] for r in pick_failures(recs, "night", 5, min_gt=3)] == ["a"]
    assert [r["name"] for r in pick_failures(recs, "small", 5)] == ["b", "c"]
    assert [r["name"] for r in pick_failures(recs, "occlusion", 1)] == ["a"]


def test_latency_stats():
    s = latency_stats([10.0, 10.0, 20.0, 20.0])
    assert s["mean_ms"] == 15.0 and s["fps"] == pytest.approx(1000 / 15) and s["frames"] == 4
    assert latency_stats([]) == {}


def test_gallery_drawing_and_grid():
    img = np.zeros((180, 320, 3), np.uint8)
    gt = GroundTruth(np.array([0, 3]), np.array([[10, 10, 50, 50], [100, 100, 140, 160]],
                                                 np.float32), np.array([False, True]))
    pred = dets_to_arrays([Detection(np.array([10, 10, 50, 50], np.float32), 0.9, 0, "car")])
    tile = draw_errors(img, gt, np.array([True, False]), pred, np.array([True]), {0: "car"}, "t")
    assert tile.shape == img.shape and tile.any()
    grid = make_grid([tile, tile, tile], cols=2, tile_w=160)
    assert grid.shape == (180, 320, 3)  # 2x2 tiles of 160x90, last one blank


def test_track_stats_and_frame_paths(tmp_path):
    s = track_stats([[1, 2], [1, 2], [1, 3], []])
    assert s["unique_ids"] == 3 and s["max_active"] == 2 and s["short_tracks_lt3"] == 2
    assert s["mean_track_len"] == pytest.approx(2.0)  # lengths 3, 2, 1
    for name in ("000002.png", "000000.png", "000001.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in frame_paths(tmp_path, 2)] == ["000000.png", "000001.png"]
    assert len(frame_paths(tmp_path, None)) == 3


def test_eval_configs_reference_existing_files():
    for cfg_file in ("configs/yolo_eval.yaml", "configs/tracker_eval.yaml"):
        cfg = yaml.safe_load((ROOT / cfg_file).read_text())
        for key in ("detector_config", "data_config", "tracker_config"):
            if key in cfg:
                assert (ROOT / cfg[key]).exists(), (cfg_file, key)
        assert cfg["wandb"]["weights_artifact"].startswith("yolo11s-bdd-best:")


# ---------- added by qa-reviewer (M2.3-M2.5 gate) ----------
def test_gt_from_yolo_exact_pixels():
    """Normalised cx, cy, w, h -> pixel xyxy on a 1280x720 BDD frame (no xywh/xyxy mix-up)."""
    gt = gt_from_yolo("3 0.5 0.5 0.25 0.5\n\n8 0.1 0.2 0.1 0.1\n", 1280, 720)
    assert gt.cls.tolist() == [3, 8]
    np.testing.assert_allclose(gt.boxes[0], [480, 180, 800, 540])
    np.testing.assert_allclose(gt.boxes[1], [64, 108, 192, 180])
    assert gt_from_yolo("", 1280, 720).boxes.shape == (0, 4)


def test_match_image_one_prediction_per_gt_and_iou_threshold():
    gt = np.array([[0, 0, 10, 10]], np.float32)
    preds = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 4]], np.float32)
    gt_hit, tp = match_image(gt, np.array([0]), preds, np.array([0.5, 0.9, 0.95]),
                             np.array([0, 0, 0]), 0.5)
    # the 0.95 box has IoU 0.4 < 0.5; of the two exact boxes only the higher score is a TP
    assert gt_hit.tolist() == [True] and tp.tolist() == [False, True, False]


# ---------- meta GT gets the same clip + min-size filter as the YOLO labels (M2 QA item 3) ----
def test_gt_from_meta_clips_orders_and_drops_thin_boxes():
    rows = [
        [0, -20, 10, 100, 60, 1, 1],  # off the left edge: clipped to x1 = 0
        [3, 1300, 700, 1200, 600, 0, 0],  # corners swapped and past the right edge
        [7, 500, 300, 500.5, 340, 0, 0],  # 0.5 px wide: dropped
        [8, 1279.5, 10, 1300, 40, 1, 0],  # 0.5 px wide after clipping: dropped
        [1, 10, -5, 50, -1, 0, 0],  # fully above the image: dropped
        [2, 200, 200, 260, 240, 1, 0],
    ]
    gt = gt_from_meta(rows, 1280, 720, min_box_px=1.0)
    assert gt.cls.tolist() == [0, 3, 2]
    assert gt.occluded.tolist() == [True, False, True]  # filtered in step with the boxes
    np.testing.assert_allclose(
        gt.boxes, [[0, 10, 100, 60], [1200, 600, 1280, 700], [200, 200, 260, 240]]
    )
    assert gt.boxes.dtype == np.float32
    assert len(gt_from_meta(rows, 1280, 720, min_box_px=50.0).cls) == 2  # threshold is honoured


def test_gt_from_meta_matches_yolo_labels_for_the_same_frame():
    """meta/frames_val rows and labels/val/*.txt from prepare_bdd must give the same GT."""
    frame = {
        "name": "x.jpg",
        "labels": [
            {"category": "car", "box2d": {"x1": -15, "y1": 300, "x2": 120, "y2": 420},
             "attributes": {"occluded": True}},
            {"category": "pedestrian", "box2d": {"x1": 640, "y1": 350, "x2": 660, "y2": 410}},
            {"category": "traffic sign", "box2d": {"x1": 800, "y1": 100, "x2": 800.4, "y2": 130}},
            {"category": "truck", "box2d": {"x1": 1200, "y1": 500, "x2": 1350, "y2": 760}},
            {"category": "train", "box2d": {"x1": 1, "y1": 1, "x2": 50, "y2": 50}},  # unmapped
        ],
    }
    w, h, min_px = 1280, 720, load_min_box_px("configs/data_bdd.yaml")
    from_meta = gt_from_meta(compact(frame)["boxes"], w, h, min_px)
    from_yolo = gt_from_yolo(format_yolo_rows(frame_to_yolo(frame, w, h, min_px)), w, h)
    assert from_meta.cls.tolist() == from_yolo.cls.tolist() == [0, 3, 1]
    np.testing.assert_allclose(from_meta.boxes, from_yolo.boxes, atol=1e-2)
    assert from_meta.occluded.tolist() == [True, False, False]


def test_min_box_px_comes_from_the_data_prep_config():
    data = yaml.safe_load((ROOT / "configs/data_bdd.yaml").read_text())
    assert load_min_box_px("configs/data_bdd.yaml") == float(data["prep"]["min_box_px"])
    ev = yaml.safe_load((ROOT / "configs/yolo_eval.yaml").read_text())
    assert ev["data_config"] == "configs/data_bdd.yaml"
