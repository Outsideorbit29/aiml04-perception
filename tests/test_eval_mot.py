"""Pure helpers of scripts/eval_mot.py on tiny synthetic KITTI sequences (no data, no GPU)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.eval_mot import (
    FrameGT,
    Hyp,
    accumulate,
    box_ioa,
    frame_gt,
    iou_distance,
    markdown_report,
    mot_metrics,
    parse_kitti_labels,
    pick_best,
)

ROOT = Path(__file__).parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "tracker_mot.yaml").read_text())
MATCH = CFG["matching"]
CMAP = CFG["kitti_class_map"]
ALL = CFG["groups"]["all"]

LABELS = """\
0 0 Car 0 0 -1.5 100.0 100.0 200.0 180.0 1.5 1.6 3.9 1.0 1.6 20.0 -1.6
0 1 Pedestrian 0 1 0.2 400.0 90.0 430.0 170.0 1.7 0.6 0.8 3.0 1.6 15.0 0.1
0 -1 DontCare -1 -1 -10 600.0 100.0 700.0 150.0 -1000 -1000 -1000 -10 -10 -10 -1
0 2 Van 1 0 -1.5 250.0 100.0 330.0 170.0 2.0 1.8 4.5 2.0 1.6 25.0 -1.6
0 3 Tram 0 0 -1.5 800.0 50.0 1000.0 200.0 3.5 2.5 15.0 5.0 1.6 30.0 -1.6
0 4 Car 0 0 -1.5 900.0 160.0 920.0 175.0 1.5 1.6 3.9 9.0 1.6 80.0 -1.6
0 5 Car 0 3 -1.5 50.0 200.0 90.0 260.0 1.5 1.6 3.9 9.0 1.6 30.0 -1.6
2 0 Car 0 0 -1.5 110.0 100.0 210.0 180.0 1.5 1.6 3.9 1.0 1.6 19.0 -1.6
"""


def test_parse_kitti_labels():
    frames = parse_kitti_labels(LABELS + "\n")
    assert sorted(frames) == [0, 2]
    assert len(frames[0]) == 7 and len(frames[2]) == 1
    dc = frames[0][2]
    assert dc.type == "DontCare" and dc.track_id == -1 and dc.box == (600, 100, 700, 150)
    ped = frames[0][1]
    assert (ped.track_id, ped.occluded, ped.truncated) == (1, 1, 0)
    with pytest.raises(ValueError):
        parse_kitti_labels("0 1 Car 0 0")


def test_class_map_and_ignore_regions():
    g = frame_gt(parse_kitti_labels(LABELS)[0], CMAP, ALL["gt"], MATCH)
    # Car 0, Pedestrian 1, Van 2 (-> car) are evaluated.
    assert sorted(g.ids.tolist()) == [0, 1, 2]
    # DontCare, Tram (ignore), tiny car 4 (15 px < 25) and fully occluded car 5 are ignored.
    assert len(g.ignore) == 4
    assert {tuple(b) for b in g.ignore.tolist()} >= {(600, 100, 700, 150), (800, 50, 1000, 200)}
    # The vehicle group turns the pedestrian into an ignore region.
    veh = frame_gt(parse_kitti_labels(LABELS)[0], CMAP, CFG["groups"]["vehicle"]["gt"], MATCH)
    assert sorted(veh.ids.tolist()) == [0, 2] and len(veh.ignore) == 5
    # Unknown types are ignored rather than evaluated.
    assert len(frame_gt(parse_kitti_labels("0 7 Alien 0 0 0 1 1 50 90")[0], CMAP, ALL["gt"],
                        MATCH).ignore) == 1


def test_config_maps_every_kitti_type():
    kitti = {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc",
             "DontCare"}
    assert set(CMAP) == kitti and CMAP["DontCare"] == "ignore"
    bdd = set(yaml.safe_load((ROOT / "configs" / "data_bdd.yaml").read_text())["names"].values())
    for spec in CFG["groups"].values():
        assert set(spec["pred"]) <= bdd and set(spec["gt"]) <= set(CMAP.values())
    assert CFG["sweep"][0]["name"] == "default" and CFG["sweep"][0]["params"] == {}


def test_box_ioa_and_distance():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[5, 0, 20, 10], [0, 0, 10, 10]], np.float32)
    assert box_ioa(a, b)[0].tolist() == pytest.approx([0.5, 1.0])
    d = iou_distance(a, b, 0.5)
    assert np.isnan(d[0, 0]) and d[0, 1] == pytest.approx(0.0)


# ---------- MOT metrics (motmetrics + scipy, which Kaggle has) ----------
def _seq(n: int = 6) -> tuple[list[int], dict[int, FrameGT]]:
    """Two objects moving right, far apart, visible in all n frames, plus one DontCare."""
    gt = {}
    for f in range(n):
        boxes = np.array([[10 + 5 * f, 50, 60 + 5 * f, 100], [300 + 5 * f, 50, 350 + 5 * f, 100]],
                         np.float32)
        gt[f] = FrameGT(np.array([7, 9]), boxes, np.array([[500, 0, 600, 100]], np.float32))
    return list(range(n)), gt


def _hyps(gt: dict[int, FrameGT], ids_at) -> dict[int, Hyp]:
    return {f: Hyp(np.asarray(ids_at(f)), g.boxes.copy(), ["car", "person"])
            for f, g in gt.items()}


def test_perfect_tracker_scores_one():
    pytest.importorskip("motmetrics")
    pytest.importorskip("scipy")
    frames, gt = _seq()
    hyps = _hyps(gt, lambda f: [1, 2])
    # An extra hypothesis inside the DontCare region must not become a false positive.
    hyps[3] = Hyp(np.array([1, 2, 3]), np.vstack([gt[3].boxes, [[510, 10, 590, 90]]]),
                  ["car", "person", "car"])
    m = mot_metrics([accumulate(frames, gt, hyps, ALL["pred"], MATCH)], ["s"])
    for key in ("s", "overall"):
        assert m[key]["mota"] == pytest.approx(1.0) and m[key]["idf1"] == pytest.approx(1.0)
        assert m[key]["idsw"] == 0 and m[key]["fp"] == 0 and m[key]["fn"] == 0
        assert m[key]["mt"] == 2 and m[key]["motp"] == pytest.approx(1.0)


def test_id_swap_counts_one_switch():
    pytest.importorskip("motmetrics")
    pytest.importorskip("scipy")
    frames, gt = _seq()
    # From frame 3 on, the tracker swaps the two ids.
    hyps = _hyps(gt, lambda f: [1, 2] if f < 3 else [2, 1])
    m = mot_metrics([accumulate(frames, gt, hyps, ALL["pred"], MATCH)], ["s"])["overall"]
    assert m["idsw"] == 2  # both objects switch once
    assert m["fp"] == 0 and m["fn"] == 0 and m["idf1"] < 1.0
    # A single object whose id changes once: exactly one switch.
    one = {f: FrameGT(g.ids[:1], g.boxes[:1], g.ignore) for f, g in gt.items()}
    hyps1 = {f: Hyp(np.array([1 if f < 3 else 5]), g.boxes[:1], ["car"]) for f, g in gt.items()}
    m1 = mot_metrics([accumulate(frames, one, hyps1, ALL["pred"], MATCH)], ["s"])["overall"]
    assert m1["idsw"] == 1
    assert m1["mota"] == pytest.approx(1 - 1 / 6)


def test_unused_classes_and_tiny_boxes_are_dropped():
    pytest.importorskip("motmetrics")
    pytest.importorskip("scipy")
    frames, gt = _seq(2)
    hyps = {f: Hyp(np.array([1, 2, 3, 4]),
                   np.vstack([g.boxes, [[700, 50, 760, 100]], [[800, 50, 810, 60]]]),
                   ["car", "person", "traffic sign", "car"]) for f, g in gt.items()}
    m = mot_metrics([accumulate(frames, gt, hyps, ALL["pred"], MATCH)], ["s"])["overall"]
    assert m["fp"] == 0 and m["mota"] == pytest.approx(1.0)


def test_pick_best_and_markdown():
    def row(name, mota, idf1):
        o = {"mota": mota, "idf1": idf1, "idsw": 3, "mt": 1, "ml": 0, "fp": 2, "fn": 4,
             "frag": 1, "precision": 0.9, "recall": 0.8}
        return {"name": name, "params": {"lost_track_buffer": 60} if name != "a" else {},
                "metrics": {"all": {"0000": o, "overall": o}, "vehicle": {"overall": o}},
                "unique_ids": 10, "short_tracks_lt3": 2, "track_ms": 3.0, "fps": 50.0}

    rows = [row("a", 0.5, 0.6), row("b", 0.4, 0.7), row("c", 0.9, None)]
    assert pick_best(rows, ["idf1", "mota"])["name"] == "b"
    result = {"sequences": ["0000"], "frames": 10, "matching": MATCH, "det_ms": 15.0,
              "device": "T4", "sweep": rows, "best": "b", "select_by": ["idf1", "mota"]}
    md = markdown_report(result)
    assert "**b**" in md and "| 0000 |" in md and "lost_track_buffer=60" in md
    table = [ln for ln in md.splitlines() if ln.startswith("| a |")][0]
    assert table.count("|") == md.splitlines()[md.splitlines().index(table) - 1].count("|")
