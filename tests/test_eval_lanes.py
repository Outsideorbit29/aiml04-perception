"""M3.3 lane-eval GT handling (scripts/eval_lanes.py) on tiny synthetic BDD-2018-style label dicts.

Covers poly2d parsing (x, y order, Bezier, dict-or-list), drivable-area rasterisation, label to
image rescaling, the "GT empty" vs "no prediction" accounting, single-side fallback scoring
(F1 measured-only, IoU with and without the fallback), short-GT / no_gt accounting, the
band/warp consistency check and the BDD camera config. CPU only, nothing is downloaded.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from perception.lanes import build_lane_detector, load_lane_config
from perception.lanes.threshold import frac_to_px
from perception.types import LaneResult, Polyline
from scripts.eval_lanes import (
    aggregate,
    band_slice,
    check_band_matches_warp,
    debug_tiles,
    direct_area_mask,
    ego_boundaries,
    evaluate_image,
    flatten_poly2d,
    gt_geometry,
    label_scale,
    lane_polylines,
    markdown_table,
    spread_indices,
)

ROOT = Path(__file__).resolve().parents[1]
LH, LW = 720, 1280  # BDD label pixel frame
MCFG = {
    "band_frac": [0.58, 0.74],
    "band_rows": 30,
    "lane_tol_px": 20,
    "min_gt_rows": 5,
    "gt_ref_frac": 0.74,
    "gt_min_bottom_frac": 0.55,
    "bezier_steps": 8,
    "exclude_lane_types": ["crosswalk"],
    "curve_px": 15,
}

# Ego lane: straight lines through a vanishing point at (640, 340), in 1280x720 label pixels.
VP = (640.0, 340.0)
LEFT_BOTTOM, RIGHT_BOTTOM = (340.0, 600.0), (940.0, 600.0)


def _x_on(p_bottom: tuple[float, float], y: np.ndarray | float) -> np.ndarray:
    (x0, y0), (xv, yv) = p_bottom, VP
    return x0 + (xv - x0) * (np.asarray(y, dtype=np.float64) - y0) / (yv - y0)


def _lane_label(vertices, types=None, direction="parallel", lane_type="single white", wrap=False):
    poly = {"vertices": vertices, "types": types or "L" * len(vertices), "closed": False}
    return {
        "category": "lane",
        "attributes": {"laneDirection": direction, "laneType": lane_type, "laneStyle": "solid"},
        "poly2d": poly if wrap else [poly],
        "id": 1,
    }


def _area_label(vertices, area_type="direct", types=None):
    return {
        "category": "drivable area",
        "attributes": {"areaType": area_type},
        "poly2d": [{"vertices": vertices, "types": types or "L" * len(vertices), "closed": True}],
        "id": 2,
    }


def _frame(labels: list[dict], name: str = "f.jpg") -> dict:
    return {
        "name": name,
        "attributes": {"weather": "clear", "scene": "highway", "timeofday": "daytime"},
        "timestamp": 10000,
        "labels": labels,
    }


def _ego_frame() -> dict:
    y_top, y_bot = 400.0, 600.0
    lt, rt = float(_x_on(LEFT_BOTTOM, y_top)), float(_x_on(RIGHT_BOTTOM, y_top))
    return _frame(
        [
            # direct area = trapezoid between the two ego lines, rows 400..600
            _area_label(
                [[lt, y_top], [rt, y_top], [RIGHT_BOTTOM[0], y_bot], [LEFT_BOTTOM[0], y_bot]]
            ),
            _area_label([[0, 450], [200, 450], [100, 700]], area_type="alternative"),
            _lane_label([[LEFT_BOTTOM[0], y_bot], [lt, y_top]]),
            _lane_label([[RIGHT_BOTTOM[0], y_bot], [rt, y_top]]),
            _lane_label([[0, 640], [1279, 640]], direction="vertical", lane_type="crosswalk"),
        ]
    )


# ---------------------------------------------------------------- parsing and rasterisation


def test_vertices_are_x_then_y():
    # Wide, short rectangle: x 100..1100, y 500..700. Swapped axes would give a tall thin box.
    f = _frame([_area_label([[100, 500], [1100, 500], [1100, 700], [100, 700]])])
    m = direct_area_mask(f, (LH, LW), 8)
    assert m.shape == (LH, LW)
    assert m[600, 1000] and m[510, 120] and not m[300, 600] and not m[600, 1200]
    assert abs(int(m.sum()) - 1001 * 201) < 0.01 * 1001 * 201


def test_only_direct_area_is_ego_gt():
    f = _frame(
        [
            _area_label([[0, 400], [300, 400], [300, 700], [0, 700]], area_type="alternative"),
            _area_label([[500, 400], [800, 400], [800, 700], [500, 700]]),
        ]
    )
    m = direct_area_mask(f, (LH, LW), 8)
    assert m[550, 650] and not m[550, 150]


def test_closed_bezier_polygon_follows_curve():
    # Square whose top edge is a cubic Bezier bulging upwards to y ~ 325 at x = 640.
    verts = [[440, 600], [440, 400], [540, 300], [740, 300], [840, 400], [840, 600]]
    f = _frame([_area_label(verts, types="LLCCLL")])
    m = direct_area_mask(f, (LH, LW), 16)
    assert m[340, 640]  # inside the bulge, above the straight chord y = 400
    assert not m[310, 640]  # above the curve's apex (y = 325)
    pts = flatten_poly2d(verts, "LLCCLL", True, 16)
    assert pts[:, 1].min() == pytest.approx(325.0, abs=1.0)


def test_lane_polylines_filters_and_accepts_dict_poly2d():
    f = _frame(
        [
            _lane_label([[300, 700], [600, 400]], wrap=True),  # poly2d as a dict, not a list
            _lane_label([[0, 650], [1279, 650]], direction="vertical"),
            _lane_label([[900, 700], [700, 400]], lane_type="crosswalk"),
            _lane_label([[1000, 700], [900, 550], [800, 500], [700, 400]], types="LCCL"),
            {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}},
        ]
    )
    lanes = lane_polylines(f, 8, ["crosswalk"])
    assert len(lanes) == 2
    assert all(np.all(np.diff(p[:, 1]) >= 0) for p in lanes)  # sorted by y (top first)
    assert lanes[1].shape[0] == 9  # L + 8 Bezier samples


def test_label_scale_rescales_mask_and_lanes():
    f = _ego_frame()
    hw = (360, 640)
    sc = label_scale(hw, [LW, LH])
    assert sc == (0.5, 0.5)
    assert label_scale((LH, LW), [LW, LH]) == (1.0, 1.0)
    full = direct_area_mask(f, (LH, LW), 8)
    half = direct_area_mask(f, hw, 8, sc)
    ref = cv2.resize(full.astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST) > 0
    assert (half ^ ref).mean() < 0.01
    lanes = lane_polylines(f, 8, ["crosswalk"], sc)
    assert max(p[:, 1].max() for p in lanes) == pytest.approx(300.0)
    left, right = ego_boundaries(lanes, hw, 0.74, 0.55)
    assert left is not None and right is not None
    assert left[-1, 0] == pytest.approx(LEFT_BOTTOM[0] / 2)


def test_gt_geometry_recovers_vanishing_row_and_width():
    f = _ego_frame()
    hw = (LH, LW)
    mask = direct_area_mask(f, hw, 8)
    lanes = lane_polylines(f, 8, ["crosswalk"])
    sides = ego_boundaries(lanes, hw, 0.74, 0.55)
    rows = np.linspace(0.58 * (LH - 1), 0.74 * (LH - 1), 30)
    g = gt_geometry(mask, sides, rows)
    assert g["vp_y"] == pytest.approx(VP[1] / (LH - 1), abs=1e-3)
    y_b = rows[-1]
    width = (_x_on(RIGHT_BOTTOM, y_b) - _x_on(LEFT_BOTTOM, y_b)) / LW
    assert g["width_bottom"] == pytest.approx(float(width), abs=1e-3)
    assert g["direct_top"] == pytest.approx(400 / (LH - 1), abs=2e-3)
    assert g["direct_bottom"] == pytest.approx(600 / (LH - 1), abs=2e-3)


# ---------------------------------------------------------------- per-image accounting


class FixedDetector:
    """Returns a fixed LaneResult; mimics the detector attributes the eval reads."""

    def __init__(self, res: LaneResult, hw: tuple[int, int]) -> None:
        self.res = res
        self.last_debug_stateless = {"binary": np.zeros(hw, bool), "y_top": None, "y_bottom": None}
        self.last_timing_ms = {"binary_ms": 1.0, "fit_ms": 2.0}

    def detect(self, img, stateless=False):
        assert stateless
        return self.res


def _poly_through(p_bottom) -> Polyline:
    # x = a*y^2 + b*y + c with a = 0 through p_bottom and the vanishing point
    (x0, y0), (xv, yv) = p_bottom, VP
    b = (xv - x0) / (yv - y0)
    return Polyline(np.array([0.0, b, x0 - b * y0]))


def test_evaluate_image_perfect_prediction_and_counts():
    f = _ego_frame()
    hw = (LH, LW)
    gt = direct_area_mask(f, hw, 8)
    perfect = LaneResult(_poly_through(LEFT_BOTTOM), _poly_through(RIGHT_BOTTOM), gt.copy(), 1.0)
    dets = {
        "good": FixedDetector(perfect, hw),
        "none": FixedDetector(LaneResult(None, None, None, 0.0), hw),
    }
    timings: dict = defaultdict(list)
    img = np.zeros((LH, LW, 3), np.uint8)
    rec, vis, g = evaluate_image(dets, img, f, MCFG, timings, [LW, LH], keep_vis=True)
    assert rec["gt_sides"] == 2 and rec["geometry"] == "straight" and not rec["label_scaled"]
    good, none = rec["good"], rec["none"]
    assert good["gt_px"] > 0 and good["inter"] == good["union"] == good["gt_px"]
    assert (good["tp"], good["fp"], good["fn"]) == (2, 0, 0)
    assert none["pred_none"] and none["inter"] == 0 and (none["tp"], none["fn"]) == (0, 2)
    assert set(vis) == {"good", "none"} and timings["good/fit_ms"] == [2.0]
    rows = aggregate([rec], ["good", "none"])
    top = {r["method"]: r for r in rows if r["group"] == "all"}
    assert top["good"]["iou_mean"] == pytest.approx(1.0)
    assert top["none"]["iou_mean"] == pytest.approx(0.0)  # GT present, no prediction -> 0
    assert top["none"]["pred_none"] == 1 and top["none"]["iou_when_pred"] is None


def test_gt_empty_in_band_is_excluded_not_zero():
    # Direct area only above the band (rows 300..380 < 0.58 * 719 = 417) -> nothing to score.
    f = _frame([_area_label([[500, 300], [800, 300], [800, 380], [500, 380]])])
    hw = (LH, LW)
    band = np.zeros(hw, bool)
    band[band_slice(LH, MCFG["band_frac"])] = True
    pred = LaneResult(None, None, band, 0.5)
    dets = {"m": FixedDetector(pred, hw)}
    rec, _, _ = evaluate_image(dets, np.zeros((LH, LW, 3), np.uint8), f, MCFG, defaultdict(list))
    assert rec["m"]["gt_px"] == 0 and rec["geometry"] == "no_gt"
    top = [r for r in aggregate([rec], ["m"]) if r["group"] == "all"][0]
    assert top["iou_mean"] is None and top["iou_images"] == 0
    assert top["gt_empty"] == 1 and top["fp_mask"] == 1


def test_markdown_has_diagnostics_and_stage_columns():
    f = _ego_frame()
    hw = (LH, LW)
    dets = {"m": FixedDetector(LaneResult(None, None, None, 0.0), hw)}
    rec, _, _ = evaluate_image(dets, np.zeros((LH, LW, 3), np.uint8), f, MCFG, defaultdict(list))
    rows = aggregate([rec], ["m"])
    fps = {"m": {"frames": 1, "mean_ms": 3.0, "fit_mean_ms": 2.0, "device": "CPU"}}
    md = markdown_table(rows, ["m"], fps, [])
    assert "| GT empty | GT sides | GT short | no pred m | synth m |" in md
    assert "| IoU m | IoU measured-only m | F1 m |" in md
    assert "| Backend ms | Model ms | Fit ms |" in md and "| 2.0 |" in md


def test_spread_indices_cover_range():
    assert spread_indices(30, 8)[0] == 0 and spread_indices(30, 8)[-1] == 29
    assert len(spread_indices(30, 8)) == 8
    assert spread_indices(3, 8) == [0, 1, 2] and spread_indices(0, 8) == []


# ---------------------------------------------------------------- single-side fallback scoring


def _left_half_mask(hw: tuple[int, int]) -> np.ndarray:
    """Ego mask between the GT left line and the image centre (stand-in for a fallback mask)."""
    h, w = hw
    m = np.zeros(hw, bool)
    for y in range(h):
        x0 = int(np.ceil(_x_on(LEFT_BOTTOM, y)))
        m[y, max(x0, 0) : w // 2] = True
    return m


def test_f1_excludes_synthesized_side_and_iou_reports_both():
    f = _ego_frame()
    hw = (LH, LW)
    left = _poly_through(LEFT_BOTTOM)
    mask = _left_half_mask(hw)
    # Detector contract: synthesized side is not in the result, only in last_debug_stateless.
    fallback = FixedDetector(LaneResult(left, None, mask, 0.6), hw)
    fallback.last_debug_stateless["synthesized_side"] = "right"
    # A backend that wrongly returns its synthesized side (far off the GT right line): the eval
    # must still not score it, because last_debug_stateless marks it as synthesized.
    leaky = FixedDetector(LaneResult(left, Polyline(np.array([0, 0, 5.0])), mask, 0.6), hw)
    leaky.last_debug_stateless["synthesized_side"] = "right"
    # Same measured side, no fallback: the old behaviour.
    old = FixedDetector(LaneResult(left, None, None, 0.5), hw)
    dets = {"fallback": fallback, "leaky": leaky, "old": old}
    img = np.zeros((LH, LW, 3), np.uint8)
    rec, vis, _ = evaluate_image(dets, img, f, MCFG, defaultdict(list), keep_vis=True)

    for m in ("fallback", "leaky", "old"):
        r = rec[m]
        assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 1), m  # right side: FN, never TP/FP
        assert not r["both"] and r["n_sides"] == 1
    for m in ("fallback", "leaky"):
        r = rec[m]
        assert r["synth"] and r["synth_side"] == "right" and not r["pred_none"]
        assert r["inter"] > 0  # fallback mask is scored for IoU
        assert r["inter_meas"] == 0 and r["union_meas"] == r["gt_px"]  # measured-only: none
    assert not rec["old"]["synth"] and rec["old"]["pred_none"]

    top = {r["method"]: r for r in aggregate([rec], list(dets)) if r["group"] == "all"}
    assert top["fallback"]["synth"] == 1 and top["old"]["synth"] == 0
    assert top["fallback"]["iou_mean"] > 0.3
    assert top["fallback"]["iou_measured"] == pytest.approx(0.0)
    assert top["old"]["iou_measured"] == pytest.approx(top["old"]["iou_mean"]) == 0.0
    assert top["fallback"]["lane_f1"] == top["old"]["lane_f1"] == pytest.approx(2 / 3)


def test_one_sided_mask_without_debug_flag_is_detected_as_synth():
    # Inferred from the result alone: mask present, exactly one side None.
    f = _ego_frame()
    hw = (LH, LW)
    det = FixedDetector(LaneResult(None, _poly_through(RIGHT_BOTTOM), _left_half_mask(hw), 0.6), hw)
    rec, _, _ = evaluate_image(
        {"m": det}, np.zeros((LH, LW, 3), np.uint8), f, MCFG, defaultdict(list)
    )
    assert rec["m"]["synth"] and rec["m"]["synth_side"] == "left"
    assert (rec["m"]["tp"], rec["m"]["fp"], rec["m"]["fn"]) == (1, 0, 1)


def test_real_detector_fallback_through_eval():
    """BDD config, one painted line: the eval sees a synth mask and scores one measured side."""
    cfg = load_lane_config(ROOT / "configs" / "lanes_bdd.yaml")
    cfg["method"] = "classical"
    h, w = LH, LW
    img = np.full((h, w, 3), 90, np.uint8)
    y0, y1 = 0.56 * (h - 1), 0.78 * (h - 1)
    lt, lb = float(_x_on(LEFT_BOTTOM, y0)), float(_x_on(LEFT_BOTTOM, y1))
    cv2.line(img, (int(lb), int(y1)), (int(lt), int(y0)), (255, 255, 255), 8)
    img[int(y1) + 1 :] = 20
    rec, _, _ = evaluate_image(
        {"classical": build_lane_detector(cfg)}, img, _ego_frame(), MCFG, defaultdict(list)
    )
    r = rec["classical"]
    assert r["synth"] and r["synth_side"] == "right" and not r["pred_none"]
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 1
    assert r["inter"] > 0 and r["inter_meas"] == 0


def test_debug_tiles_with_real_configs_and_stateless_debug():
    """Failure-figure path (run after the loop on Kaggle): keep_vis + debug_tiles with the real
    lanes_eval.yaml `figure:` and lanes_bdd.yaml `viz:` keys, and the detector's stateless debug
    (synthesized line, binary) rather than its video-mode `last_debug`."""
    ecfg = yaml.safe_load((ROOT / "configs" / "lanes_eval.yaml").read_text())
    lcfg = load_lane_config(ROOT / ecfg["lanes_config"])
    lcfg["method"] = "classical"
    h, w = LH, LW
    img = np.full((h, w, 3), 90, np.uint8)
    y0, y1 = 0.56 * (h - 1), 0.78 * (h - 1)
    lt, lb = float(_x_on(LEFT_BOTTOM, y0)), float(_x_on(LEFT_BOTTOM, y1))
    cv2.line(img, (int(lb), int(y1)), (int(lt), int(y0)), (255, 255, 255), 8)
    det = build_lane_detector(lcfg)
    rec, vis, gt = evaluate_image(
        {"classical": det}, img, _ego_frame(), MCFG, defaultdict(list), keep_vis=True
    )
    v = vis["classical"]
    assert v["synth_line"] is not None and v["binary"].any()
    assert det.last_debug == {}  # stateless eval never touches the video debug
    tiles = debug_tiles(img, rec, gt, vis, lcfg, MCFG, ecfg["figure"])
    tw = int(ecfg["figure"]["tile_width"])
    assert tiles.shape == (round(h * tw / w), 2 * tw, 3) and tiles.dtype == np.uint8


# ---------------------------------------------------------------- no_gt / short GT accounting


def test_short_gt_side_is_not_counted_as_gt_or_scored():
    # Left GT line ends just inside the band top (0.58 * 719 = 417): ranked as the left ego
    # boundary (reaches below 0.55 H) but covers < min_gt_rows band rows -> not scorable.
    # Right side has no GT at all.
    f = _frame(
        [
            _area_label([[300, 420], [900, 420], [900, 530], [300, 530]]),
            _lane_label([[420, 422], [560, 300]]),
        ]
    )
    hw = (LH, LW)
    pred_left = Polyline(np.array([0.0, 0.0, 415.0]))
    pred_right = Polyline(np.array([0.0, 0.0, 900.0]))
    det = FixedDetector(LaneResult(pred_left, pred_right, None, 0.5), hw)
    rec, _, _ = evaluate_image(
        {"m": det}, np.zeros((LH, LW, 3), np.uint8), f, MCFG, defaultdict(list)
    )
    assert rec["gt_sides"] == 0 and rec["gt_short"] == 1  # GT sides now match "no_gt"
    assert rec["geometry"] == "no_gt"
    assert (rec["m"]["tp"], rec["m"]["fp"], rec["m"]["fn"]) == (0, 1, 0)  # only the right FP
    top = [r for r in aggregate([rec], ["m"]) if r["group"] == "geometry"][0]
    assert top["key"] == "no_gt" and top["gt_sides"] == 0 and top["gt_short"] == 1


def test_no_gt_group_has_no_scorable_sides():
    # Every image in "no_gt" must contribute 0 GT sides and 0 FN (recall undefined).
    ego = _ego_frame()
    short = _frame([_lane_label([[420, 422], [560, 300]])], name="s.jpg")
    empty = _frame([], name="e.jpg")
    hw = (LH, LW)
    det = FixedDetector(LaneResult(None, None, None, 0.0), hw)
    img = np.zeros((LH, LW, 3), np.uint8)
    recs = [
        evaluate_image({"m": det}, img, fr, MCFG, defaultdict(list))[0]
        for fr in (ego, short, empty)
    ]
    assert [r["geometry"] for r in recs] == ["straight", "no_gt", "no_gt"]
    rows = {r["key"]: r for r in aggregate(recs, ["m"]) if r["group"] == "geometry"}
    assert rows["no_gt"]["n"] == 2 and rows["no_gt"]["gt_sides"] == 0
    assert rows["no_gt"]["lane_recall"] is None
    assert rows["straight"]["gt_sides"] == 2 and rows["straight"]["lane_recall"] == 0.0


# ---------------------------------------------------------------- configs


def test_eval_band_matches_bdd_warp():
    cfg = yaml.safe_load((ROOT / "configs" / "lanes_eval.yaml").read_text())
    lcfg = load_lane_config(cfg["lanes_config"])
    check_band_matches_warp(cfg["metrics"]["band_frac"], lcfg)
    assert cfg["metrics"]["gt_ref_frac"] == cfg["metrics"]["band_frac"][1]
    with pytest.raises(ValueError):
        check_band_matches_warp([0.62, 0.90], lcfg)


def test_bdd_warp_sides_meet_at_bdd_horizon():
    src = np.asarray(
        load_lane_config(ROOT / "configs" / "lanes_bdd.yaml")["fitting"]["warp"]["src"]
    )
    (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = src
    for (xb, yb), (xt, yt) in (((blx, bly), (tlx, tly)), ((brx, bry), (trx, try_))):
        y_at_centre = yb + (0.5 - xb) * (yt - yb) / (xt - xb)
        assert y_at_centre == pytest.approx(0.47, abs=0.01)


def test_bdd_config_recovers_synthetic_bdd_road():
    """Lines through the BDD vanishing point, hood below 0.78 H: both sides and a mask."""
    cfg = load_lane_config(ROOT / "configs" / "lanes_bdd.yaml")
    cfg["method"] = "classical"
    h, w = 360, 640
    img = np.full((h, w, 3), 90, np.uint8)
    src = frac_to_px(cfg["fitting"]["warp"]["src"], w, h)
    y0, y1 = 0.56 * (h - 1), 0.78 * (h - 1)
    for b, t in ((src[3], src[0]), (src[2], src[1])):
        slope = (t[0] - b[0]) / (t[1] - b[1])
        p0 = (int(round(b[0] + slope * (y0 - b[1]))), int(round(y0)))
        p1 = (int(round(b[0] + slope * (y1 - b[1]))), int(round(y1)))
        cv2.line(img, p0, p1, (255, 255, 255), 4)
    img[int(y1) + 1 :] = 20  # dark hood
    det = build_lane_detector(cfg)
    res = det.detect(img, stateless=True)
    assert res.left is not None and res.right is not None and res.ego_mask is not None
    (tlx, tly), (blx, bly) = src[0], src[3]
    ys = np.linspace(tly, bly, 10)
    true_left = blx + (ys - bly) * (tlx - blx) / (tly - bly)
    assert np.abs(res.left.x_at(ys) - true_left).max() < 6
