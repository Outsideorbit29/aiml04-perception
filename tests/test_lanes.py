"""Classical lane pipeline (M3.1): synthetic geometry, smoothing and fixture smoke tests (CPU)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from perception.lanes import ClassicalLaneDetector, build_lane_detector, load_lane_config
from perception.lanes.fitting import find_lane_bases, fit_poly, sliding_window
from perception.lanes.smoothing import LaneSmoother
from perception.lanes.threshold import frac_to_px, roi_mask
from perception.types import LaneResult, Polyline

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
H, W = 360, 640

# Test-local geometry, so retuning configs/lanes.yaml cannot break the synthetic tests.
ROI = [[0.00, 0.95], [0.40, 0.60], [0.60, 0.60], [1.00, 0.95]]
SRC = [[0.44, 0.62], [0.56, 0.62], [0.80, 0.90], [0.20, 0.90]]  # TL, TR, BR, BL
DST = [[0.25, 0.00], [0.75, 0.00], [0.75, 1.00], [0.25, 1.00]]


@pytest.fixture()
def cfg() -> dict:
    c = copy.deepcopy(load_lane_config(ROOT / "configs" / "lanes.yaml"))
    c["method"] = "classical"
    c["fitting"]["roi"]["polygon"] = ROI
    c["fitting"]["warp"]["src"] = SRC
    c["fitting"]["warp"]["dst"] = DST
    return c


LineFn = Callable[[float | np.ndarray], np.ndarray]


def _line_through(p_bottom: np.ndarray, p_top: np.ndarray) -> LineFn:
    """x(y) for the straight image line through two points."""
    (x0, y0), (x1, y1) = p_bottom, p_top
    slope = (x1 - x0) / (y1 - y0)
    return lambda y: x0 + slope * (np.asarray(y, dtype=np.float64) - y0)


def _painted_road() -> tuple[np.ndarray, LineFn, LineFn]:
    """Grey road with two white straight lane lines that lie on the warp trapezoid edges."""
    img = np.full((H, W, 3), 90, np.uint8)
    src = frac_to_px(SRC, W, H)
    left_x = _line_through(src[3], src[0])
    right_x = _line_through(src[2], src[1])
    y0, y1 = 0.58 * (H - 1), H - 1.0
    for fx in (left_x, right_x):
        p0 = (int(round(fx(y0))), int(round(y0)))
        p1 = (int(round(fx(y1))), int(round(y1)))
        cv2.line(img, p0, p1, (255, 255, 255), 5)
    return img, left_x, right_x


def _check_result_types(res: LaneResult, shape_hw: tuple[int, int]) -> None:
    assert isinstance(res, LaneResult)
    assert isinstance(res.confidence, float) and 0.0 <= res.confidence <= 1.0
    for side in (res.left, res.right):
        if side is not None:
            assert isinstance(side, Polyline)
            assert side.coeffs.shape == (3,) and np.all(np.isfinite(side.coeffs))
    if res.ego_mask is not None:
        assert res.ego_mask.shape == shape_hw and res.ego_mask.dtype == bool


# ---------------------------------------------------------------- factory / interface


def test_factory_builds_classical(cfg):
    det = build_lane_detector(cfg)
    assert isinstance(det, ClassicalLaneDetector)


def test_factory_yolop_is_stub(cfg):
    cfg["method"] = "yolop"
    with pytest.raises(NotImplementedError):
        build_lane_detector(cfg)


def test_factory_rejects_unknown_method(cfg):
    cfg["method"] = "nope"
    with pytest.raises(ValueError):
        build_lane_detector(cfg)


def test_detect_rejects_grayscale(cfg):
    with pytest.raises(ValueError):
        build_lane_detector(cfg).detect(np.zeros((H, W), np.uint8))


def test_every_n_frames_reuses_last_result(cfg):
    cfg["classical"]["every_n_frames"] = 2
    det = build_lane_detector(cfg)
    img, _, _ = _painted_road()
    r0, r1, r2 = det.detect(img), det.detect(img), det.detect(img)
    assert r1 is r0 and r2 is not r0


# ---------------------------------------------------------------- geometry


def test_roi_mask_inside_and_outside():
    mask = roi_mask((H, W), ROI)
    assert mask.shape == (H, W) and mask.dtype == bool
    assert mask[int(0.9 * H), W // 2]  # road ahead
    assert not mask[int(0.2 * H), W // 2]  # sky
    assert not mask[H - 1, W // 2]  # hood row below the ROI


def test_sliding_window_recovers_curved_lines_in_birds_eye():
    # Two parallel curves x = c + k * ((H - y) / H)^2 painted directly in the bird's-eye view.
    binary = np.zeros((H, W), np.uint8)
    ys = np.arange(H, dtype=np.float64)
    truths = {}
    for name, c in (("left", 160.0), ("right", 480.0)):
        xs = c + 60.0 * ((H - ys) / H) ** 2
        truths[name] = xs
        pts = np.round(np.stack([xs, ys], 1)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(binary, [pts], False, 1, 6)
    binary = binary.astype(bool)

    bases = find_lane_bases(binary, 0.5)
    assert bases[0] is not None and bases[0] < W // 2 <= bases[1]
    for name, base in zip(("left", "right"), bases, strict=True):
        xs, ys_px, hits = sliding_window(binary, base, 9, 50, 20)
        assert hits == 9
        coeffs = fit_poly(xs, ys_px, 50)
        assert coeffs is not None
        err = np.abs(np.polyval(coeffs, ys) - truths[name])
        assert err.max() < 2.0, f"{name}: max error {err.max():.2f}px"


def test_fit_poly_needs_enough_points():
    assert fit_poly(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 3) is None


# ---------------------------------------------------------------- full pipeline on synthetic road


def test_synthetic_road_recovers_both_lines(cfg):
    img, left_x, right_x = _painted_road()
    det = build_lane_detector(cfg)
    res = det.detect(img)
    _check_result_types(res, (H, W))
    assert res.left is not None and res.right is not None
    assert res.confidence > 0.9

    ys = np.linspace(0.64 * (H - 1), 0.88 * (H - 1), 20)
    assert np.abs(res.left.x_at(ys) - left_x(ys)).max() < 3.0
    assert np.abs(res.right.x_at(ys) - right_x(ys)).max() < 3.0

    y = 0.8 * (H - 1)
    mid = 0.5 * (left_x(y) + right_x(y))
    assert res.contains(mid, y)
    assert not res.contains(left_x(y) - 30, y)
    assert not res.contains(right_x(y) + 30, y)


def test_blank_road_gives_no_lanes(cfg):
    cfg["smoothing"]["enabled"] = False
    res = build_lane_detector(cfg).detect(np.full((H, W, 3), 90, np.uint8))
    _check_result_types(res, (H, W))
    assert res.left is None and res.right is None and res.ego_mask is None
    assert res.confidence == 0.0


def test_detector_holds_lanes_over_missing_frames_then_drops(cfg):
    cfg["smoothing"].update(enabled=True, max_missed=3, missed_conf_decay=0.5)
    det = build_lane_detector(cfg)
    img, _, _ = _painted_road()
    blank = np.full((H, W, 3), 90, np.uint8)

    first = det.detect(img)
    assert first.left is not None
    confs = [first.confidence]
    for _ in range(3):  # within max_missed: held with the same geometry, decaying confidence
        held = det.detect(blank)
        assert held.left is not None and held.right is not None
        np.testing.assert_allclose(held.left.coeffs, first.left.coeffs)
        confs.append(held.confidence)
    assert all(b < a for a, b in zip(confs, confs[1:], strict=False))
    dropped = det.detect(blank)  # 4th miss > max_missed
    assert dropped.left is None and dropped.right is None and dropped.confidence == 0.0

    det.reset()
    assert det.detect(blank).left is None


# ---------------------------------------------------------------- smoother unit tests


def _smoother(**over) -> LaneSmoother:
    base = dict(enabled=True, alpha=0.5, max_missed=2, missed_conf_decay=0.5, max_jump_frac=0.1)
    base.update(over)
    return LaneSmoother(base)


def _line(c: float) -> Polyline:
    return Polyline(np.array([0.0, 0.0, c]))


def test_smoother_ema_blends_coefficients():
    sm = _smoother()
    sm.update(_line(100.0), _line(300.0), 1.0, 1.0, y_ref=99, width=640)
    left, right, cl, cr = sm.update(_line(120.0), _line(300.0), 0.8, 1.0, y_ref=99, width=640)
    assert left.coeffs[2] == pytest.approx(110.0)  # 0.5 * 120 + 0.5 * 100
    assert right.coeffs[2] == pytest.approx(300.0)
    assert cl == pytest.approx(0.8) and cr == pytest.approx(1.0)


def test_smoother_holds_then_drops_missing_side():
    sm = _smoother()
    sm.update(_line(100.0), _line(300.0), 1.0, 1.0, y_ref=99, width=640)
    l1, r1, c1, _ = sm.update(None, _line(300.0), 0.0, 1.0, y_ref=99, width=640)
    l2, _, c2, _ = sm.update(None, _line(300.0), 0.0, 1.0, y_ref=99, width=640)
    l3, r3, c3, cr3 = sm.update(None, _line(300.0), 0.0, 1.0, y_ref=99, width=640)
    assert l1.coeffs[2] == pytest.approx(100.0) and c1 == pytest.approx(0.5)
    assert l2.coeffs[2] == pytest.approx(100.0) and c2 == pytest.approx(0.25)
    assert l3 is None and c3 == 0.0  # 3rd miss > max_missed=2
    assert r3 is not None and cr3 == pytest.approx(1.0)


def test_smoother_rejects_jump_as_outlier():
    sm = _smoother()
    sm.update(_line(100.0), None, 1.0, 0.0, y_ref=99, width=640)
    left, _, cl, _ = sm.update(_line(400.0), None, 1.0, 0.0, y_ref=99, width=640)  # 300 > 64 px
    assert left.coeffs[2] == pytest.approx(100.0) and cl == pytest.approx(0.5)


def test_smoother_disabled_passes_through():
    sm = _smoother(enabled=False)
    sm.update(_line(100.0), None, 1.0, 0.0, y_ref=99, width=640)
    left, right, cl, cr = sm.update(None, _line(5.0), 0.0, 0.7, y_ref=99, width=640)
    assert left is None and cl == 0.0
    assert right.coeffs[2] == 5.0 and cr == pytest.approx(0.7)


# ---------------------------------------------------------------- real fixtures (smoke)


def test_runs_on_bdd_fixture_images():
    cfg = load_lane_config(ROOT / "configs" / "lanes.yaml")
    det = build_lane_detector(cfg)
    files = sorted((FIX / "images").glob("bdd_*.jpg"))
    assert files, "BDD fixture images missing"
    for path in files:
        img = cv2.imread(str(path))
        assert img is not None
        det.reset()
        _check_result_types(det.detect(img), img.shape[:2])


def test_runs_on_kitti_fixture_clip():
    cap = cv2.VideoCapture(str(FIX / "clip_kitti.mp4"))
    if not cap.isOpened():
        pytest.skip("OpenCV build cannot decode mp4v")
    det = build_lane_detector(load_lane_config(ROOT / "configs" / "lanes.yaml"))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _check_result_types(det.detect(frame), frame.shape[:2])
        n += 1
    cap.release()
    if n == 0:
        pytest.skip("no frames decoded from clip_kitti.mp4")
