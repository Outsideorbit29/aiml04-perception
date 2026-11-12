"""Lane pipelines (M3.1-M3.3): classical + YOLOP (mocked ONNX session) on synthetic inputs, plus
the pure BDD metric helpers of scripts/eval_lanes.py. CPU only, nothing is downloaded."""

from __future__ import annotations

import copy
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

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


def _painted_road(h: int = H, w: int = W) -> tuple[np.ndarray, LineFn, LineFn]:
    """Grey road with two white straight lane lines that lie on the warp trapezoid edges."""
    img = np.full((h, w, 3), 90, np.uint8)
    src = frac_to_px(SRC, w, h)
    left_x = _line_through(src[3], src[0])
    right_x = _line_through(src[2], src[1])
    y0, y1 = 0.58 * (h - 1), h - 1.0
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


def test_factory_yolop_without_weights_raises_and_never_downloads(cfg, tmp_path, monkeypatch):
    def no_download(*_a, **_k):
        raise AssertionError("detector must not download weights")

    monkeypatch.setattr(urllib.request, "urlretrieve", no_download)
    cfg["method"] = "yolop"
    cfg["yolop"]["weights"] = str(tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError):
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


@pytest.mark.parametrize(
    "shape_hw",
    [(180, 320), (375, 1242), (450, 800), (720, 1280)],
    ids=["bdd_fixture", "kitti", "carla", "bdd_full"],
)
def test_synthetic_road_non_default_sizes(cfg, shape_hw):
    # Fractional config must give the same image-space result at every aspect ratio
    # (checks BEV <-> image back-projection is not tied to 640x360).
    h, w = shape_hw
    cfg["smoothing"]["enabled"] = False
    img, left_x, right_x = _painted_road(h, w)
    res = build_lane_detector(cfg).detect(img)
    _check_result_types(res, (h, w))
    assert res.left is not None and res.right is not None
    assert res.confidence >= 7 / 9 - 1e-9
    ys = np.linspace(0.64 * (h - 1), 0.88 * (h - 1), 20)
    tol = max(3.0, 0.005 * w)
    assert np.abs(res.left.x_at(ys) - left_x(ys)).max() < tol
    assert np.abs(res.right.x_at(ys) - right_x(ys)).max() < tol


def test_one_detector_handles_frame_size_change(cfg):
    # LaneFitter caches the warp/ROI per size; a new size must rebuild them, not reuse stale ones.
    cfg["smoothing"]["enabled"] = False
    det = build_lane_detector(cfg)
    for h, w in ((H, W), (450, 800), (H, W)):
        img, left_x, right_x = _painted_road(h, w)
        res = det.detect(img)
        _check_result_types(res, (h, w))
        assert res.left is not None and res.right is not None
        y = 0.8 * (h - 1)
        assert abs(float(res.left.x_at(y)) - float(left_x(y))) < max(3.0, 0.005 * w)


def test_stateless_detect_ignores_and_keeps_video_state(cfg):
    # M3.3 BDD eval: independent images via detect(stateless=True), even with smoothing on and
    # every_n_frames > 1. The video state must be neither used nor changed.
    cfg["smoothing"]["enabled"] = True
    cfg["classical"]["every_n_frames"] = 2
    det = build_lane_detector(cfg)
    img, _, _ = _painted_road()
    blank = np.full((H, W, 3), 90, np.uint8)
    r0 = det.detect(img)
    assert r0.left is not None
    assert det.detect(blank, stateless=True).left is None  # no cache, no smoother hold-over
    assert det.detect(blank, stateless=True).left is None
    assert det.detect(img) is r0  # video path: 2nd frame still reuses its cached result


def test_frame_size_change_resets_video_state(cfg):
    cfg["smoothing"].update(enabled=True, alpha=0.3)
    det = build_lane_detector(cfg)
    small, _, _ = _painted_road()
    big, _, _ = _painted_road(450, 800)
    det.detect(small)
    res = det.detect(big)  # without a reset this would be blended with the 640x360 lanes
    ref = build_lane_detector(cfg).detect(big, stateless=True)
    np.testing.assert_allclose(res.left.coeffs, ref.left.coeffs)
    np.testing.assert_allclose(res.right.coeffs, ref.right.coeffs)
    assert res.ego_mask is not None and res.ego_mask.shape == (450, 800)


def test_ego_mask_spans_only_measured_rows(cfg):
    cfg["smoothing"]["enabled"] = False
    img, left_x, right_x = _painted_road()
    det = build_lane_detector(cfg)
    res = det.detect(img)
    assert res.ego_mask is not None
    y_top = int(np.ceil(det.last_debug["y_top"]))
    y_bottom = int(np.floor(det.last_debug["y_bottom"]))
    assert not res.ego_mask[:y_top].any()  # nothing above the bird's-eye region (sky)
    assert not res.ego_mask[y_bottom + 1 :].any()  # no extrapolation below the warp source
    assert res.ego_mask[y_bottom].any()
    y = int(0.8 * (H - 1))
    row = np.flatnonzero(res.ego_mask[y])
    assert row.size > 0
    assert abs(row[0] - left_x(y)) <= 3 and abs(row[-1] - right_x(y)) <= 3


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
        _check_result_types(det.detect(img, stateless=True), img.shape[:2])


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


# ---------------------------------------------------------------- QA follow-ups (M3.1 review)


def test_sobel_peak_is_measured_inside_region():
    # A huge edge in the sky must not scale down a moderate road edge inside the region.
    from perception.lanes.threshold import threshold_binary

    img = np.full((100, 100, 3), 90, np.uint8)
    img[:20, 50:] = 255  # sky: 90 -> 255 step
    img[60:, 50:] = 100  # road: 90 -> 100 step
    region = np.zeros((100, 100), bool)
    region[40:] = True
    tcfg = dict(
        sobel_kernel=3,
        sobel_x=[20, 255],
        s_channel=[170, 255],
        s_min_lightness=80,
        l_channel=[250, 255],
    )
    assert threshold_binary(img, tcfg, region)[80, 49:51].any()
    assert not threshold_binary(img, tcfg)[80, 49:51].any()  # global peak hides it


def test_yellow_test_needs_minimum_lightness():
    from perception.lanes.threshold import threshold_binary

    tcfg = dict(
        sobel_kernel=3,
        sobel_x=[20, 100],
        s_channel=[170, 255],
        s_min_lightness=80,
        l_channel=[250, 255],
    )
    dark = np.zeros((10, 10, 3), np.uint8)
    dark[:] = (0, 0, 60)  # saturated but dark (L = 30)
    yellow = np.zeros((10, 10, 3), np.uint8)
    yellow[:] = (0, 200, 255)  # yellow paint (L = 127, S = 255)
    assert not threshold_binary(dark, tcfg).any()
    assert threshold_binary(yellow, tcfg).all()


def test_overlay_is_clamped_to_measured_rows(cfg):
    from perception.lanes.render import draw_lanes

    img = np.zeros((200, 200, 3), np.uint8)
    line = Polyline(np.array([0.0, 0.0, 100.0]))
    res = LaneResult(line, None, None, 1.0)
    out = draw_lanes(img, res, 50.2, cfg["viz"], 150.0)
    rows = np.flatnonzero(out.any(axis=(1, 2)))
    t = int(cfg["viz"]["line_thickness"])
    assert rows.min() >= 51 - t and rows.max() <= 150 + t
    assert "fallback_fps" in cfg["viz"]


def test_carla_override_merges_on_base():
    base = load_lane_config(ROOT / "configs" / "lanes.yaml")
    carla = load_lane_config(ROOT / "configs" / "lanes_carla.yaml")
    assert "base" not in carla
    assert carla["fitting"]["warp"]["src"] != base["fitting"]["warp"]["src"]
    assert carla["fitting"]["warp"]["dst"] == base["fitting"]["warp"]["dst"]
    assert carla["classical"] == base["classical"]
    img, _, _ = _painted_road(450, 800)
    _check_result_types(build_lane_detector(carla).detect(img, stateless=True), (450, 800))


# ---------------------------------------------------------------- YOLOP (M3.2), mocked session


class FakeYolopSession:
    """Stands in for onnxruntime.InferenceSession: lane = near-white input pixels, so the test
    checks the letterbox, crop, resize and fit round trip without the real model."""

    def __init__(self, ycfg: dict) -> None:
        self.mean = np.asarray(ycfg["mean"], np.float32)
        self.std = np.asarray(ycfg["std"], np.float32)
        self.calls = 0
        self.shapes: list[tuple] = []

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, feed):
        assert names == ["lane_line_seg"]
        x = feed["images"]
        self.calls += 1
        self.shapes.append(x.shape)
        rgb = x[0].transpose(1, 2, 0) * self.std + self.mean
        lane = rgb.min(axis=2) > 0.9
        logits = np.stack([np.where(lane, -4.0, 4.0), np.where(lane, 4.0, -4.0)])
        return [logits[None].astype(np.float32)]


def _yolop(cfg: dict):
    from perception.lanes.yolop import YolopLaneDetector

    cfg["method"] = "yolop"
    sess = FakeYolopSession(cfg["yolop"])
    return YolopLaneDetector(cfg, session=sess), sess


def test_yolop_module_does_not_import_onnxruntime():
    code = "import sys, perception.lanes.yolop; assert 'onnxruntime' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_letterbox_params_fit_and_center():
    from perception.lanes.yolop import letterbox_params

    assert letterbox_params((720, 1280), (640, 640), False) == (0.5, 640, 360, 0, 140)
    assert letterbox_params((360, 640), (640, 640), False) == (1.0, 640, 360, 0, 140)
    assert letterbox_params((100, 100), (640, 640), True) == (6.4, 640, 640, 0, 0)


def test_lane_probability_softmax_and_sigmoid_outputs():
    from perception.lanes.yolop import lane_probability

    logits = np.array([[[2.0, -1.0]], [[-2.0, 3.0]]], np.float32)  # (2, 1, 2)
    p = lane_probability(logits)
    assert p.shape == (1, 2) and p[0, 0] < 0.5 < p[0, 1]
    sig = np.array([[[0.9, 0.2]], [[0.3, 0.7]]], np.float32)  # per-channel sigmoid like YOLOP
    np.testing.assert_array_equal(lane_probability(sig) >= 0.5, np.argmax(sig, axis=0) == 1)


@pytest.mark.parametrize("shape_hw", [(360, 640), (720, 1280), (450, 800)])
def test_yolop_mocked_recovers_lines(cfg, shape_hw):
    h, w = shape_hw
    det, sess = _yolop(cfg)
    img, left_x, right_x = _painted_road(h, w)
    res = det.detect(img, stateless=True)
    _check_result_types(res, (h, w))
    assert sess.shapes == [(1, 3, 640, 640)]
    assert res.left is not None and res.right is not None
    ys = np.linspace(0.64 * (h - 1), 0.88 * (h - 1), 20)
    tol = max(4.0, 0.006 * w)
    assert np.abs(res.left.x_at(ys) - left_x(ys)).max() < tol
    assert np.abs(res.right.x_at(ys) - right_x(ys)).max() < tol
    assert set(det.last_timing_ms) == {"pre_ms", "model_ms", "post_ms"}


def test_yolop_every_n_frames_and_stateless(cfg):
    cfg["yolop"]["every_n_frames"] = 2
    det, sess = _yolop(cfg)
    img, _, _ = _painted_road()
    r0 = det.detect(img)
    assert det.detect(img) is r0 and sess.calls == 1  # 2nd frame reuses the result
    det.detect(img, stateless=True)
    det.detect(img, stateless=True)
    assert sess.calls == 3  # stateless always runs the model


def test_ensure_weights_uses_existing_file_and_needs_url(tmp_path, monkeypatch):
    from perception.lanes.yolop import ensure_weights

    monkeypatch.setattr(urllib.request, "urlretrieve", lambda *a, **k: pytest.fail("download"))
    f = tmp_path / "w.onnx"
    f.write_bytes(b"x")
    assert ensure_weights({"weights": str(f)}) == f
    with pytest.raises(FileNotFoundError):
        ensure_weights({"weights": str(tmp_path / "none.onnx"), "weights_url": None})


# ---------------------------------------------------------------- BDD eval helpers (M3.3)


def _lane(pts: list, lane_type: str = "single white", direction: str = "parallel") -> dict:
    return {
        "category": "lane",
        "attributes": {"laneDirection": direction, "laneType": lane_type},
        "poly2d": [{"vertices": pts, "types": "L" * len(pts), "closed": False}],
    }


def _area(pts: list, area_type: str) -> dict:
    return {
        "category": "drivable area",
        "attributes": {"areaType": area_type},
        "poly2d": [{"vertices": pts, "types": "L" * len(pts), "closed": True}],
    }


BDD_FRAME = {
    "name": "x.jpg",
    "attributes": {"timeofday": "night", "weather": "rainy", "scene": "highway"},
    "labels": [
        _area([[40, 100], [60, 60], [80, 100]], "direct"),
        _area([[0, 100], [10, 60], [30, 100]], "alternative"),
        _lane([[30, 99], [45, 50]]),  # left marking, outer edge
        _lane([[34, 99], [47, 50]]),  # left marking, inner edge -> chosen
        _lane([[90, 80], [70, 50]]),  # right line, extrapolated to the reference row
        _lane([[99, 70], [95, 50]], lane_type="road curb"),
        _lane([[0, 95], [99, 95]], lane_type="crosswalk", direction="vertical"),
        _lane([[52, 40], [55, 30]]),  # ends above gt_min_bottom_frac -> ignored
    ],
}


def test_flatten_poly2d_bezier_and_closed():
    from scripts.eval_lanes import flatten_poly2d

    pts = flatten_poly2d([[0, 0], [0, 10], [10, 10], [10, 0]], "LCCL", False, 4)
    assert pts.shape == (5, 2)
    np.testing.assert_allclose(pts[0], [0, 0])
    np.testing.assert_allclose(pts[-1], [10, 0])
    np.testing.assert_allclose(pts[2], [5, 7.5])  # cubic Bezier midpoint
    assert flatten_poly2d([[0, 0], [4, 0], [4, 4]], "LLL", True, 4).shape == (3, 2)


def test_direct_area_mask_only_direct():
    from scripts.eval_lanes import direct_area_mask

    m = direct_area_mask(BDD_FRAME, (100, 100), 8)
    assert m[95, 60] and not m[95, 15] and not m[20, 60]


def test_ego_boundaries_pick_nearest_lines():
    from scripts.eval_lanes import ego_boundaries, lane_polylines

    lanes = lane_polylines(BDD_FRAME, 8, ["crosswalk"])
    assert len(lanes) == 5  # vertical crosswalk dropped
    assert all(np.all(np.diff(p[:, 1]) >= 0) for p in lanes)  # sorted by y
    left, right = ego_boundaries(lanes, (100, 100), 0.9, 0.55)
    assert left is not None and left[-1, 0] == 34  # inner edge of the left marking
    assert right is not None and right[-1, 0] == 90  # nearer than the curb


def test_score_side_cases():
    from scripts.eval_lanes import score_side

    rows = np.linspace(60, 90, 10)
    gt = np.array([[50.0, 50.0], [50.0, 99.0]])
    near, far = Polyline(np.array([0, 0, 55.0])), Polyline(np.array([0, 0, 80.0]))
    s = score_side(near, gt, rows, 10, 5)
    assert (s.tp, s.fp, s.fn) == (1, 0, 0) and s.err_px == pytest.approx(5.0)
    s = score_side(far, gt, rows, 10, 5)
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)
    assert score_side(None, gt, rows, 10, 5).fn == 1
    assert score_side(near, None, rows, 10, 5).fp == 1
    short = np.array([[50.0, 85.0], [50.0, 99.0]])  # covers < 5 sampled rows -> no GT
    s = score_side(near, short, rows, 10, 5)
    assert (s.tp, s.fp, s.fn) == (0, 1, 0)
    s = score_side(None, None, rows, 10, 5)
    assert (s.tp, s.fp, s.fn) == (0, 0, 0)


def test_band_iou_counts_and_geometry():
    from scripts.eval_lanes import band_iou_counts, band_rows, geometry_of

    gt = np.zeros((101, 10), bool)
    gt[:, :6] = True
    pred = np.zeros_like(gt)
    pred[:, 3:] = True
    assert band_iou_counts(pred, gt, [0.5, 0.6]) == (11 * 3, 11 * 10)  # rows 50..60
    assert band_iou_counts(None, gt, [0.5, 0.6]) == (0, 66)
    rows = band_rows((101, 10), [0.5, 0.9], 20)
    straight = np.array([[10.0, 0.0], [50.0, 100.0]])
    ys = np.linspace(0, 100, 30)
    curved = np.stack([10 + 0.02 * (ys - 50) ** 2, ys], 1)
    assert geometry_of([straight, None], rows, 5) == "straight"
    assert geometry_of([straight, curved], rows, 5) == "curve"
    assert geometry_of([None, None], rows, 5) == "no_gt"


def test_stratified_sample_is_deterministic():
    from scripts.eval_lanes import stratified_sample

    frames = [
        {
            "name": f"{i:03d}.jpg",
            "attributes": {"timeofday": tod},
            "labels": [{"category": "lane"}] if i % 5 else [],
        }
        for i, tod in enumerate(["daytime", "night"] * 50)
    ]
    quotas = {"daytime": 10, "night": 5, "dawn/dusk": 3}
    a = stratified_sample(frames, quotas, 42, True)
    b = stratified_sample(list(reversed(frames)), quotas, 42, True)
    assert [f["name"] for f in a] == [f["name"] for f in b]
    assert len(a) == 15 and all(f["labels"] for f in a)


def test_aggregate_and_markdown_table():
    from scripts.eval_lanes import aggregate, markdown_table, timing_stats

    def m(inter, union, tp, fp, fn):
        return {
            "inter": inter,
            "union": union,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "errs": [3.0],
            "both": tp == 2,
        }

    recs = [
        {
            "timeofday": "night",
            "weather": "rainy",
            "geometry": "curve",
            "classical": m(1, 2, 1, 1, 1),
            "yolop": m(2, 2, 2, 0, 0),
        },
        {
            "timeofday": "daytime",
            "weather": "clear",
            "geometry": "straight",
            "classical": m(0, 0, 0, 0, 0),
            "yolop": m(0, 4, 0, 2, 0),
        },
    ]
    rows = aggregate(recs, ["classical", "yolop"])
    top = {r["method"]: r for r in rows if r["group"] == "all"}
    assert top["classical"]["iou_mean"] == pytest.approx(0.5)  # 2nd image skipped (empty)
    assert top["yolop"]["iou_mean"] == pytest.approx(0.5)
    assert top["yolop"]["iou_micro"] == pytest.approx(1 / 3)
    assert top["classical"]["lane_f1"] == pytest.approx(0.5)
    assert top["yolop"]["lane_f1"] == pytest.approx(4 / 6)
    fps = {"classical": {**timing_stats([100.0, 10.0, 20.0], 1), "device": "CPU"}}
    assert fps["classical"]["fps"] == pytest.approx(1000 / 15)
    md = markdown_table(rows, ["classical", "yolop"], fps, ["note"])
    assert "| all | all | 2 |" in md and "| timeofday | night | 1 |" in md and "- note" in md


def test_find_input_falls_back_to_shallow_search(tmp_path):
    from scripts.eval_lanes import find_input

    target = tmp_path / "datasets" / "owner" / "labels" / "val.json"
    target.parent.mkdir(parents=True)
    target.write_text("[]")
    assert find_input(str(target), str(tmp_path), 4, 1) == target
    assert find_input("/nope/labels/val.json", str(tmp_path), 4, 2) == target
    with pytest.raises(FileNotFoundError):
        find_input("/nope/other.json", str(tmp_path), 4, 1)
