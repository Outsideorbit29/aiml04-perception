"""M6 deploy helpers and exported-model backends (CPU only: no torch, onnxruntime or TensorRT)."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.benchmark import accuracy_cells, markdown_report
from deploy.build_local_engine import expand
from deploy.build_local_engine import main as build_main
from deploy.common import blob_fn, fixture_frames, load_bench_config, runtime_configs, spread
from deploy.metrics import (
    MapAccumulator,
    abs_rel,
    compute_ap,
    latency_stats,
    mask_iou,
    match_predictions,
    merge_parity,
    parity,
)
from deploy.trt_build import engine_path_for
from perception.depth.estimator import DepthEstimator, engine_path, load_depth_config
from perception.detection.detector import Detector, backend_weights, names_from_metadata
from perception.detection.processing import nms
from perception.detection.runtime import (
    TrtSession,
    check_input_hw,
    ort_providers,
    static_shape,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------- metrics
def test_parity_and_merge():
    ref = np.array([[0.0, 1.0], [2.0, 4.0]])
    p = parity(ref, ref + np.array([[0.0, 0.1], [0.0, -0.2]]))
    assert p["max_abs"] == pytest.approx(0.2)
    assert p["mean_abs"] == pytest.approx(0.075)
    assert p["ref_range"] == pytest.approx(4.0)
    assert p["max_abs_rel_range"] == pytest.approx(0.05)
    m = merge_parity([p, parity(ref, ref)])
    assert m["max_abs"] == pytest.approx(0.2) and m["frames"] == 2
    assert m["mean_abs"] == pytest.approx(0.0375)
    with pytest.raises(ValueError):
        parity(np.zeros(3), np.zeros(4))


def test_latency_stats():
    s = latency_stats([10.0] * 95 + [20.0] * 5)
    assert s["p50_ms"] == 10.0 and s["iters"] == 100
    assert s["mean_ms"] == pytest.approx(10.5)
    assert s["fps"] == pytest.approx(1000 / 10.5)
    assert latency_stats([]) == {}


def test_match_predictions_one_to_one_and_class_aware():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
    gt_cls = np.array([0, 1])
    pred = np.array(
        [[0, 0, 10, 10], [0, 0, 10, 9], [20, 20, 30, 30], [50, 50, 60, 60]], np.float32
    )
    pred_cls = np.array([0, 0, 0, 1])  # 3rd is on GT 1 but with the wrong class
    tp = match_predictions(pred, pred_cls, gt, gt_cls)
    assert tp.shape == (4, 10)
    assert tp[0].all()  # exact match at every threshold
    assert not tp[1].any()  # duplicate of GT 0: the better box took it
    assert not tp[2].any() and not tp[3].any()
    assert not match_predictions(pred, pred_cls, gt[:0], gt_cls[:0]).any()


def test_compute_ap_matches_ultralytics_conventions():
    # Ultralytics appends (recall 1, precision 0), so a perfect curve scores 0.995, not 1
    assert compute_ap(np.array([1.0]), np.array([1.0])) == pytest.approx(0.995)
    # FP first, then TP: precision 0.5 at recall 1 -> envelope 0.5 (x 0.995)
    assert compute_ap(np.array([0.0, 1.0]), np.array([0.0, 0.5])) == pytest.approx(0.4975)


def test_map_accumulator():
    gt = np.array([[0, 0, 10, 10]], np.float32)
    m = MapAccumulator()
    m.add(gt, np.array([0.9]), np.array([2]), gt, np.array([2]))
    m.add(np.zeros((0, 4)), np.zeros(0), np.zeros(0), gt, np.array([2]))  # missed
    r = m.result()
    assert r["images"] == 2 and r["classes"] == 1
    # precision 1 up to recall 0.5, then linear to (1, 0) as in Ultralytics: 0.5 + 0.25
    assert r["map50"] == pytest.approx(0.75, abs=0.01)
    hi_fp = MapAccumulator()  # a higher-scored false positive halves precision
    hi_fp.add(
        np.array([[50, 50, 60, 60], [0, 0, 10, 10]], np.float32),
        np.array([0.95, 0.9]), np.array([2, 2]), gt, np.array([2]),
    )
    assert hi_fp.result()["map50"] == pytest.approx(0.4975)
    assert MapAccumulator().result()["images"] == 0


def test_abs_rel_and_mask_iou():
    ref = np.array([1.0, 2.0, 0.0])
    assert abs_rel(ref, np.array([1.1, 1.8, 5.0])) == pytest.approx(0.1)
    a = np.zeros((4, 4), bool)
    b = a.copy()
    assert mask_iou(a, b) == 1.0
    a[0, :2] = True
    b[0, 1:3] = True
    assert mask_iou(a, b) == pytest.approx(1 / 3)


def test_nms_max_keep_equals_slice():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 500, (300, 2)).astype(np.float32)
    boxes = np.concatenate([xy, xy + rng.uniform(5, 60, (300, 2))], axis=1)
    scores = rng.uniform(0, 1, 300).astype(np.float32)
    full = nms(boxes, scores, 0.5)
    np.testing.assert_array_equal(nms(boxes, scores, 0.5, max_keep=7), full[:7])
    np.testing.assert_array_equal(nms(boxes, scores, 0.5, max_keep=10_000), full)


# ---------- runtime helpers and backends
def test_ort_providers_default_skips_tensorrt_ep():
    avail = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert ort_providers(None, avail) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert ort_providers(["CUDAExecutionProvider"], ["CPUExecutionProvider"]) == [
        "CPUExecutionProvider"
    ]


def test_static_shape_and_input_check():
    assert static_shape([1, 3, 640, 640]) == (1, 3, 640, 640)
    assert static_shape(["batch", 3, 640, 640]) is None
    check_input_hw([1, 3, 256, 448], (256, 448), "m")
    check_input_hw(["n", 3, "h", "w"], (256, 832), "m")  # dynamic: accepted
    with pytest.raises(ValueError, match="256x448"):
        check_input_hw([1, 3, 256, 448], (256, 832), "m")


def test_backend_weights_switches_suffix(tmp_path):
    pt = tmp_path / "yolo11s-bdd.pt"
    assert backend_weights(pt, "onnx") == tmp_path / "yolo11s-bdd.onnx"
    assert backend_weights(pt, "tensorrt") == tmp_path / "yolo11s-bdd.engine"
    assert backend_weights(tmp_path / "m.onnx", "onnx") == tmp_path / "m.onnx"
    assert backend_weights(tmp_path / "m.plan", "tensorrt") == tmp_path / "m.plan"


def test_names_from_ultralytics_metadata():
    assert names_from_metadata({"names": "{0: 'car', 1: 'truck'}"}) == {0: "car", 1: "truck"}
    assert names_from_metadata({}) is None
    assert names_from_metadata({"names": "not python"}) is None


def test_exported_backends_raise_clearly_without_files(tmp_path):
    cfg = {"backend": "onnx", "weights": str(tmp_path / "y.pt"), "imgsz": 640, "conf": 0.1,
           "iou": 0.7}
    with pytest.raises(FileNotFoundError, match="y.onnx"):
        Detector(cfg)
    with pytest.raises(FileNotFoundError, match="build_local_engine"):
        Detector({**cfg, "backend": "tensorrt"})
    dcfg = load_depth_config("configs/depth.yaml", backend="tensorrt")
    dcfg["models"]["midas_small"]["onnx"] = str(tmp_path / "midas_small.onnx")
    with pytest.raises(FileNotFoundError, match="midas_small.engine"):
        DepthEstimator(dcfg)


def test_depth_engine_path_default_and_override(tmp_path):
    assert engine_path({"onnx": str(tmp_path / "m.onnx")}) == tmp_path / "m.engine"
    assert engine_path({"onnx": "x.onnx", "engine": str(tmp_path / "e.plan")}) == (
        tmp_path / "e.plan"
    )


class FakeRunner:
    def __init__(self):
        self.inputs = [SimpleNamespace(name="images", shape=(1, 3, 4, 4))]
        self.outputs = [
            SimpleNamespace(name="det_out", shape=(1, 2)),
            SimpleNamespace(name="lane_line_seg", shape=(1, 2, 4, 4)),
        ]
        self.input_names = ["images"]
        self.output_names = ["det_out", "lane_line_seg"]

    def __call__(self, x):
        return [np.zeros((1, 2)), x[:, :2] * 2]


def test_trt_session_mimics_onnxruntime():
    s = TrtSession("unused.engine", runner=FakeRunner())
    assert s.get_inputs()[0].name == "images"
    assert [o.name for o in s.get_outputs()] == ["det_out", "lane_line_seg"]
    x = np.ones((1, 3, 4, 4), np.float32)
    (ll,) = s.run(["lane_line_seg"], {"images": x})
    np.testing.assert_allclose(ll, 2.0)
    assert len(s.run(None, {"images": x})) == 2
    with pytest.raises(KeyError):
        s.run(None, {"other": x})


# ---------- deploy scripts
def test_bench_config_smoke_merge():
    full = load_bench_config("configs/bench.yaml")
    smoke = load_bench_config("configs/bench.yaml", smoke=True)
    assert full["bench"]["warmup"] == 200 and full["bench"]["iters"] == 1000
    assert smoke["bench"]["iters"] < full["bench"]["iters"]
    assert smoke["bench"]["models"] == full["bench"]["models"]
    assert smoke["wandb"]["artifact_suffix"] == "-smoke" and not full["wandb"]["artifact_suffix"]


def test_fixture_frames_and_live_input_blobs():
    cfg = load_bench_config("configs/bench.yaml")
    frames = fixture_frames(cfg)
    assert len(frames) == cfg["parity"]["frames"] == 20
    rc = runtime_configs(cfg)
    shapes = {m: blob_fn(m, rc)(frames[0]).shape for m in ("yolo", "midas_small", "yolop")}
    assert shapes == {
        "yolo": (1, 3, 640, 640),
        "midas_small": (1, 3, 256, 448),
        "yolop": (1, 3, 640, 640),
    }
    # file names match what the live configs resolve to
    assert Path(rc["depth"]["models"]["midas_small"]["onnx"]).name == cfg["files"]["midas_small"]
    assert Path(rc["lanes"]["yolop"]["weights"]).name == cfg["files"]["yolop"]
    assert backend_weights(rc["detector"]["weights"], "onnx").name == cfg["files"]["yolo"]


def test_spread():
    assert spread(list(range(10)), 3) == [0, 4, 9] or spread(list(range(10)), 3) == [0, 5, 9]
    assert spread([1, 2], 5) == [1, 2]


def test_engine_paths_and_glob_expand(tmp_path):
    (tmp_path / "a.onnx").write_bytes(b"x")
    (tmp_path / "b.onnx").write_bytes(b"x")
    files = expand([str(tmp_path / "*.onnx"), str(tmp_path / "a.onnx")])
    assert [f.name for f in files] == ["a.onnx", "b.onnx"]
    assert engine_path_for(tmp_path / "a.onnx") == tmp_path / "a.engine"
    assert engine_path_for(tmp_path / "a.onnx", tmp_path / "e") == tmp_path / "e" / "a.engine"
    assert build_main(["--onnx", str(tmp_path / "missing.onnx")]) == 2


def _row(model, backend, p50, acc=None, **kw):
    return {
        "model": model, "backend": backend, "ok": True,
        "latency_model": {"p50_ms": p50, "p95_ms": p50 * 1.2, "fps": 1000 / p50},
        "latency_e2e": {"p50_ms": p50 + 5, "fps": 1000 / (p50 + 5)},
        "vram": {"process_peak_mb": 500.0}, "accuracy": acc or {}, **kw,
    }


def test_markdown_report_rows_and_deltas():
    results = [
        _row("yolo", "torch", 20.0, {"map50": 0.55, "map50_95": 0.31}, write_ref=True),
        _row("yolo", "onnx", 10.0, {"map50": 0.55, "map50_95": 0.31},
             parity={"boxes": {"max_abs": 0.01}, "scores": {"max_abs": 1e-4}}),
        _row("yolo", "tensorrt", 5.0, {"map50": 0.548, "map50_95": 0.309}),
        _row("midas_small", "torch", 14.0, {"abs_rel_vs_ref": 0.0, "is_reference": True},
             write_ref=True),
        _row("midas_small", "tensorrt", 4.0, {"abs_rel_vs_ref": 0.004}),
        {"model": "yolop", "backend": "tensorrt", "ok": False, "error": "no engine | x"},
    ]
    md = markdown_report(results, {"env": {"gpu": "Tesla T4", "tensorrt-cu12": "10.9"},
                                   "warmup": 200, "iters": 1000, "reference": {}})
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert all(ln.count("|") == lines[0].count("|") for ln in lines)  # 12 columns everywhere
    yolo_trt = next(ln for ln in lines if "YOLO11s" in ln and "TensorRT" in ln)
    assert "4.00x" in yolo_trt and "-0.0020 / -0.0010" in yolo_trt
    assert "ref" in next(ln for ln in lines if "YOLO11s" in ln and "PyTorch" in ln)
    assert "boxes 0.01" in next(ln for ln in lines if "YOLO11s" in ln and "ONNX" in ln)
    assert "failed: no engine / x" in md
    assert "Tesla T4" in md and "tensorrt-cu12 10.9" in md


def test_accuracy_cells_depth_and_lanes():
    assert accuracy_cells("midas_small", {"accuracy": {"abs_rel_vs_ref": 0.01}}, None)[1] == (
        "+0.0100"
    )
    cell, delta = accuracy_cells("yolop", {"accuracy": {"lane_iou_vs_ref": 0.97}}, None)
    assert "0.9700" in cell and delta == "-0.0300"
    assert accuracy_cells("yolo", {"accuracy": {}}, None) == ("–", "–")


def test_runtime_and_deploy_imports_stay_light():
    """Laptop runtime: no torch; and nothing imports tensorrt / onnxruntime at import time."""
    code = (
        "import sys; import perception.detection.runtime, perception.detection, "
        "perception.depth, deploy.benchmark, deploy.export, deploy.build_local_engine, "
        "deploy.torch_ref; "
        "bad = [m for m in ('torch', 'tensorrt', 'onnxruntime', 'ultralytics', 'wandb') "
        "if m in sys.modules]; assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)
