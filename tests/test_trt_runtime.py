"""M6 runtime backends on CPU fakes: TrtRunner buffers, exported Detector / depth / YOLOP paths.

No GPU, tensorrt, cuda-python or onnxruntime needed: ``tests/fake_trt.py`` stands in for them.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from fake_trt import FakeEngine, TensorIOMode, install, make_onnxruntime, yolo_head

from perception.depth.estimator import DepthEstimator, load_depth_config
from perception.detection.detector import Detector
from perception.detection.processing import decode_yolo
from perception.detection.runtime import TrtRunner, TrtSession
from perception.lanes.base import load_lane_config
from perception.lanes.yolop import YolopLaneDetector

REPO_ROOT = Path(__file__).resolve().parents[1]
IN, OUT = TensorIOMode.INPUT, TensorIOMode.OUTPUT


def _engine_file(tmp_path: Path, name: str) -> Path:
    """Engine files hold their own name; the fake Runtime looks engines up by that text."""
    p = tmp_path / name
    p.write_text(name)
    return p


def _double_engine(out_dtype=np.float16) -> FakeEngine:
    return FakeEngine(
        {"images": ((1, 3, 4, 4), np.float32, IN), "out": ((1, 2, 4, 4), out_dtype, OUT)},
        {"out": lambda x: x[:, :2] * 2},
    )


# ---------------------------------------------------------------- TrtRunner
def test_trt_runner_round_trip_fp16_output_and_call_order(monkeypatch, tmp_path):
    cr = install(monkeypatch, {"m.engine": _double_engine()})
    r = TrtRunner(_engine_file(tmp_path, "m.engine"))
    assert r.input_names == ["images"] and r.output_names == ["out"]
    assert r.inputs[0].host.shape == (1, 3, 4, 4) and r.inputs[0].host.dtype == np.float32
    assert r.outputs[0].host.dtype == np.float16 and r.outputs[0].nbytes == 2 * 2 * 16
    x = np.arange(48, dtype=np.float64).reshape(1, 3, 4, 4) / 10  # float64 is cast in
    (y,) = r(x)
    assert y.dtype == np.float16 and y.shape == (1, 2, 4, 4)
    np.testing.assert_allclose(y, (x[:, :2] * 2).astype(np.float16))
    kinds = [c[0] for c in cr.calls]
    assert kinds == ["h2d", "execute", "d2h", "sync"]  # copies around one execute, then sync
    # outputs are copies: the next call must not overwrite an earlier result
    (y2,) = r(np.zeros_like(x))
    np.testing.assert_allclose(y, (x[:, :2] * 2).astype(np.float16))
    assert not y2.any()


def test_trt_runner_rejects_bad_inputs(monkeypatch, tmp_path):
    install(monkeypatch, {"m.engine": _double_engine()})
    r = TrtRunner(_engine_file(tmp_path, "m.engine"))
    with pytest.raises(ValueError, match="expected shape"):
        r(np.zeros((1, 3, 8, 8), np.float32))
    with pytest.raises(ValueError, match="takes 1 inputs"):
        r(np.zeros((1, 3, 4, 4)), np.zeros((1, 3, 4, 4)))
    with pytest.raises(TypeError):  # complex -> float is not a same_kind cast
        r(np.zeros((1, 3, 4, 4), np.complex64))


def test_trt_runner_rejects_dynamic_unknown_and_missing_engines(monkeypatch, tmp_path):
    dyn = FakeEngine(
        {"images": ((-1, 3, 4, 4), np.float32, IN), "out": ((-1, 2), np.float32, OUT)}, {}
    )
    install(monkeypatch, {"dyn.engine": dyn})
    with pytest.raises(ValueError, match="dynamic shape"):
        TrtRunner(_engine_file(tmp_path, "dyn.engine"))
    with pytest.raises(RuntimeError, match="rebuild it"):  # other GPU / TensorRT version
        TrtRunner(_engine_file(tmp_path, "other_gpu.engine"))
    with pytest.raises(FileNotFoundError, match="build_local_engine"):
        TrtRunner(tmp_path / "missing.engine")


def test_trt_runner_close_frees_buffers_and_stream(monkeypatch, tmp_path):
    cr = install(monkeypatch, {"m.engine": _double_engine()})
    r = TrtRunner(_engine_file(tmp_path, "m.engine"))
    assert len(cr.device) == 2 and len(cr.host) == 2 and len(cr.streams) == 1
    r.close()
    assert not cr.device and not cr.host and not cr.streams
    assert r.inputs == [] and r.outputs == []


def test_trt_session_returns_requested_outputs_in_order(monkeypatch, tmp_path):
    eng = FakeEngine(
        {
            "images": ((1, 3, 4, 4), np.float32, IN),
            "a": ((1, 1), np.float32, OUT),
            "b": ((1, 1), np.float32, OUT),
        },
        {"a": lambda x: np.ones((1, 1)), "b": lambda x: np.full((1, 1), 2.0)},
    )
    install(monkeypatch, {"s.engine": eng})
    s = TrtSession(_engine_file(tmp_path, "s.engine"))
    x = np.zeros((1, 3, 4, 4), np.float32)
    b, a = s.run(["b", "a"], {"images": x})
    assert float(a[0, 0]) == 1.0 and float(b[0, 0]) == 2.0
    assert s.get_inputs()[0].shape == [1, 3, 4, 4] and s.get_providers() == ["TensorRT"]


# ---------------------------------------------------------------- Detector (exported backends)
DET_CFG = {"weights": "y.pt", "imgsz": 64, "conf": 0.25, "iou": 0.7, "names": ["car", "person"]}
# One car at letterbox (cx, cy, w, h) = (32, 32, 16, 16), plus a weak person box under conf.
HEAD = yolo_head(np.array([[32, 32, 16, 16], [10, 10, 4, 4]]), np.array([0, 1]),
                 np.array([0.9, 0.1]), nc=2, n=8)


def test_tensorrt_detector_end_to_end_with_fp16_engine(monkeypatch, tmp_path):
    eng = FakeEngine(
        {"images": ((1, 3, 64, 64), np.float32, IN), "output0": ((1, 6, 8), np.float16, OUT)},
        {"output0": lambda x: HEAD},
    )
    install(monkeypatch, {"y.engine": eng})
    _engine_file(tmp_path, "y.engine")
    det = Detector({**DET_CFG, "backend": "tensorrt", "weights": str(tmp_path / "y.pt")})
    assert det.backend.path == tmp_path / "y.engine"  # .pt in the config -> .engine next to it
    # 128x128 frame -> letterbox ratio 0.5, no padding: (24..40) in letterbox -> (48..80)
    (d,) = det.detect(np.zeros((128, 128, 3), np.uint8))
    np.testing.assert_allclose(d.bbox_xyxy, [48, 48, 80, 80], atol=0.1)
    assert d.cls_name == "car" and d.score == pytest.approx(0.9, abs=1e-3)
    assert d.bbox_xyxy.dtype == np.float32


def test_tensorrt_detector_engine_size_must_match_imgsz(monkeypatch, tmp_path):
    eng = FakeEngine(
        {"images": ((1, 3, 64, 64), np.float32, IN), "output0": ((1, 6, 8), np.float32, OUT)},
        {"output0": lambda x: HEAD},
    )
    install(monkeypatch, {"y.engine": eng})
    _engine_file(tmp_path, "y.engine")
    with pytest.raises(ValueError, match="64x64"):
        Detector({**DET_CFG, "backend": "tensorrt", "weights": str(tmp_path / "y.engine"),
                  "imgsz": 32})


def test_onnx_detector_with_fake_onnxruntime(monkeypatch, tmp_path):
    ort = make_onnxruntime([1, 3, 64, 64], {"output0": lambda x: HEAD},
                           meta={"names": "{0: 'car', 1: 'person'}"})
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    (tmp_path / "y.onnx").write_bytes(b"x")
    cfg = {**DET_CFG, "backend": "onnx", "weights": str(tmp_path / "y.pt"), "names": None}
    det = Detector(cfg)
    assert det.names == {0: "car", 1: "person"}  # from the ONNX metadata
    assert det.backend.providers == ["CPUExecutionProvider"]  # CUDA missing -> CPU fallback
    (d,) = det.detect(np.zeros((128, 128, 3), np.uint8))
    np.testing.assert_allclose(d.bbox_xyxy, [48, 48, 80, 80], atol=1e-3)
    with pytest.raises(ValueError, match="64x64"):
        Detector({**cfg, "imgsz": 32})


@pytest.mark.parametrize("agnostic", [False, True])
def test_decode_yolo_max_det_early_stop_matches_slice(agnostic):
    rng = np.random.default_rng(1)
    n, nc = 400, 3
    raw = np.zeros((1, 4 + nc, n), np.float32)
    raw[0, :2] = rng.uniform(20, 600, (2, n))
    raw[0, 2:4] = rng.uniform(8, 120, (2, n))
    raw[0, 4:] = rng.uniform(0, 1, (nc, n))
    full = decode_yolo(raw, 0.001, 0.7, 10_000, agnostic=agnostic)
    for k in (1, 5, 50):
        cut = decode_yolo(raw, 0.001, 0.7, k, agnostic=agnostic)
        assert len(cut.boxes) == min(k, len(full.boxes))
        np.testing.assert_array_equal(cut.boxes, full.boxes[:k])
        np.testing.assert_array_equal(cut.scores, full.scores[:k])
        np.testing.assert_array_equal(cut.cls, full.cls[:k])


# ---------------------------------------------------------------- depth backends
def _depth_cfg(tmp_path: Path, backend: str) -> dict:
    cfg = load_depth_config("configs/depth.yaml", backend=backend)
    cfg["models"]["midas_small"]["onnx"] = str(tmp_path / "midas_small.onnx")
    return cfg


def test_depth_tensorrt_backend_runs_and_checks_input_size(monkeypatch, tmp_path):
    eng = FakeEngine(
        {
            "image": ((1, 3, 256, 448), np.float32, IN),
            "inv_depth": ((1, 256, 448), np.float16, OUT),
        },
        {"inv_depth": lambda x: x[:, 0] + 3.0},
    )
    install(monkeypatch, {"midas_small.engine": eng})
    _engine_file(tmp_path, "midas_small.engine")
    est = DepthEstimator(_depth_cfg(tmp_path, "tensorrt"))  # engine = ONNX path with .engine
    out = est.estimate(np.full((90, 160, 3), 128, np.uint8))
    assert out.shape == (90, 160) and out.dtype == np.float32 and np.isfinite(out).all()
    cfg = _depth_cfg(tmp_path, "tensorrt")
    cfg["models"]["midas_small"]["input_hw"] = [256, 832]
    with pytest.raises(ValueError, match="256x448"):
        DepthEstimator(cfg)


def test_depth_onnx_backend_checks_input_size(monkeypatch, tmp_path):
    ort = make_onnxruntime([1, 3, 256, 448], {"inv_depth": lambda x: x[:, 0]})
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    (tmp_path / "midas_small.onnx").write_bytes(b"x")
    est = DepthEstimator(_depth_cfg(tmp_path, "onnx"))
    assert est.estimate(np.zeros((90, 160, 3), np.uint8)).shape == (90, 160)
    cfg = _depth_cfg(tmp_path, "onnx")
    cfg["models"]["midas_small"]["input_hw"] = [256, 832]
    with pytest.raises(ValueError, match="256x448"):
        DepthEstimator(cfg)


# ---------------------------------------------------------------- YOLOP on an engine
def _yolop_engine() -> FakeEngine:
    def lane(x):  # probabilities: lane channel high in a vertical stripe
        ll = np.zeros((1, 2, 640, 640), np.float32)
        ll[0, 0] = 0.9
        ll[0, 1, :, 300:340] = 0.95
        ll[0, 0, :, 300:340] = 0.05
        return ll

    return FakeEngine(
        {
            "images": ((1, 3, 640, 640), np.float32, IN),
            "det_out": ((1, 4, 6), np.float32, OUT),
            "drive_area_seg": ((1, 2, 640, 640), np.float16, OUT),
            "lane_line_seg": ((1, 2, 640, 640), np.float16, OUT),
        },
        {"det_out": lambda x: np.zeros((1, 4, 6)),
         "drive_area_seg": lambda x: np.zeros((1, 2, 640, 640)), "lane_line_seg": lane},
    )


def test_yolop_lane_detector_on_tensorrt_engine(monkeypatch, tmp_path):
    install(monkeypatch, {"yolop.engine": _yolop_engine()})
    _engine_file(tmp_path, "yolop.engine")
    cfg = load_lane_config("configs/lanes_carla.yaml")
    cfg["method"] = "yolop"
    cfg["yolop"]["weights"] = str(tmp_path / "yolop.engine")
    lanes = YolopLaneDetector(cfg)
    assert lanes.providers == ["TensorRT"]
    mask = lanes.lane_binary(np.zeros((360, 640, 3), np.uint8))
    assert mask.shape == (360, 640) and mask[:, 310:330].all() and not mask[:, :250].any()


# ---------------------------------------------------------------- laptop path: no torch
def test_exported_backends_build_and_run_without_torch(tmp_path):
    """Detector (tensorrt + onnx), depth (tensorrt) and YOLOP (engine) on the fakes, in a fresh
    interpreter: torch / ultralytics must never be imported on the laptop runtime path."""
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        import numpy as np
        sys.path.insert(0, {str(REPO_ROOT / "tests")!r})
        import fake_trt
        from test_trt_runtime import HEAD, _yolop_engine
        import perception.detection.runtime as runtime
        tmp = Path({str(tmp_path)!r})
        IN, OUT = fake_trt.TensorIOMode.INPUT, fake_trt.TensorIOMode.OUTPUT
        engines = {{
            "y.engine": fake_trt.FakeEngine(
                {{"images": ((1, 3, 64, 64), np.float32, IN),
                  "output0": ((1, 6, 8), np.float16, OUT)}}, {{"output0": lambda x: HEAD}}),
            "midas_small.engine": fake_trt.FakeEngine(
                {{"image": ((1, 3, 256, 448), np.float32, IN),
                  "inv_depth": ((1, 256, 448), np.float16, OUT)}},
                {{"inv_depth": lambda x: x[:, 0] + 3}}),
            "yolop.engine": _yolop_engine(),
        }}
        for name in engines:
            (tmp / name).write_text(name)
        (tmp / "y.onnx").write_bytes(b"x")
        cr = fake_trt.FakeCudart()
        sys.modules["tensorrt"] = fake_trt.make_tensorrt(engines, cr)
        sys.modules["onnxruntime"] = fake_trt.make_onnxruntime(
            [1, 3, 64, 64], {{"output0": lambda x: HEAD}})
        runtime._cudart = lambda: cr
        from perception.depth.estimator import DepthEstimator, load_depth_config
        from perception.detection.detector import Detector
        from perception.lanes.base import build_lane_detector, load_lane_config
        frame = np.zeros((128, 128, 3), np.uint8)
        for backend in ("tensorrt", "onnx"):
            det = Detector({{"backend": backend, "weights": str(tmp / "y.pt"), "imgsz": 64,
                            "conf": 0.25, "iou": 0.7}})
            assert len(det.detect(frame)) == 1, backend
        d = load_depth_config("configs/depth.yaml", backend="tensorrt")
        d["models"]["midas_small"]["onnx"] = str(tmp / "midas_small.onnx")
        DepthEstimator(d).estimate(frame)
        lc = load_lane_config("configs/lanes_carla.yaml")
        lc["method"] = "yolop"
        lc["yolop"]["weights"] = str(tmp / "yolop.engine")
        build_lane_detector(lc).lane_binary(frame)
        bad = [m for m in ("torch", "ultralytics", "torchvision") if m in sys.modules]
        assert not bad, bad
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=REPO_ROOT)
