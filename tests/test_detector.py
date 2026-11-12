"""Detector and shared pre/post-processing (CPU only: fake backends, no torch, no weights)."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from perception.detection import BACKENDS, Detector, ExportedBackend, load_config
from perception.detection.detector import REQUIRED_KEYS, load_names, result_to_raw
from perception.detection.processing import (
    RawDetections,
    box_iou,
    decode_yolo,
    letterbox,
    nms,
    scale_boxes,
    to_blob,
)
from perception.types import Detection

FIX = Path(__file__).parent / "fixtures"
CFG = {
    "backend": "torch",
    "weights": "unused.pt",
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.7,
    "max_det": 300,
    "classes": None,
    "data_config": "configs/data_bdd.yaml",
}


class FakeBackend:
    names = {0: "fake0", 4: "fake4"}

    def __init__(self, raw: RawDetections) -> None:
        self.raw = raw

    def __call__(self, frame: np.ndarray) -> RawDetections:
        return self.raw


def fixture_frame() -> np.ndarray:
    img = cv2.imread(str(sorted((FIX / "images").glob("*.jpg"))[0]))
    assert img is not None and img.shape == (180, 320, 3)
    return img


def test_detect_returns_contract_dataclasses_with_bdd_names():
    raw = RawDetections(
        np.array([[10, 20, 50, 80], [0, 0, 5, 5]], np.float32),
        np.array([0.9, 0.1], np.float32),
        np.array([0, 4]),
    )
    out = Detector(CFG, backend=FakeBackend(raw)).detect(fixture_frame())
    assert len(out) == 1  # 0.1 is below conf 0.25
    d = out[0]
    assert isinstance(d, Detection)
    assert d.cls == 0 and d.cls_name == "car" and d.score == pytest.approx(0.9)
    assert d.bbox_xyxy.dtype == np.float32
    np.testing.assert_allclose(d.bbox_xyxy, [10, 20, 50, 80])


def test_names_fall_back_to_backend():
    raw = RawDetections(np.array([[1, 1, 9, 9]], np.float32), np.array([0.5]), np.array([4]))
    cfg = {k: v for k, v in CFG.items() if k != "data_config"}
    out = Detector(cfg, backend=FakeBackend(raw)).detect(np.zeros((20, 20, 3), np.uint8))
    assert out[0].cls_name == "fake4"


def test_empty_output_and_bad_frame():
    det = Detector(CFG, backend=FakeBackend(RawDetections(np.zeros((0, 4)), np.zeros(0), [])))
    assert det.detect(np.zeros((10, 10, 3), np.uint8)) == []
    with pytest.raises(ValueError):
        det.detect(np.zeros((10, 10), np.uint8))


def test_backend_selection_errors():
    for kind in ("onnx", "tensorrt"):  # filled in M6
        with pytest.raises(NotImplementedError):
            Detector({**CFG, "backend": kind})
    with pytest.raises(ValueError):
        Detector({**CFG, "backend": "openvino"})
    with pytest.raises(KeyError):
        Detector({"backend": "torch"})


def test_repo_config_is_complete():
    cfg = load_config("configs/yolo_detector.yaml")
    assert all(k in cfg for k in REQUIRED_KEYS)
    assert cfg["backend"] in BACKENDS
    names = load_names(cfg)
    assert names[0] == "car" and names[8] == "traffic sign" and len(names) == 9
    assert load_config("configs/yolo_detector.yaml", conf=0.5)["conf"] == 0.5


def test_result_to_raw_accepts_tensor_like_and_numpy():
    class T:  # mimics a torch tensor: .cpu().numpy()
        def __init__(self, a):
            self.a = np.asarray(a)

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    for wrap in (np.asarray, T):
        boxes = SimpleNamespace(
            xyxy=wrap([[1.0, 2.0, 3.0, 4.0]]), conf=wrap([0.5]), cls=wrap([2.0])
        )
        raw = result_to_raw(SimpleNamespace(boxes=boxes))
        assert raw.boxes.shape == (1, 4) and raw.cls.dtype == np.int64 and raw.cls[0] == 2


def test_letterbox_and_blob():
    img = np.full((180, 320, 3), 255, np.uint8)
    out, r, (left, top) = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and r == pytest.approx(2.0)
    assert (left, top) == (0, 140)
    assert out[0, 0, 0] == 114 and out[320, 320, 0] == 255
    blob = to_blob(out)
    assert blob.shape == (1, 3, 640, 640) and blob.dtype == np.float32 and blob.max() <= 1.0


def test_scale_boxes_inverts_letterbox_and_clips():
    boxes = np.array([[80, 200, 240, 320], [-10, 100, 700, 600]], np.float32)
    out = scale_boxes(boxes, 2.0, (0, 140), (180, 320))
    np.testing.assert_allclose(out[0], [40, 30, 120, 90])
    assert out[1, 0] == 0 and out[1, 2] == 320 and out[1, 3] == 180


def test_box_iou_and_nms():
    a = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]], np.float32)
    iou = box_iou(a, a)
    assert iou[0, 0] == pytest.approx(1.0) and iou[0, 1] == pytest.approx(1 / 3)
    assert iou[0, 2] == 0
    keep = nms(a, np.array([0.9, 0.8, 0.7]), iou_thr=0.3)
    assert keep.tolist() == [0, 2]


def make_raw(rows: list[tuple[float, float, float, float, int, float]], nc: int) -> np.ndarray:
    """Head output (1, 4 + nc, N) from (cx, cy, w, h, cls, score) rows."""
    raw = np.zeros((1, 4 + nc, len(rows)), np.float32)
    for j, (cx, cy, w, h, c, s) in enumerate(rows):
        raw[0, :4, j] = cx, cy, w, h
        raw[0, 4 + c, j] = s
    return raw


def test_decode_yolo_threshold_class_aware_nms_and_filter():
    raw = make_raw(
        [
            (50, 50, 20, 20, 0, 0.9),
            (51, 50, 20, 20, 0, 0.8),  # duplicate of the first: suppressed
            (50, 50, 20, 20, 2, 0.7),  # same place, other class: kept (class-aware NMS)
            (300, 300, 20, 20, 1, 0.05),  # below conf
        ],
        nc=3,
    )
    out = decode_yolo(raw, conf=0.25, iou=0.7, max_det=300)
    assert out.cls.tolist() == [0, 2]
    np.testing.assert_allclose(out.boxes[0], [40, 40, 60, 60])
    assert decode_yolo(raw, 0.25, 0.7, 300, classes=[2]).cls.tolist() == [2]
    assert decode_yolo(raw, 0.25, 0.7, max_det=1).cls.tolist() == [0]
    assert len(decode_yolo(raw, conf=0.95, iou=0.7, max_det=300).boxes) == 0


def test_exported_backend_shares_pre_and_post_processing():
    """An M6 backend only implements infer(); the box must come back in original pixels."""

    class FakeExported(ExportedBackend):
        def infer(self, blob: np.ndarray) -> np.ndarray:
            assert blob.shape == (1, 3, 640, 640)
            # original box (40, 30, 120, 90) on 320x180 -> letterbox (80, 200, 240, 320)
            return make_raw([(160, 260, 160, 120, 3, 0.8)], nc=9)

    det = Detector(CFG, backend=FakeExported(CFG))
    out = det.detect(fixture_frame())
    assert len(out) == 1 and out[0].cls_name == "person"
    np.testing.assert_allclose(out[0].bbox_xyxy, [40, 30, 120, 90], atol=1e-4)
