"""Pre- and post-processing for YOLO models exported to ONNX / TensorRT (numpy + OpenCV only).

The torch backend lets Ultralytics do these steps internally. Exported backends (M6) must use the
functions here so every backend sees the same letterbox, confidence filter, NMS and box rescale
as Ultralytics' predictor:

    img, ratio, pad = letterbox(frame, imgsz)
    raw = session.run(to_blob(img))              # (1, 4 + nc, N) for YOLO11 detect heads
    boxes, scores, cls = decode_yolo(raw, conf, iou, max_det)
    boxes = scale_boxes(boxes, ratio, pad, frame.shape[:2])
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

PAD_VALUE = 114  # Ultralytics letterbox fill colour
MAX_WH = 7680.0  # class offset for class-aware NMS (same constant as Ultralytics)


class RawDetections(NamedTuple):
    """Backend output in original-image pixels, before conversion to ``Detection`` objects."""

    boxes: np.ndarray  # (N, 4) float32 xyxy
    scores: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) int64


def empty_raw() -> RawDetections:
    return RawDetections(
        np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    )


def as_hw(imgsz: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(imgsz, int):
        return imgsz, imgsz
    h, w = imgsz
    return int(h), int(w)


def letterbox(
    img: np.ndarray, imgsz: int | tuple[int, int] | list[int]
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping the aspect ratio and pad to ``imgsz`` (centred, like Ultralytics LetterBox).

    Returns the padded image, the scale ratio and the (left, top) padding in pixels.
    """
    new_h, new_w = as_hw(imgsz)
    h, w = img.shape[:2]
    r = min(new_h / h, new_w / w)
    unpad_w, unpad_h = int(round(w * r)), int(round(h * r))
    dw, dh = (new_w - unpad_w) / 2, (new_h - unpad_h) / 2
    if (w, h) != (unpad_w, unpad_h):
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    return out, r, (left, top)


def to_blob(img_bgr: np.ndarray) -> np.ndarray:
    """HxWx3 BGR uint8 -> 1x3xHxW RGB float32 in [0, 1], contiguous."""
    rgb = img_bgr[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb[None], dtype=np.float32) / 255.0


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes: (N, 4) x (M, 4) -> (N, M)."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS. Returns kept indices sorted by descending score."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def decode_yolo(
    raw: np.ndarray,
    conf: float,
    iou: float,
    max_det: int,
    classes: list[int] | None = None,
) -> RawDetections:
    """Decode a YOLOv8/11 detect head output ``(1, 4 + nc, N)`` into boxes in letterbox pixels.

    Rows are cx, cy, w, h followed by one score per class (no objectness). NMS is class-aware.
    """
    pred = np.asarray(raw, np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.T  # (N, 4 + nc)
    cls_scores = pred[:, 4:]
    cls = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(pred)), cls]
    keep = scores > conf
    if classes is not None:
        keep &= np.isin(cls, classes)
    pred, cls, scores = pred[keep], cls[keep], scores[keep]
    if not len(pred):
        return empty_raw()
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    idx = nms(boxes + cls[:, None] * MAX_WH, scores, iou)[:max_det]
    return RawDetections(
        boxes[idx].astype(np.float32), scores[idx].astype(np.float32), cls[idx].astype(np.int64)
    )


def scale_boxes(
    boxes: np.ndarray, ratio: float, pad: tuple[int, int], orig_hw: tuple[int, int]
) -> np.ndarray:
    """Map xyxy boxes from letterbox pixels back to the original image and clip to it."""
    out = np.asarray(boxes, np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / ratio
    h, w = orig_hw
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out
