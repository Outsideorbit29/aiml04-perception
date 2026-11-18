"""Pure numpy helpers for M6 parity and benchmark numbers (unit-tested, no torch / GPU).

- ``parity``: max / mean abs diff between a reference output (PyTorch FP32) and a backend's.
- ``latency_stats``: p50 / p95 / mean latency and FPS from per-iteration milliseconds.
- ``match_predictions`` + ``ap_per_class`` + ``detection_map``: mAP@0.5 and mAP@0.5:0.95 with
  the same matching and 101-point interpolated AP as Ultralytics ``model.val()``, so the
  accuracy delta between backends is measured with one harness on identical inputs.
- ``abs_rel`` and ``mask_iou``: depth and lane agreement of a backend with the PyTorch output.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from perception.detection.processing import box_iou

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def parity(ref: np.ndarray, out: np.ndarray) -> dict[str, float]:
    """Max / mean absolute difference, and the max relative to the reference's value range."""
    ref = np.asarray(ref, np.float64)
    out = np.asarray(out, np.float64)
    if ref.shape != out.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs output {out.shape}")
    diff = np.abs(ref - out)
    rng = float(ref.max() - ref.min()) if ref.size else 0.0
    max_abs = float(diff.max()) if diff.size else 0.0
    return {
        "max_abs": max_abs,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "ref_range": rng,
        "max_abs_rel_range": max_abs / rng if rng > 0 else 0.0,
    }


def merge_parity(items: Sequence[dict[str, float]]) -> dict[str, float]:
    """Combine per-frame ``parity`` dicts: worst max, mean of means, widest range."""
    if not items:
        return {}
    rng = max(i["ref_range"] for i in items)
    max_abs = max(i["max_abs"] for i in items)
    return {
        "max_abs": max_abs,
        "mean_abs": float(np.mean([i["mean_abs"] for i in items])),
        "ref_range": rng,
        "max_abs_rel_range": max_abs / rng if rng > 0 else 0.0,
        "frames": len(items),
    }


def latency_stats(ms: Sequence[float]) -> dict[str, float]:
    a = np.asarray(ms, np.float64)
    if not a.size:
        return {}
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "fps": float(1000.0 / a.mean()),
        "iters": int(a.size),
    }


def match_predictions(
    pred_boxes: np.ndarray,
    pred_cls: np.ndarray,
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    iouv: np.ndarray = IOU_THRESHOLDS,
) -> np.ndarray:
    """(N, T) bool: prediction i is a true positive at IoU threshold t.

    Same rule as Ultralytics ``DetectionValidator.match_predictions``: class must match; at each
    threshold, candidate pairs are sorted by IoU and each prediction and each GT is used once.
    """
    n = len(pred_boxes)
    correct = np.zeros((n, len(iouv)), bool)
    if n == 0 or len(gt_boxes) == 0:
        return correct
    iou = box_iou(gt_boxes, pred_boxes)  # (M, N)
    iou = iou * (np.asarray(gt_cls)[:, None] == np.asarray(pred_cls)[None, :])
    for t, thr in enumerate(iouv):
        gi, pi = np.nonzero(iou >= thr)
        if not len(gi):
            continue
        m = np.stack([gi, pi, iou[gi, pi]], axis=1)
        m = m[np.argsort(-m[:, 2], kind="stable")]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        m = m[np.unique(m[:, 0], return_index=True)[1]]
        correct[m[:, 1].astype(int), t] = True
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP (COCO / Ultralytics) of one precision-recall curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz
    return float(trapezoid(np.interp(x, mrec, mpre), x))


def ap_per_class(
    tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, gt_cls: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """AP (C, T) for every class C that has ground truth, and those class ids."""
    tp = np.asarray(tp, bool).reshape(len(conf), -1)
    order = np.argsort(-np.asarray(conf), kind="stable")
    tp, pred_cls = tp[order], np.asarray(pred_cls)[order]
    classes, n_gt = np.unique(np.asarray(gt_cls), return_counts=True)
    ap = np.zeros((len(classes), tp.shape[1]))
    for ci, (c, n_l) in enumerate(zip(classes, n_gt, strict=True)):
        sel = pred_cls == c
        if not sel.any():
            continue
        tpc = np.cumsum(tp[sel], axis=0)
        fpc = np.cumsum(~tp[sel], axis=0)
        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc)
        for t in range(tp.shape[1]):
            ap[ci, t] = compute_ap(recall[:, t], precision[:, t])
    return ap, classes


class MapAccumulator:
    """Collect per-image predictions and GT, then ``result()`` -> mAP@0.5, mAP@0.5:0.95."""

    def __init__(self) -> None:
        self.tp: list[np.ndarray] = []
        self.conf: list[np.ndarray] = []
        self.pred_cls: list[np.ndarray] = []
        self.gt_cls: list[np.ndarray] = []
        self.images = 0

    def add(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_cls: np.ndarray,
        gt_boxes: np.ndarray,
        gt_cls: np.ndarray,
    ) -> None:
        self.tp.append(match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls))
        self.conf.append(np.asarray(pred_scores, np.float64).reshape(-1))
        self.pred_cls.append(np.asarray(pred_cls, np.int64).reshape(-1))
        self.gt_cls.append(np.asarray(gt_cls, np.int64).reshape(-1))
        self.images += 1

    def result(self) -> dict[str, float]:
        if not self.images:
            return {"map50": 0.0, "map50_95": 0.0, "images": 0}
        tp = np.concatenate(self.tp) if self.tp else np.zeros((0, len(IOU_THRESHOLDS)), bool)
        ap, classes = ap_per_class(
            tp, np.concatenate(self.conf), np.concatenate(self.pred_cls),
            np.concatenate(self.gt_cls),
        )
        if not len(classes):
            return {"map50": 0.0, "map50_95": 0.0, "images": self.images}
        return {
            "map50": float(ap[:, 0].mean()),
            "map50_95": float(ap.mean()),
            "images": self.images,
            "classes": int(len(classes)),
        }


def abs_rel(ref: np.ndarray, out: np.ndarray, eps: float = 1e-6) -> float:
    """Mean |out - ref| / |ref| over pixels where |ref| > eps (depth agreement)."""
    ref = np.asarray(ref, np.float64).ravel()
    out = np.asarray(out, np.float64).ravel()
    ok = np.abs(ref) > eps
    if not ok.any():
        return 0.0
    return float(np.mean(np.abs(out[ok] - ref[ok]) / np.abs(ref[ok])))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks; 1.0 when both are empty (they agree)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0
