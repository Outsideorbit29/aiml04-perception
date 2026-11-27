"""Imbalanced-classification metrics in plain numpy (no sklearn needed at serving or in CI).

- average_precision: area under the precision-recall curve (step-wise, as sklearn computes it)
- roc_auc: rank-based (Mann-Whitney), ties averaged
- threshold_at_fpr / recall_at_fpr: the alert threshold that keeps the false-positive rate within
  a budget (e.g. 1% of legit transactions flagged) and the recall it buys
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype="float64")
    if y.shape != s.shape:
        raise ValueError("labels and scores differ in shape")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    return y, s


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y, s = _check(y, s)
    pos = y.sum()
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]  # end of each tie block
    tps, fps = tps[last], fps[last]
    precision = tps / (tps + fps)
    recall = tps / pos
    prev_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_recall) * precision))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y, s = _check(y, s)
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, max_fpr: float) -> float:
    """Threshold t such that flagging `s > t` keeps the false-positive rate <= max_fpr."""
    y, s = _check(y, s)
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0:
        return float("-inf")
    k = int(np.floor(max_fpr * len(neg)))
    return float(neg[min(k, len(neg) - 1)])


def confusion_at(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, float]:
    """Counts and rates when flagging `s > threshold`."""
    y, s = _check(y, s)
    flag = s > threshold
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def recall_at_fpr(y: np.ndarray, s: np.ndarray, max_fpr: float) -> float:
    return confusion_at(y, s, threshold_at_fpr(y, s, max_fpr))["recall"]


def evaluate(y: np.ndarray, s: np.ndarray, threshold: float, prefix: str) -> dict[str, float]:
    """PR-AUC, ROC-AUC and the confusion numbers at `threshold`, keys prefixed (e.g. 'val_')."""
    c = confusion_at(y, s, threshold)
    out = {
        "pr_auc": average_precision(y, s),
        "roc_auc": roc_auc(y, s),
        "precision": c["precision"],
        "recall": c["recall"],
        "fpr": c["fpr"],
        "alerts": c["tp"] + c["fp"],
        "frauds": c["tp"] + c["fn"],
    }
    return {f"{prefix}{k}": float(v) for k, v in out.items()}
