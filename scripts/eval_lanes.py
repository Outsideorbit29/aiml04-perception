"""M3.3 + M3.4: classical vs YOLOP lanes on BDD100K val. Runs on Kaggle (kaggle/eval_lanes).

    python -m scripts.eval_lanes --config configs/lanes_eval.yaml            # 500 images
    python -m scripts.eval_lanes --config configs/lanes_eval.yaml --smoke    # 30 images

Data: BDD100K val images plus the 2018 release label file, which holds per-image attributes
(timeofday, weather, scene), ``lane`` polylines and ``drivable area`` polygons. 500 images are
sampled with a fixed seed, stratified by timeofday (quotas in the config). Label coordinates are
in ``data.label_wh`` pixels (1280x720) and are rescaled if an image is loaded at another size.

Both detectors use the BDD camera geometry (``lanes_config`` = configs/lanes_bdd.yaml) and run
with ``detect(frame, stateless=True)``: each image is independent, so no smoothing, no
``every_n_frames`` reuse and no state carried between images.

Metrics (all scored inside the row band ``metrics.band_frac``, which must equal the warp source
rows of ``lanes_config``, i.e. the part of the image both methods model; checked at start-up):

* **Ego-lane IoU.** Prediction: ``LaneResult.ego_mask`` (region between the fitted left and
  right boundaries). With ``single_side_fallback`` a mask is also built when only one side is
  measured (the other is synthesized at a nominal lane width, see perception/lanes/base.py);
  those images are counted as ``synth``. ``iou_measured`` is the old behaviour: a mask that
  used a synthesized side counts as no prediction (IoU 0). Ground truth: union of the BDD
  ``drivable area`` polygons with ``areaType == "direct"`` (the ego vehicle's own lane). Per
  image IoU = |P & G| / |P | G| over the band rows. Images whose GT is empty inside the band are
  excluded from IoU (counted as ``gt_empty``; a prediction there is counted as ``fp_mask``).
  Images with GT but no prediction score IoU 0 (counted as ``pred_none``). Reported as the mean
  of per-image IoU, the micro IoU (sum of intersections / sum of unions) and the mean over
  images that have a prediction (``iou_when_pred``, separates "no fit" from "wrong fit").
  Note: BDD "direct" stops at the vehicle ahead, the predicted mask does not.
* **Lane-line F1.** GT boundaries: among BDD ``lane`` polylines with
  ``laneDirection == "parallel"`` (excluding ``metrics.exclude_lane_types``) that reach below
  ``gt_min_bottom_frac``, rank them by their x at row ``gt_ref_frac`` (linear extrapolation
  below the last vertex). The GT left boundary is the closest one left of the image centre, the
  GT right boundary the closest one right of it (so each image has 0-2 GT boundaries). A GT
  boundary is *scorable* only if it covers at least ``min_gt_rows`` of the ``band_rows`` sampled
  rows; a side whose GT boundary exists but is shorter is *ignored* (no TP/FP/FN, counted as
  ``gt_short``): the marking is there but cannot be scored in the band. Only *measured* sides
  are predictions (``LaneResult.left/right``; synthesized fallback sides never count).
  For each side: predicted and GT -> distance = mean |x_pred(y) - x_gt(y)| over the GT-covered
  sampled rows; distance <= ``lane_tol_px`` is a TP, otherwise one FP and one FN. Prediction
  without any GT boundary -> FP; GT without prediction -> FN. F1 = 2TP / (2TP + FP + FN), pooled
  over sides and images.
* **Speed.** Per image after ``fps.warmup_runs`` untimed calls on the first image: ``total_ms``
  = wall time of ``detect`` (pre/post-processing and polyline fit included, disk I/O excluded),
  split into ``binary_ms`` (backend stage), ``fit_ms`` (shared warp + sliding window + fit) and
  for YOLOP ``model_ms`` (``session.run`` only). Classical runs on the CPU (OpenCV), YOLOP on
  the GPU (onnxruntime CUDA provider), same Kaggle session. YOLOP also gets a model-only
  micro-benchmark on a fixed tensor (``model_bench``: plain run and CUDA I/O binding).

Breakdowns: overall, timeofday, weather and geometry (``curve`` when a scorable GT boundary
deviates more than ``curve_px`` from its chord inside the band, ``straight`` otherwise, ``no_gt``
without a scorable GT boundary; ``gt_sides`` counts scorable boundaries with the same rule).
Diagnostics per group: ``gt_empty``, ``pred_none``, ``fp_mask``, ``synth``, ``gt_short`` and the
GT geometry
(median vanishing row, lane width, drivable-area extent) used to set the warp. Outputs in
``out_dir``: results.json, per_image.jsonl, lanes_comparison.md, lanes_failures.jpg and, with
``debug.enabled`` (always on with --smoke), ``debug/*.jpg`` overlays; the same go to W&B.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import platform
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from perception.lanes.base import (
    build_lane_detector,
    load_lane_config,
    merge_config,
    resolve_path,
)
from perception.types import LaneResult, Polyline

GROUPS = ("all", "timeofday", "weather", "geometry")
STAGES = ("binary_ms", "model_ms", "fit_ms")

Scale = tuple[float, float]


# ---------------------------------------------------------------- BDD label helpers (pure)


def flatten_poly2d(
    vertices: Sequence[Sequence[float]], types: str | None, closed: bool, steps: int
) -> np.ndarray:
    """BDD poly2d -> (N, 2) float (x, y) points. ``types`` marks each vertex as L (on the curve)
    or C (cubic Bezier control point); each ``L C C L`` run is sampled with ``steps`` points."""
    pts = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return pts
    t = (types or "").ljust(pts.shape[0], "L")[: pts.shape[0]]
    if closed:
        pts = np.vstack([pts, pts[:1]])
        t = t + "L"
    out = [pts[0]]
    i = 1
    n = pts.shape[0]
    while i < n:
        if t[i] == "C" and i + 2 < n and t[i + 1] == "C":
            p0, c1, c2, p3 = out[-1], pts[i], pts[i + 1], pts[i + 2]
            s = np.linspace(0.0, 1.0, steps + 1)[1:, None]
            bez = (
                (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * c1 + 3 * (1 - s) * s**2 * c2 + s**3 * p3
            )
            out.extend(bez)
            i += 3
        else:
            out.append(pts[i])
            i += 1
    res = np.asarray(out)
    return res[:-1] if closed and res.shape[0] > 1 else res


def _poly2d_entries(label: dict) -> list[dict]:
    poly = label.get("poly2d") or []
    return poly if isinstance(poly, list) else [poly]


def label_scale(hw: tuple[int, int], label_wh: Sequence[float] | None) -> Scale:
    """(sx, sy) from label pixel coordinates to the loaded image (1, 1 when sizes match)."""
    if not label_wh:
        return 1.0, 1.0
    return hw[1] / float(label_wh[0]), hw[0] / float(label_wh[1])


def direct_area_mask(
    frame: dict, hw: tuple[int, int], steps: int, scale: Scale = (1.0, 1.0)
) -> np.ndarray:
    """HxW bool mask of the BDD ``drivable area`` polygons with areaType ``direct``."""
    mask = np.zeros(hw, dtype=np.uint8)
    for lab in frame.get("labels") or []:
        if lab.get("category") != "drivable area":
            continue
        if (lab.get("attributes") or {}).get("areaType") != "direct":
            continue
        for p in _poly2d_entries(lab):
            pts = flatten_poly2d(p.get("vertices", []), p.get("types"), True, steps)
            if pts.shape[0] >= 3:
                pts = pts * np.asarray(scale)
                cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def lane_polylines(
    frame: dict, steps: int, exclude_types: Iterable[str], scale: Scale = (1.0, 1.0)
) -> list[np.ndarray]:
    """Parallel BDD lane polylines (points sorted by y) excluding ``exclude_types``."""
    excl = set(exclude_types)
    out = []
    for lab in frame.get("labels") or []:
        if lab.get("category") != "lane":
            continue
        attrs = lab.get("attributes") or {}
        if attrs.get("laneDirection") != "parallel" or attrs.get("laneType") in excl:
            continue
        for p in _poly2d_entries(lab):
            pts = flatten_poly2d(p.get("vertices", []), p.get("types"), False, steps)
            if pts.shape[0] >= 2:
                pts = pts * np.asarray(scale)
                out.append(pts[np.argsort(pts[:, 1], kind="stable")])
    return out


def x_on_rows(pts: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rows inside the polyline's y range, x interpolated on them). ``pts`` sorted by y."""
    ys, xs = pts[:, 1], pts[:, 0]
    inside = (rows >= ys[0]) & (rows <= ys[-1])
    return rows[inside], np.interp(rows[inside], ys, xs)


def x_at_ref(pts: np.ndarray, y: float) -> float | None:
    """x of a y-sorted polyline at row ``y``; linear extrapolation below its lowest vertex."""
    ys, xs = pts[:, 1], pts[:, 0]
    if ys[0] <= y <= ys[-1]:
        return float(np.interp(y, ys, xs))
    if y < ys[0]:
        return None
    lower = np.flatnonzero(ys < ys[-1] - 1.0)
    if lower.size == 0:
        return None
    j = lower[-1]
    slope = (xs[-1] - xs[j]) / (ys[-1] - ys[j])
    return float(xs[-1] + slope * (y - ys[-1]))


def ego_boundaries(
    lanes: list[np.ndarray], hw: tuple[int, int], ref_frac: float, min_bottom_frac: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """GT (left, right) ego-lane boundaries: the lane lines nearest the image centre on each side
    at row ``ref_frac * (H-1)``, among lines reaching below ``min_bottom_frac * (H-1)``."""
    h, w = hw
    cx, y_ref, y_min = (w - 1) / 2.0, ref_frac * (h - 1), min_bottom_frac * (h - 1)
    left: tuple[float, np.ndarray] | None = None
    right: tuple[float, np.ndarray] | None = None
    for pts in lanes:
        if pts[-1, 1] < y_min:
            continue
        x = x_at_ref(pts, y_ref)
        if x is None:
            continue
        if x < cx and (left is None or x > left[0]):
            left = (x, pts)
        elif x >= cx and (right is None or x < right[0]):
            right = (x, pts)
    return (left[1] if left else None), (right[1] if right else None)


def band_rows(hw: tuple[int, int], band_frac: Sequence[float], n: int) -> np.ndarray:
    h = hw[0]
    return np.linspace(band_frac[0] * (h - 1), band_frac[1] * (h - 1), int(n))


def check_band_matches_warp(band_frac: Sequence[float], lcfg: dict, tol: float = 1e-6) -> None:
    """The scoring band must be the rows the detectors model (warp source top/bottom)."""
    ys = [float(p[1]) for p in lcfg["fitting"]["warp"]["src"]]
    if abs(min(ys) - float(band_frac[0])) > tol or abs(max(ys) - float(band_frac[1])) > tol:
        raise ValueError(
            f"metrics.band_frac {list(band_frac)} != warp src rows [{min(ys)}, {max(ys)}] of the "
            "lanes config; GT and predictions would be scored on different rows"
        )


def scorable(pts: np.ndarray | None, rows: np.ndarray, min_rows: int) -> bool:
    """A GT boundary is scored only if it covers at least ``min_rows`` of the sampled rows."""
    return pts is not None and x_on_rows(pts, rows)[0].size >= max(int(min_rows), 1)


def geometry_of(
    gt_sides: Sequence[np.ndarray | None], rows: np.ndarray, curve_px: float, min_rows: int = 3
) -> str:
    """``curve`` if any scorable GT boundary (>= ``min_rows`` band rows, at least 3) deviates
    > curve_px from its chord inside the band; ``no_gt`` if no boundary is scorable."""
    seen = False
    for pts in gt_sides:
        if pts is None:
            continue
        r, x = x_on_rows(pts, rows)
        if r.size < max(int(min_rows), 3):
            continue
        seen = True
        chord = x[0] + (x[-1] - x[0]) * (r - r[0]) / (r[-1] - r[0])
        if np.abs(x - chord).max() > curve_px:
            return "curve"
    return "straight" if seen else "no_gt"


def gt_geometry(
    gt_mask: np.ndarray,
    sides: Sequence[np.ndarray | None],
    rows: np.ndarray,
) -> dict[str, float | None]:
    """Camera-geometry facts from the GT, as fractions of H (rows) or W (widths):
    ``direct_top``/``direct_bottom`` rows of the GT direct area, ``vp_y`` the row where straight
    fits of the two GT boundaries (over the band) meet, ``width_bottom`` the GT lane width at the
    band's bottom row. Used to check the warp of configs/lanes_bdd.yaml."""
    h, w = gt_mask.shape
    out: dict[str, float | None] = {
        "direct_top": None,
        "direct_bottom": None,
        "vp_y": None,
        "width_bottom": None,
    }
    any_rows = np.flatnonzero(gt_mask.any(axis=1))
    if any_rows.size:
        out["direct_top"] = float(any_rows[0] / (h - 1))
        out["direct_bottom"] = float(any_rows[-1] / (h - 1))
    if sides[0] is None or sides[1] is None:
        return out
    fits = []
    for pts in sides:
        r, x = x_on_rows(pts, rows)
        if r.size < 3:
            return out
        fits.append(np.polyfit(r, x, 1))  # x = a*y + b
    (a1, b1), (a2, b2) = fits
    if abs(a1 - a2) > 1e-9:
        out["vp_y"] = float((b2 - b1) / (a1 - a2) / (h - 1))
    y_b = float(rows[-1])
    out["width_bottom"] = float((np.polyval(fits[1], y_b) - np.polyval(fits[0], y_b)) / w)
    return out


@dataclass
class SideScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    err_px: float | None = None
    ignored: int = 0  # GT boundary exists but covers < min_rows band rows: not scored


def score_side(
    pred: Polyline | None, gt: np.ndarray | None, rows: np.ndarray, tol_px: float, min_rows: int
) -> SideScore:
    """Score one boundary (definition in the module docstring)."""
    if gt is None:
        return SideScore(fp=int(pred is not None))
    gt_rows, gt_x = x_on_rows(gt, rows)
    if gt_rows.size < min_rows:
        return SideScore(ignored=1)
    if pred is None:
        return SideScore(fn=1)
    err = float(np.mean(np.abs(np.asarray(pred.x_at(gt_rows)) - gt_x)))
    if err <= tol_px:
        return SideScore(tp=1, err_px=err)
    return SideScore(fp=1, fn=1, err_px=err)


def band_slice(h: int, band_frac: Sequence[float]) -> slice:
    y0 = max(int(np.ceil(band_frac[0] * (h - 1))), 0)
    y1 = min(int(np.floor(band_frac[1] * (h - 1))), h - 1) + 1
    return slice(y0, y1)


def band_iou_counts(
    pred: np.ndarray | None, gt: np.ndarray, band_frac: Sequence[float]
) -> tuple[int, int]:
    """(intersection, union) pixel counts inside the row band."""
    sl = band_slice(gt.shape[0], band_frac)
    g = gt[sl]
    p = pred[sl] if pred is not None else np.zeros_like(g)
    return int(np.count_nonzero(p & g)), int(np.count_nonzero(p | g))


def frame_attrs(frame: dict) -> tuple[str, str, str]:
    a = frame.get("attributes") or {}
    return (
        a.get("timeofday") or "undefined",
        a.get("weather") or "undefined",
        a.get("scene") or "undefined",
    )


def has_lane_labels(frame: dict) -> bool:
    return any(
        lab.get("category") in ("lane", "drivable area") for lab in frame.get("labels") or []
    )


def stratified_sample(
    frames: Sequence[dict], quotas: dict[str, int], seed: int, require_labels: bool
) -> list[dict]:
    """``quotas[tod]`` frames per timeofday, drawn with ``random.Random(seed)`` from the frames
    sorted by name (so the draw is independent of the label-file order)."""
    rng = random.Random(seed)
    by_tod: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(frames, key=lambda f: f["name"]):
        if require_labels and not has_lane_labels(f):
            continue
        by_tod[frame_attrs(f)[0]].append(f)
    out: list[dict] = []
    for tod, n in quotas.items():
        pool = by_tod.get(tod, [])
        out.extend(rng.sample(pool, min(int(n), len(pool))))
    return out


# ---------------------------------------------------------------- aggregation (pure)


@dataclass
class Tally:
    n: int = 0
    iou_sum: float = 0.0
    iou_n: int = 0
    iou_pred_sum: float = 0.0
    iou_pred_n: int = 0
    inter: int = 0
    union: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    errs: list[float] = field(default_factory=list)
    both: int = 0
    gt_empty: int = 0
    pred_none: int = 0
    fp_mask: int = 0
    gt_sides: int = 0
    gt_short: int = 0
    synth: int = 0
    iou_meas_sum: float = 0.0

    def add(self, m: dict) -> None:
        """``m``: one method's per-image record (see ``evaluate_image``)."""
        self.n += 1
        self.pred_none += int(m["pred_none"])
        self.synth += int(m.get("synth", False))
        if m["gt_px"] > 0:
            iou = m["inter"] / m["union"]
            self.iou_meas_sum += m.get("inter_meas", m["inter"]) / m.get("union_meas", m["union"])
            self.inter += m["inter"]
            self.union += m["union"]
            self.iou_sum += iou
            self.iou_n += 1
            if not m["pred_none"]:
                self.iou_pred_sum += iou
                self.iou_pred_n += 1
        else:
            self.gt_empty += 1
            self.fp_mask += int(not m["pred_none"])
        self.tp += m["tp"]
        self.fp += m["fp"]
        self.fn += m["fn"]
        self.errs.extend(m["errs"])
        self.both += int(m["both"])
        self.gt_sides += int(m.get("gt_sides", 0))
        self.gt_short += int(m.get("gt_short", 0))

    def row(self) -> dict:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else None
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else None
        denom = 2 * self.tp + self.fp + self.fn
        return {
            "n": self.n,
            "iou_mean": self.iou_sum / self.iou_n if self.iou_n else None,
            "iou_micro": self.inter / self.union if self.union else None,
            "iou_when_pred": self.iou_pred_sum / self.iou_pred_n if self.iou_pred_n else None,
            "iou_measured": self.iou_meas_sum / self.iou_n if self.iou_n else None,
            "lane_f1": 2 * self.tp / denom if denom else None,
            "lane_precision": p,
            "lane_recall": r,
            "lane_err_px": float(np.mean(self.errs)) if self.errs else None,
            "both_sides_rate": self.both / self.n if self.n else None,
            "iou_images": self.iou_n,
            "gt_empty": self.gt_empty,
            "pred_none": self.pred_none,
            "fp_mask": self.fp_mask,
            "gt_sides": self.gt_sides,
            "gt_short": self.gt_short,
            "synth": self.synth,
        }


def aggregate(records: Sequence[dict], methods: Sequence[str]) -> list[dict]:
    """One row per (method, group, key) from per-image records."""
    tallies: dict[tuple[str, str, str], Tally] = defaultdict(Tally)
    for rec in records:
        keys = {"all": "all", **{g: rec[g] for g in GROUPS if g != "all"}}
        for m in methods:
            for g, k in keys.items():
                extra = {key: rec.get(key, 0) for key in ("gt_sides", "gt_short")}
                tallies[(m, g, k)].add({**rec[m], **extra})
    order = {g: i for i, g in enumerate(GROUPS)}
    rows = [{"method": m, "group": g, "key": k, **t.row()} for (m, g, k), t in tallies.items()]
    return sorted(rows, key=lambda r: (order[r["group"]], r["key"], methods.index(r["method"])))


def gt_geometry_summary(records: Sequence[dict]) -> dict[str, dict]:
    """Median / p10 / p90 / count of each ``gt_geometry`` field over the images that have it."""
    out: dict[str, dict] = {}
    for key in ("direct_top", "direct_bottom", "vp_y", "width_bottom"):
        vals = np.asarray(
            [r["gt_geom"][key] for r in records if r.get("gt_geom", {}).get(key) is not None]
        )
        if vals.size == 0:
            out[key] = {"n": 0}
            continue
        out[key] = {
            "n": int(vals.size),
            "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90)),
        }
    return out


def timing_stats(ms: Sequence[float], warmup: int = 0) -> dict:
    arr = np.asarray(ms[warmup:] if len(ms) > warmup else ms, dtype=np.float64)
    if arr.size == 0:
        return {"frames": 0}
    mean = float(arr.mean())
    return {
        "frames": int(arr.size),
        "mean_ms": mean,
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "fps": 1000.0 / mean if mean > 0 else None,
    }


def _fmt(v: Any, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def markdown_table(
    rows: Sequence[dict], methods: Sequence[str], fps: dict[str, dict], notes: Sequence[str]
) -> str:
    """Comparison table (IoU and F1 side by side), a diagnostics table and a speed table."""
    by_key: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_key[(r["group"], r["key"])][r["method"]] = r
    head = ["Group", "Subset", "N"]
    head += [f"IoU {m}" for m in methods] + [f"IoU measured-only {m}" for m in methods]
    head += [f"F1 {m}" for m in methods] + [f"P/R {m}" for m in methods]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (g, k), per in by_key.items():
        n = next(iter(per.values()))["n"]
        cells = [g, k, str(n)]
        cells += [_fmt(per.get(m, {}).get("iou_mean")) for m in methods]
        cells += [_fmt(per.get(m, {}).get("iou_measured")) for m in methods]
        cells += [_fmt(per.get(m, {}).get("lane_f1")) for m in methods]
        cells += [
            f"{_fmt(per.get(m, {}).get('lane_precision'), 2)}/"
            f"{_fmt(per.get(m, {}).get('lane_recall'), 2)}"
            for m in methods
        ]
        lines.append("| " + " | ".join(cells) + " |")

    head = ["Group", "Subset", "N", "GT empty", "GT sides", "GT short"]
    head += [f"no pred {m}" for m in methods] + [f"synth {m}" for m in methods]
    head += [f"IoU when pred {m}" for m in methods]
    head += [f"both sides {m}" for m in methods]
    lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (g, k), per in by_key.items():
        first = next(iter(per.values()))
        cells = [g, k, str(first["n"]), str(first.get("gt_empty", "-"))]
        cells += [str(first.get("gt_sides", "-")), str(first.get("gt_short", "-"))]
        cells += [str(per.get(m, {}).get("pred_none", "-")) for m in methods]
        cells += [str(per.get(m, {}).get("synth", "-")) for m in methods]
        cells += [_fmt(per.get(m, {}).get("iou_when_pred")) for m in methods]
        cells += [_fmt(per.get(m, {}).get("both_sides_rate"), 2) for m in methods]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "| Method | Device | Frames | Total ms (mean) | p50 ms | p90 ms | FPS "
        "| Backend ms | Model ms | Fit ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in methods:
        s = fps.get(m, {})
        cells = [m, s.get("device", "-"), str(s.get("frames", 0))]
        cells += [_fmt(s.get(k), 1) for k in ("mean_ms", "p50_ms", "p90_ms", "fps")]
        cells += [_fmt(s.get(f"{k[:-3]}_mean_ms"), 1) for k in STAGES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines + [""] + [f"- {n}" for n in notes]) + "\n"


# ---------------------------------------------------------------- Kaggle-side helpers


def find_input(primary: str, root: str, depth: int, tail_parts: int) -> Path:
    """``primary`` if it exists, else the first match of its last ``tail_parts`` path parts at
    depth 1..``depth`` under ``root`` (shallow globs, no recursive walk)."""
    p = Path(primary)
    if p.exists():
        return p
    tail = Path(*p.parts[-tail_parts:])
    for d in range(0, depth + 1):
        hits = sorted(glob.glob(str(Path(root, *(["*"] * d), tail))))
        if hits:
            print(f"{primary} missing; using {hits[0]}", flush=True)
            return Path(hits[0])
    raise FileNotFoundError(f"{primary} not found, and no {tail} under {root} (depth {depth})")


def load_frames(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["frames"] if isinstance(data, dict) and "frames" in data else data


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "cpu"


def gpu_name() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "gpu?"


def preload_cuda_libs() -> None:
    """Let onnxruntime-gpu find CUDA/cuDNN on Kaggle: torch (preinstalled there) loads them into
    the process, and onnxruntime>=1.21 can preload them itself. Eval-only; the detector never
    imports torch."""
    try:
        import torch  # noqa: F401
    except Exception as exc:
        print(f"torch not importable ({type(exc).__name__}); relying on system CUDA libs")
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception as exc:
        print(f"onnxruntime preload skipped ({type(exc).__name__}: {exc})")


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    scale = max(0.4, img.shape[1] / 900)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(out, (0, 0), (tw + 8, th + base + 8), (0, 0, 0), -1)
    cv2.putText(
        out, text, (4, th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA
    )
    return out


def _tint(img: np.ndarray, mask: np.ndarray | None, color: Sequence[int], alpha: float) -> None:
    if mask is None or not mask.any():
        return
    tint = np.zeros_like(img)
    tint[mask] = color
    img[mask] = cv2.addWeighted(img, 1.0 - alpha, tint, alpha, 0)[mask]


def _gt_lines(img: np.ndarray, sides: Sequence[np.ndarray | None], color, thick: int) -> None:
    for pts in sides:
        if pts is not None:
            cv2.polylines(
                img,
                [np.round(pts).astype(np.int32).reshape(-1, 1, 2)],
                False,
                tuple(int(c) for c in color),
                thick,
            )


def _band_lines(img: np.ndarray, band_frac: Sequence[float], color) -> None:
    h, w = img.shape[:2]
    for f in band_frac:
        y = int(round(f * (h - 1)))
        cv2.line(img, (0, y), (w - 1, y), tuple(int(c) for c in color), 1)


def draw_gt(
    img: np.ndarray, gt_mask: np.ndarray, sides: Sequence[np.ndarray | None], fig: dict
) -> np.ndarray:
    out = img.copy()
    _tint(out, gt_mask, fig["gt_mask_color"], 0.4)
    _gt_lines(out, sides, fig["gt_color"], 3)
    return out


def debug_tiles(
    img: np.ndarray,
    rec: dict,
    gt: dict,
    vis: dict[str, dict],
    lcfg: dict,
    mcfg: dict,
    fig: dict,
) -> np.ndarray:
    """One row of side-by-side tiles: GT, then per method the lane binary (cyan), predicted ego
    mask and boundaries, with the GT boundaries (yellow, thin) and the scoring band on top."""
    from perception.lanes.render import draw_lanes

    band = mcfg["band_frac"]
    g = draw_gt(img, gt["mask"], gt["sides"], fig)
    _band_lines(g, band, fig["band_color"])
    gpx = rec[next(iter(vis))]["gt_px"] if vis else 0
    tiles = [
        _label(
            g,
            f"GT {rec['timeofday']}/{rec['weather']}/{rec['geometry']} "
            f"band px {gpx} sides {rec['gt_sides']}",
        )
    ]
    for m, v in vis.items():
        out = img.copy()
        _tint(out, v["binary"], fig["binary_color"], 0.7)
        out = draw_lanes(
            out, v["result"], v["y_top"] or 0.0, lcfg["viz"], v["y_bottom"], v.get("synth_line")
        )
        _gt_lines(out, gt["sides"], fig["gt_color"], 1)
        _band_lines(out, band, fig["band_color"])
        r = rec[m]
        iou = r["inter"] / r["union"] if r["gt_px"] > 0 else float("nan")
        tiles.append(
            _label(
                out,
                f"{m} IoU {iou:.2f} {'NO MASK ' if r['pred_none'] else ''}"
                f"{'SYN-' + r['synth_side'][0].upper() + ' ' if r.get('synth_side') else ''}"
                f"TP{r['tp']} FP{r['fp']} FN{r['fn']}",
            )
        )
    tw = int(fig["tile_width"])
    th = int(round(img.shape[0] * tw / img.shape[1]))
    return np.hstack([cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA) for t in tiles])


def pick_failures(records: Sequence[dict], methods: Sequence[str], per_method: int) -> list[int]:
    """Indices of the lowest-IoU images per method (GT present in the band), deduplicated."""
    chosen: list[int] = []
    for m in methods:
        cand = [i for i, r in enumerate(records) if r[m]["gt_px"] > 0 and i not in chosen]
        cand.sort(key=lambda i: records[i][m]["inter"] / records[i][m]["union"])
        chosen.extend(cand[:per_method])
    return chosen


def spread_indices(n: int, k: int) -> list[int]:
    """``k`` evenly spaced indices into ``range(n)`` (the sample is grouped by timeofday, so this
    covers every group)."""
    if n <= 0 or k <= 0:
        return []
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, min(k, n))})


# ---------------------------------------------------------------- main


def measured_sides(
    res: LaneResult, synthesized_side: str | None
) -> tuple[Polyline | None, Polyline | None, str | None]:
    """(left, right, synthesized side name or None) with any synthesized side removed.

    The lane detectors never put a synthesized fallback side into ``LaneResult.left/right``, so
    a mask with exactly one side ``None`` is a one-sided (fallback) mask. A side the detector
    reports as synthesized (``last_debug["synthesized_side"]``) is dropped here as well, so F1
    stays measured-only even if a backend did return it.
    """
    left, right = res.left, res.right
    syn = synthesized_side if res.ego_mask is not None else None
    if syn is None and res.ego_mask is not None and (left is None) != (right is None):
        syn = "right" if left is not None else "left"
    if syn == "left":
        left = None
    elif syn == "right":
        right = None
    return left, right, syn


def evaluate_image(
    detectors: dict,
    img: np.ndarray,
    frame: dict,
    mcfg: dict,
    timings: dict,
    label_wh: Sequence[float] | None = None,
    keep_vis: bool = False,
) -> tuple[dict, dict[str, dict], dict]:
    """Score every detector on one image. Returns (record, per-method visual data if
    ``keep_vis`` else {}, GT dict with ``mask`` and ``sides``)."""
    hw = img.shape[:2]
    steps = int(mcfg["bezier_steps"])
    scale = label_scale(hw, label_wh)
    gt_mask = direct_area_mask(frame, hw, steps, scale)
    lanes = lane_polylines(frame, steps, mcfg["exclude_lane_types"], scale)
    gl, gr = ego_boundaries(
        lanes, hw, float(mcfg["gt_ref_frac"]), float(mcfg["gt_min_bottom_frac"])
    )
    rows = band_rows(hw, mcfg["band_frac"], mcfg["band_rows"])
    tol, min_rows = float(mcfg["lane_tol_px"]), int(mcfg["min_gt_rows"])
    n_scorable = int(scorable(gl, rows, min_rows)) + int(scorable(gr, rows, min_rows))
    n_found = int(gl is not None) + int(gr is not None)
    gt_px = int(np.count_nonzero(gt_mask[band_slice(hw[0], mcfg["band_frac"])]))
    tod, weather, scene = frame_attrs(frame)
    rec: dict[str, Any] = {
        "name": frame["name"],
        "hw": [int(hw[0]), int(hw[1])],
        "label_scaled": scale != (1.0, 1.0),
        "timeofday": tod,
        "weather": weather,
        "scene": scene,
        "geometry": geometry_of((gl, gr), rows, float(mcfg["curve_px"]), min_rows),
        "gt_sides": n_scorable,  # scorable GT boundaries (same rule as F1 and geometry)
        "gt_short": n_found - n_scorable,  # GT boundary present but too short in the band
        "n_lane_labels": len(lanes),
        "gt_geom": gt_geometry(gt_mask, (gl, gr), rows),
    }
    vis: dict[str, dict] = {}
    for m, det in detectors.items():
        t0 = time.perf_counter()
        res = det.detect(img, stateless=True)
        timings[m].append(1e3 * (time.perf_counter() - t0))
        for k, v in (getattr(det, "last_timing_ms", None) or {}).items():
            timings[f"{m}/{k}"].append(v)
        dbg = getattr(det, "last_debug", None) or {}
        left, right, syn = measured_sides(res, dbg.get("synthesized_side"))
        inter, union = band_iou_counts(res.ego_mask, gt_mask, mcfg["band_frac"])
        # Measured-only IoU (behaviour before the single-side fallback): a mask that used a
        # synthesized side counts as no prediction.
        meas_mask = None if syn else res.ego_mask
        inter_m, union_m = band_iou_counts(meas_mask, gt_mask, mcfg["band_frac"])
        sides = [  # only measured sides are predictions; synthesized ones never score
            score_side(left, gl, rows, tol, min_rows),
            score_side(right, gr, rows, tol, min_rows),
        ]
        rec[m] = {
            "inter": inter,
            "union": union,
            "gt_px": gt_px,
            "pred_none": res.ego_mask is None,
            "synth": syn is not None,
            "synth_side": syn,
            "inter_meas": inter_m,
            "union_meas": union_m,
            "tp": sum(s.tp for s in sides),
            "fp": sum(s.fp for s in sides),
            "fn": sum(s.fn for s in sides),
            "errs": [s.err_px for s in sides if s.err_px is not None],
            "both": left is not None and right is not None,
            "n_sides": int(left is not None) + int(right is not None),
            "y_top": det.last_debug.get("y_top"),
            "y_bottom": det.last_debug.get("y_bottom"),
        }
        if keep_vis:
            binary = np.asarray(det.last_debug.get("binary"), dtype=bool)
            fitter = getattr(det, "fitter", None)
            if fitter is not None:
                binary = binary & fitter.roi(hw)  # what the sliding window actually sees
            vis[m] = {
                "result": res,
                "binary": binary.copy(),
                "y_top": det.last_debug.get("y_top"),
                "y_bottom": det.last_debug.get("y_bottom"),
                "synth_line": dbg.get("synthesized_line") if syn else None,
            }
    return rec, vis, {"mask": gt_mask, "sides": (gl, gr)}


def failure_figure(
    cases: list[int],
    frames: list[dict],
    images_dir: Path,
    detectors: dict,
    lcfg: dict,
    cfg: dict,
    out_path: Path,
) -> Path | None:
    fig, mcfg = cfg["figure"], cfg["metrics"]
    label_wh = cfg["data"].get("label_wh")
    grid = []
    for i in cases:
        img = cv2.imread(str(images_dir / frames[i]["name"]))
        if img is None:
            continue
        rec, vis, gt = evaluate_image(
            detectors, img, frames[i], mcfg, defaultdict(list), label_wh, keep_vis=True
        )
        grid.append(debug_tiles(img, rec, gt, vis, lcfg, mcfg, fig))
    if not grid:
        return None
    cv2.imwrite(str(out_path), np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path


def speed_summary(
    timings: dict[str, list[float]], methods: Sequence[str], info: dict
) -> dict[str, dict]:
    fps: dict[str, dict] = {}
    for m in methods:
        fps[m] = timing_stats(timings[m])
        for k in STAGES:
            if timings.get(f"{m}/{k}"):
                st = timing_stats(timings[f"{m}/{k}"])
                fps[m][f"{k[:-3]}_mean_ms"] = st["mean_ms"]
                fps[m][f"{k[:-3]}_p50_ms"] = st["p50_ms"]
        for k in ("pre_ms", "post_ms"):
            if timings.get(f"{m}/{k}"):
                fps[m][f"{k[:-3]}_mean_ms"] = timing_stats(timings[f"{m}/{k}"])["mean_ms"]
        if m == "yolop":
            gpu = "CUDAExecutionProvider" in info.get("yolop_providers", [])
            fps[m]["device"] = f"GPU {info.get('gpu')}" if gpu else f"CPU {info['cpu']}"
        else:
            fps[m]["device"] = f"CPU {info['cpu']}"
    return fps


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/lanes_eval.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    t_start = time.time()

    cfg = yaml.safe_load(resolve_path(args.config).read_text())
    methods: list[str] = list(cfg["methods"])
    mcfg = cfg["metrics"]
    fig = cfg["figure"]
    quotas = cfg["smoke"]["per_timeofday"] if args.smoke else cfg["sample"]["per_timeofday"]
    warmup_runs = int(cfg["fps"]["warmup_runs"])
    dcfg = cfg["debug"]
    debug_on = bool(dcfg["enabled"]) or (args.smoke and bool(cfg["smoke"]["debug"]))
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    debug_dir = out / "debug"

    lcfg = merge_config(load_lane_config(cfg["lanes_config"]), cfg.get("overrides") or {})
    check_band_matches_warp(mcfg["band_frac"], lcfg)

    d = cfg["data"]
    label_wh = d.get("label_wh")
    images_dir = find_input(d["images_val"], d["search_root"], int(d["search_depth"]), 3)
    labels = find_input(d["labels_val"], d["search_root"], int(d["search_depth"]), 1)
    all_frames = load_frames(labels)
    n_lane_frames = sum(
        any(lab.get("category") == "lane" for lab in f.get("labels") or []) for f in all_frames
    )
    print(f"{len(all_frames)} label frames, {n_lane_frames} with lane labels", flush=True)
    if n_lane_frames == 0:
        raise SystemExit(f"{labels} has no 'lane' labels; point data.labels_val elsewhere")
    frames = [
        f
        for f in stratified_sample(
            all_frames, quotas, int(cfg["sample"]["seed"]), bool(cfg["sample"]["require_labels"])
        )
        if (images_dir / f["name"]).exists()
    ]
    del all_frames
    print(f"sampled {len(frames)} images: {dict(quotas)}", flush=True)

    info: dict[str, Any] = {"cpu": cpu_name(), "cv2_threads": cv2.getNumThreads()}
    detectors = {}
    for m in methods:
        mc = copy.deepcopy(lcfg)
        mc["method"] = m
        if m == "yolop":
            from perception.lanes.yolop import ensure_weights, file_sha256

            preload_cuda_libs()
            info["yolop_weights_sha256"] = file_sha256(ensure_weights(mc["yolop"]))
        detectors[m] = build_lane_detector(mc)
        if m == "yolop":
            info["yolop_providers"] = detectors[m].providers
            info["gpu"] = gpu_name()
            if "CUDAExecutionProvider" not in info["yolop_providers"]:
                print("WARNING: YOLOP runs without the CUDA provider; FPS is CPU FPS", flush=True)
    print("setup:", info, flush=True)

    # Warm-up (cuDNN algorithm search, OpenCV/ORT allocations) on the first image, untimed.
    first = cv2.imread(str(images_dir / frames[0]["name"])) if frames else None
    if first is not None:
        for det in detectors.values():
            for _ in range(warmup_runs):
                det.detect(first, stateless=True)
    if "yolop" in detectors:
        from perception.lanes.yolop import benchmark_session

        b = cfg["fps"]["model_bench"]
        det = detectors["yolop"]
        in_w, in_h = det.input_wh
        info["yolop_model_bench"] = benchmark_session(
            det.session, (in_h, in_w), int(b["runs"]), int(b["warmup"])
        )
        print("yolop model-only:", info["yolop_model_bench"], flush=True)

    import wandb

    w = cfg["wandb"]
    run = wandb.init(
        project=w["project"],
        group=w["group"],
        job_type="smoke" if args.smoke else "eval",
        name=w["name"] + ("-smoke" if args.smoke else ""),
        config={**cfg, "lanes": lcfg, "n_images": len(frames), "smoke": args.smoke},
    )

    records: list[dict] = []
    kept_frames: list[dict] = []
    timings: dict[str, list[float]] = defaultdict(list)
    debug_idx = set(spread_indices(len(frames), int(dcfg["n_images"]))) if debug_on else set()
    debug_files: list[Path] = []
    if debug_on:
        debug_dir.mkdir(parents=True, exist_ok=True)
    t_loop = time.time()
    for k, frame in enumerate(frames):
        img = cv2.imread(str(images_dir / frame["name"]))
        if img is None:
            continue
        keep = k in debug_idx
        rec, vis, gt = evaluate_image(detectors, img, frame, mcfg, timings, label_wh, keep)
        records.append(rec)
        kept_frames.append(frame)
        if keep:
            path = debug_dir / f"{len(debug_files):02d}_{Path(frame['name']).stem}.jpg"
            tiles = debug_tiles(img, rec, gt, vis, lcfg, mcfg, fig)
            cv2.imwrite(str(path), tiles, [cv2.IMWRITE_JPEG_QUALITY, 88])
            debug_files.append(path)
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(frames)} images, {time.time() - t_loop:.0f} s", flush=True)

    rows = aggregate(records, methods)
    fps = speed_summary(timings, methods, info)
    geom = gt_geometry_summary(records)
    n_scaled = sum(int(r["label_scaled"]) for r in records)
    print("gt geometry (fractions of H / W):", json.dumps(geom), flush=True)
    if n_scaled:
        print(f"{n_scaled} images were not {label_wh}; GT rescaled to the image", flush=True)

    bench = info.get("yolop_model_bench") or {}
    fallback_note = (
        f"on (lane width {lcfg.get('fallback_lane_width_frac')} of the bird's-eye width, "
        f"confidence x{lcfg.get('fallback_conf_mult')})"
        if lcfg.get("single_side_fallback")
        else "off"
    )
    notes = [
        f"BDD100K val, {len(records)} images (seed {cfg['sample']['seed']}, stratified by "
        f"timeofday {dict(quotas)}); each image independent (`detect(stateless=True)`); "
        f"camera geometry {cfg['lanes_config']}.",
        f"IoU: ego_mask vs BDD drivable area 'direct', rows {mcfg['band_frac']} of the image "
        "height; mean of per-image IoU over images with GT in the band (no prediction = 0). "
        f"Single-side fallback {fallback_note}: masks with one synthesized side are counted "
        "in 'synth'; 'IoU measured-only' scores those images as no prediction (old behaviour).",
        f"F1: measured left/right ego boundaries only (synthesized sides never count) vs the "
        f"nearest BDD parallel lane lines, TP if mean x error <= {mcfg['lane_tol_px']} px over "
        f"the band; P/R = precision/recall. GT sides = scorable GT boundaries (>= "
        f"{mcfg['min_gt_rows']} of {mcfg['band_rows']} band rows); 'GT short' boundaries exist "
        "but are too short in the band and are not scored; geometry 'no_gt' = no scorable GT.",
        f"Speed: detect() wall time after {warmup_runs} warm-up calls, incl. pre/post-processing "
        "and polyline fit; Backend = binary stage, Model = onnxruntime session.run, Fit = shared "
        "warp + sliding window + polyfit. Classical on CPU, YOLOP (ONNX Runtime FP32) on GPU.",
    ]
    if bench:
        notes.append(
            f"YOLOP model only, fixed tensor, median of {bench.get('runs')}: "
            f"{_fmt(bench.get('run_ms'), 1)} ms with host I/O, "
            f"{_fmt(bench.get('iobind_ms'), 1)} ms with CUDA I/O binding."
        )
    md = markdown_table(rows, methods, fps, notes)
    (out / "lanes_comparison.md").write_text(md)
    with open(out / "per_image.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    fig_path = failure_figure(
        pick_failures(records, methods, int(fig["per_method"])),
        kept_frames,
        images_dir,
        detectors,
        lcfg,
        cfg,
        out / "lanes_failures.jpg",
    )
    print(md, flush=True)

    summary: dict[str, Any] = {}
    for r in rows:
        prefix = r["method"] if r["group"] == "all" else f"{r['method']}/{r['group']}/{r['key']}"
        for key in (
            "iou_mean",
            "iou_micro",
            "iou_when_pred",
            "iou_measured",
            "synth",
            "lane_f1",
            "lane_precision",
            "lane_recall",
            "both_sides_rate",
            "gt_empty",
            "pred_none",
        ):
            if r[key] is not None:
                summary[f"{prefix}/{key}"] = r[key]
    for m, s in fps.items():
        summary.update({f"{m}/{k}": v for k, v in s.items() if isinstance(v, float)})
    for k, v in geom.items():
        if "median" in v:
            summary[f"gt_geometry/{k}_median"] = v["median"]
    run.summary.update(summary)
    log: dict[str, Any] = {
        "comparison": wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows]),
        "comparison_md": wandb.Html(f"<pre>{md}</pre>"),
    }
    caption = (
        "GT tile: magenta direct area, yellow GT boundaries. Method tiles: cyan lane binary, "
        "green predicted ego lane, blue/red predicted left/right, thin orange synthesized "
        "(single-side fallback, label SYN-L/R), thin yellow GT; grey band"
    )
    if fig_path:
        log["failures"] = wandb.Image(str(fig_path), caption=caption)
    if debug_files:
        log["debug"] = [wandb.Image(str(p), caption=f"{p.stem}: {caption}") for p in debug_files]
    run.log(log)

    results = {
        "images": len(records),
        "smoke": args.smoke,
        "setup": info,
        "fps": fps,
        "gt_geometry": geom,
        "label_rescaled_images": n_scaled,
        "rows": rows,
        "failures": str(fig_path) if fig_path else None,
        "debug": [str(p) for p in debug_files],
        "wandb": run.url,
        "minutes": (time.time() - t_start) / 60,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out}; W&B run {run.url}; {results['minutes']:.1f} min", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
