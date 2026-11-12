"""M3.3 + M3.4: classical vs YOLOP lanes on BDD100K val. Runs on Kaggle (kaggle/eval_lanes).

    python -m scripts.eval_lanes --config configs/lanes_eval.yaml            # 500 images
    python -m scripts.eval_lanes --config configs/lanes_eval.yaml --smoke    # 30 images

Data: BDD100K val images plus the 2018 release label file, which holds per-image attributes
(timeofday, weather, scene), ``lane`` polylines and ``drivable area`` polygons. 500 images are
sampled with a fixed seed, stratified by timeofday (quotas in the config).

Every detector runs with ``detect(frame, stateless=True)``: each image is independent, so no
smoothing, no ``every_n_frames`` reuse and no state carried between images.

Metrics (all scored inside the row band ``metrics.band_frac`` = the warp source rows, i.e. the
part of the image both methods model):

* **Ego-lane IoU.** Prediction: ``LaneResult.ego_mask`` (region between the fitted left and
  right boundaries; empty if either side is missing). Ground truth: union of the BDD
  ``drivable area`` polygons with ``areaType == "direct"`` (the ego vehicle's own lane). Per
  image IoU = |P & G| / |P | G| over the band rows; images where both are empty are skipped.
  Reported as the mean of per-image IoU and as the micro IoU (sum of intersections / sum of
  unions). Note: BDD "direct" stops at the vehicle ahead, the predicted mask does not.
* **Lane-line F1.** GT boundaries: among BDD ``lane`` polylines with
  ``laneDirection == "parallel"`` (excluding ``metrics.exclude_lane_types``) that reach below
  ``gt_min_bottom_frac``, rank them by their x at row ``gt_ref_frac`` (linear extrapolation
  below the last vertex). The GT left boundary is the closest one left of the image centre, the
  GT right boundary the closest one right of it (so each image has 0-2 GT boundaries). A GT
  boundary counts only if it covers at least ``min_gt_rows`` of the ``band_rows`` sampled rows.
  For each side: predicted and GT -> distance = mean |x_pred(y) - x_gt(y)| over the GT-covered
  sampled rows; distance <= ``lane_tol_px`` is a TP, otherwise one FP and one FN. Prediction
  without GT -> FP; GT without prediction -> FN. F1 = 2TP / (2TP + FP + FN), pooled over sides
  and images.
* **FPS.** Wall time of ``detect`` per image after ``fps.warmup`` images, excluding disk I/O.
  Classical runs on the CPU (OpenCV), YOLOP on the GPU (onnxruntime CUDA provider), both in the
  same Kaggle session. YOLOP time includes pre/post-processing and the shared polyline fit.

Breakdowns: overall, timeofday, weather and geometry (``curve`` when a GT boundary deviates more
than ``curve_px`` from its chord inside the band, ``straight`` otherwise, ``no_gt`` without GT
boundaries). Outputs in ``out_dir``: results.json, per_image.jsonl, lanes_comparison.md,
lanes_failures.jpg; the same go to W&B (group ``lanes``).
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


# ---------------------------------------------------------------- BDD label helpers (pure)


def flatten_poly2d(
    vertices: Sequence[Sequence[float]], types: str | None, closed: bool, steps: int
) -> np.ndarray:
    """BDD poly2d -> (N, 2) float points. ``types`` marks each vertex as L (on the curve) or C
    (cubic Bezier control point); each ``L C C L`` run is sampled with ``steps`` points."""
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


def direct_area_mask(frame: dict, hw: tuple[int, int], steps: int) -> np.ndarray:
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
                cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def lane_polylines(frame: dict, steps: int, exclude_types: Iterable[str]) -> list[np.ndarray]:
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


def geometry_of(gt_sides: Sequence[np.ndarray | None], rows: np.ndarray, curve_px: float) -> str:
    """``curve`` if any GT boundary deviates > curve_px from its chord inside the band."""
    seen = False
    for pts in gt_sides:
        if pts is None:
            continue
        r, x = x_on_rows(pts, rows)
        if r.size < 3:
            continue
        seen = True
        chord = x[0] + (x[-1] - x[0]) * (r - r[0]) / (r[-1] - r[0])
        if np.abs(x - chord).max() > curve_px:
            return "curve"
    return "straight" if seen else "no_gt"


@dataclass
class SideScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    err_px: float | None = None


def score_side(
    pred: Polyline | None, gt: np.ndarray | None, rows: np.ndarray, tol_px: float, min_rows: int
) -> SideScore:
    """Score one boundary (definition in the module docstring)."""
    gt_rows = gt_x = None
    if gt is not None:
        gt_rows, gt_x = x_on_rows(gt, rows)
        if gt_rows.size < min_rows:
            gt_rows = gt_x = None
    if gt_rows is None:
        return SideScore(fp=int(pred is not None))
    if pred is None:
        return SideScore(fn=1)
    err = float(np.mean(np.abs(np.asarray(pred.x_at(gt_rows)) - gt_x)))
    if err <= tol_px:
        return SideScore(tp=1, err_px=err)
    return SideScore(fp=1, fn=1, err_px=err)


def band_iou_counts(
    pred: np.ndarray | None, gt: np.ndarray, band_frac: Sequence[float]
) -> tuple[int, int]:
    """(intersection, union) pixel counts inside the row band."""
    h = gt.shape[0]
    y0 = max(int(np.ceil(band_frac[0] * (h - 1))), 0)
    y1 = min(int(np.floor(band_frac[1] * (h - 1))), h - 1) + 1
    g = gt[y0:y1]
    p = pred[y0:y1] if pred is not None else np.zeros_like(g)
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
    inter: int = 0
    union: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    errs: list[float] = field(default_factory=list)
    both: int = 0

    def add(self, m: dict) -> None:
        self.n += 1
        self.inter += m["inter"]
        self.union += m["union"]
        if m["union"] > 0:
            self.iou_sum += m["inter"] / m["union"]
            self.iou_n += 1
        self.tp += m["tp"]
        self.fp += m["fp"]
        self.fn += m["fn"]
        self.errs.extend(m["errs"])
        self.both += int(m["both"])

    def row(self) -> dict:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else None
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else None
        denom = 2 * self.tp + self.fp + self.fn
        return {
            "n": self.n,
            "iou_mean": self.iou_sum / self.iou_n if self.iou_n else None,
            "iou_micro": self.inter / self.union if self.union else None,
            "lane_f1": 2 * self.tp / denom if denom else None,
            "lane_precision": p,
            "lane_recall": r,
            "lane_err_px": float(np.mean(self.errs)) if self.errs else None,
            "both_sides_rate": self.both / self.n if self.n else None,
        }


def aggregate(records: Sequence[dict], methods: Sequence[str]) -> list[dict]:
    """One row per (method, group, key) from per-image records."""
    tallies: dict[tuple[str, str, str], Tally] = defaultdict(Tally)
    for rec in records:
        keys = {"all": "all", **{g: rec[g] for g in GROUPS if g != "all"}}
        for m in methods:
            for g, k in keys.items():
                tallies[(m, g, k)].add(rec[m])
    order = {g: i for i, g in enumerate(GROUPS)}
    rows = [{"method": m, "group": g, "key": k, **t.row()} for (m, g, k), t in tallies.items()]
    return sorted(rows, key=lambda r: (order[r["group"]], r["key"], methods.index(r["method"])))


def timing_stats(ms: Sequence[float], warmup: int) -> dict:
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
    """Comparison table: one line per subset, IoU and F1 side by side, then a speed table."""
    by_key: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_key[(r["group"], r["key"])][r["method"]] = r
    head = ["Group", "Subset", "N"]
    head += [f"IoU {m}" for m in methods] + [f"F1 {m}" for m in methods]
    head += [f"P/R {m}" for m in methods]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (g, k), per in by_key.items():
        n = next(iter(per.values()))["n"]
        cells = [g, k, str(n)]
        cells += [_fmt(per.get(m, {}).get("iou_mean")) for m in methods]
        cells += [_fmt(per.get(m, {}).get("lane_f1")) for m in methods]
        cells += [
            f"{_fmt(per.get(m, {}).get('lane_precision'), 2)}/"
            f"{_fmt(per.get(m, {}).get('lane_recall'), 2)}"
            for m in methods
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Method | Device | Frames | Mean ms | p50 ms | p90 ms | FPS |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in methods:
        s = fps.get(m, {})
        cells = [m, s.get("device", "-"), str(s.get("frames", 0))]
        cells += [_fmt(s.get(k), 1) for k in ("mean_ms", "p50_ms", "p90_ms", "fps")]
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


def draw_gt(
    img: np.ndarray, gt_mask: np.ndarray, sides: Sequence[np.ndarray | None], fig: dict
) -> np.ndarray:
    out = img.copy()
    tint = np.zeros_like(out)
    tint[gt_mask] = fig["gt_mask_color"]
    out[gt_mask] = cv2.addWeighted(out, 0.6, tint, 0.4, 0)[gt_mask]
    for pts in sides:
        if pts is not None:
            cv2.polylines(
                out,
                [np.round(pts).astype(np.int32).reshape(-1, 1, 2)],
                False,
                tuple(int(c) for c in fig["gt_color"]),
                3,
            )
    return out


def pick_failures(records: Sequence[dict], methods: Sequence[str], per_method: int) -> list[int]:
    """Indices of the lowest-IoU images per method (GT or prediction present), deduplicated."""
    chosen: list[int] = []
    for m in methods:
        cand = [i for i, r in enumerate(records) if r[m]["union"] > 0 and i not in chosen]
        cand.sort(key=lambda i: records[i][m]["inter"] / records[i][m]["union"])
        chosen.extend(cand[:per_method])
    return chosen


# ---------------------------------------------------------------- main


def evaluate_image(
    detectors: dict, img: np.ndarray, frame: dict, mcfg: dict, timings: dict
) -> tuple[dict, dict[str, LaneResult], dict]:
    hw = img.shape[:2]
    steps = int(mcfg["bezier_steps"])
    gt_mask = direct_area_mask(frame, hw, steps)
    lanes = lane_polylines(frame, steps, mcfg["exclude_lane_types"])
    gl, gr = ego_boundaries(
        lanes, hw, float(mcfg["gt_ref_frac"]), float(mcfg["gt_min_bottom_frac"])
    )
    rows = band_rows(hw, mcfg["band_frac"], mcfg["band_rows"])
    tod, weather, scene = frame_attrs(frame)
    rec: dict[str, Any] = {
        "name": frame["name"],
        "timeofday": tod,
        "weather": weather,
        "scene": scene,
        "geometry": geometry_of((gl, gr), rows, float(mcfg["curve_px"])),
        "gt_sides": int(gl is not None) + int(gr is not None),
        "n_lane_labels": len(lanes),
    }
    results = {}
    for m, det in detectors.items():
        t0 = time.perf_counter()
        res = det.detect(img, stateless=True)
        timings[m].append(1e3 * (time.perf_counter() - t0))
        extra = getattr(det, "last_timing_ms", None)
        if extra:
            timings[m + "_model"].append(extra["model_ms"])
        inter, union = band_iou_counts(res.ego_mask, gt_mask, mcfg["band_frac"])
        sides = [
            score_side(res.left, gl, rows, float(mcfg["lane_tol_px"]), int(mcfg["min_gt_rows"])),
            score_side(res.right, gr, rows, float(mcfg["lane_tol_px"]), int(mcfg["min_gt_rows"])),
        ]
        rec[m] = {
            "inter": inter,
            "union": union,
            "tp": sum(s.tp for s in sides),
            "fp": sum(s.fp for s in sides),
            "fn": sum(s.fn for s in sides),
            "errs": [s.err_px for s in sides if s.err_px is not None],
            "both": res.left is not None and res.right is not None,
            "y_top": det.last_debug.get("y_top"),
            "y_bottom": det.last_debug.get("y_bottom"),
        }
        results[m] = res
    return rec, results, {"mask": gt_mask, "sides": (gl, gr)}


def failure_figure(
    cases: list[int],
    records: list[dict],
    frames: list[dict],
    images_dir: Path,
    detectors: dict,
    lcfg: dict,
    cfg: dict,
    out_path: Path,
) -> Path | None:
    from perception.lanes.render import draw_lanes

    fig, mcfg = cfg["figure"], cfg["metrics"]
    tw = int(fig["tile_width"])
    grid = []
    for i in cases:
        rec, frame = records[i], frames[i]
        img = cv2.imread(str(images_dir / frame["name"]))
        if img is None:
            continue
        rec2, results, gt = evaluate_image(detectors, img, frame, mcfg, defaultdict(list))
        tiles = [
            _label(
                draw_gt(img, gt["mask"], gt["sides"], fig),
                f"GT {rec['timeofday']}, {rec['weather']}, {rec['geometry']}",
            )
        ]
        for m, res in results.items():
            r = rec2[m]
            iou = r["inter"] / r["union"] if r["union"] else float("nan")
            vis = draw_lanes(img, res, r["y_top"] or 0.0, lcfg["viz"], r["y_bottom"])
            tiles.append(_label(vis, f"{m} IoU {iou:.2f} TP{r['tp']} FP{r['fp']} FN{r['fn']}"))
        th = int(round(img.shape[0] * tw / img.shape[1]))
        grid.append(
            np.hstack([cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA) for t in tiles])
        )
    if not grid:
        return None
    cv2.imwrite(str(out_path), np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/lanes_eval.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    t_start = time.time()

    cfg = yaml.safe_load(resolve_path(args.config).read_text())
    methods: list[str] = list(cfg["methods"])
    mcfg = cfg["metrics"]
    quotas = cfg["smoke"]["per_timeofday"] if args.smoke else cfg["sample"]["per_timeofday"]
    warmup = int(cfg["smoke"]["fps_warmup"] if args.smoke else cfg["fps"]["warmup"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    d = cfg["data"]
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

    lcfg = merge_config(load_lane_config(cfg["lanes_config"]), cfg.get("overrides") or {})
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
    t_loop = time.time()
    for k, frame in enumerate(frames):
        img = cv2.imread(str(images_dir / frame["name"]))
        if img is None:
            continue
        rec, _, _ = evaluate_image(detectors, img, frame, mcfg, timings)
        records.append(rec)
        kept_frames.append(frame)
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(frames)} images, {time.time() - t_loop:.0f} s", flush=True)

    rows = aggregate(records, methods)
    fps: dict[str, dict] = {}
    for m in methods:
        fps[m] = timing_stats(timings[m], warmup)
        if timings.get(m + "_model"):
            fps[m]["model_mean_ms"] = timing_stats(timings[m + "_model"], warmup).get("mean_ms")
        if m == "yolop":
            gpu = "CUDAExecutionProvider" in info.get("yolop_providers", [])
            fps[m]["device"] = f"GPU {info.get('gpu')}" if gpu else f"CPU {info['cpu']}"
        else:
            fps[m]["device"] = f"CPU {info['cpu']}"

    notes = [
        f"BDD100K val, {len(records)} images (seed {cfg['sample']['seed']}, stratified by "
        f"timeofday {dict(quotas)}); each image independent (`detect(stateless=True)`).",
        f"IoU: ego_mask vs BDD drivable area 'direct', rows {mcfg['band_frac']} of the image "
        "height; mean of per-image IoU.",
        f"F1: left/right ego boundaries vs the nearest BDD parallel lane lines, TP if mean x "
        f"error <= {mcfg['lane_tol_px']} px over the band; P/R = precision/recall.",
        "FPS: detect() wall time after warm-up, incl. pre/post-processing and polyline fit; "
        "classical on CPU, YOLOP (ONNX Runtime) on GPU, same Kaggle session.",
    ]
    md = markdown_table(rows, methods, fps, notes)
    (out / "lanes_comparison.md").write_text(md)
    with open(out / "per_image.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    fig_path = failure_figure(
        pick_failures(records, methods, int(cfg["figure"]["per_method"])),
        records,
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
        for key in ("iou_mean", "iou_micro", "lane_f1", "lane_precision", "lane_recall"):
            if r[key] is not None:
                summary[f"{prefix}/{key}"] = r[key]
    for m, s in fps.items():
        summary.update({f"{m}/{k}": v for k, v in s.items() if isinstance(v, float)})
    run.summary.update(summary)
    log: dict[str, Any] = {
        "comparison": wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows]),
        "comparison_md": wandb.Html(f"<pre>{md}</pre>"),
    }
    if fig_path:
        log["failures"] = wandb.Image(
            str(fig_path),
            caption="magenta: GT direct area, yellow: GT boundaries; "
            "green: predicted ego lane, blue/red: predicted left/right",
        )
    run.log(log)

    results = {
        "images": len(records),
        "smoke": args.smoke,
        "setup": info,
        "fps": fps,
        "rows": rows,
        "failures": str(fig_path) if fig_path else None,
        "wandb": run.url,
        "minutes": (time.time() - t_start) / 60,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out}; W&B run {run.url}; {results['minutes']:.1f} min", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
