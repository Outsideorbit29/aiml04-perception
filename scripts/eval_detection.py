"""M2.3: evaluate the fine-tuned YOLO11s on BDD val. Runs on Kaggle (kaggle/eval_track).

    python -m scripts.eval_detection --config configs/yolo_eval.yaml            # full val (10k)
    python -m scripts.eval_detection --config configs/yolo_eval.yaml --smoke    # 300 images

Steps:
1. Download ``best.pt`` from the W&B artifact (``wandb.weights_artifact``), stage the val split.
2. Ultralytics ``model.val()``: mAP, per-class P/R/AP, confusion matrix and PR curves; then mAP
   per BDD time-of-day subset (day / night / dawn-dusk).
3. Error breakdown with ``perception.detection.Detector`` at the operating threshold: recall by
   box size, time of day, occlusion and class; precision by time of day; batch-1 latency.
4. Failure galleries (night, small objects, occlusion): missed GT in red, false positives in
   yellow, found GT in green.
Everything is logged to W&B (group ``detect``) and saved to ``out_dir``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from perception.detection.detector import load_names, resolve_path
from perception.detection.processing import box_iou
from perception.types import Detection

SIZE_KEYS = ("small", "medium", "large")
COLOR_HIT = (0, 200, 0)  # BGR
COLOR_MISS = (0, 0, 255)
COLOR_FP = (0, 230, 255)


# ---------- pure helpers (unit-tested) ----------
@dataclass
class GroundTruth:
    cls: np.ndarray  # (N,) int
    boxes: np.ndarray  # (N, 4) float32 xyxy, original pixels
    occluded: np.ndarray  # (N,) bool


def gt_from_meta(
    rows: Sequence[Sequence[float]], width: int, height: int, min_box_px: float
) -> GroundTruth:
    """Rows from data-prep ``meta/frames_val.jsonl.gz``: [cls, x1, y1, x2, y2, occluded, trunc].

    The meta boxes are raw BDD boxes, so they get the same treatment as the YOLO labels
    (``scripts.data.bdd_yolo.frame_to_yolo``): order the corners, clip to the image, and drop
    boxes thinner than ``min_box_px`` after clipping.
    """
    arr = np.asarray(rows, np.float32).reshape(-1, 7)
    xs, ys = np.sort(arr[:, [1, 3]], axis=1), np.sort(arr[:, [2, 4]], axis=1)
    xs, ys = xs.clip(0, float(width)), ys.clip(0, float(height))
    keep = (xs[:, 1] - xs[:, 0] >= min_box_px) & (ys[:, 1] - ys[:, 0] >= min_box_px)
    boxes = np.stack([xs[:, 0], ys[:, 0], xs[:, 1], ys[:, 1]], axis=1)[keep]
    return GroundTruth(arr[keep, 0].astype(int), boxes.astype(np.float32), arr[keep, 5] > 0)


def load_min_box_px(data_config: str | Path) -> float:
    """``prep.min_box_px`` from the data config that ``scripts.data.prepare_bdd`` used."""
    return float(yaml.safe_load(resolve_path(data_config).read_text())["prep"]["min_box_px"])


def gt_from_yolo(text: str, w: int, h: int) -> GroundTruth:
    """YOLO label text (cls cx cy bw bh, normalised) -> pixel xyxy. Occlusion is unknown."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    arr = np.asarray(rows, np.float32).reshape(-1, 5)
    cx, cy, bw, bh = arr[:, 1] * w, arr[:, 2] * h, arr[:, 3] * w, arr[:, 4] * h
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    return GroundTruth(arr[:, 0].astype(int), boxes, np.zeros(len(arr), bool))


def dets_to_arrays(dets: Sequence[Detection]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not dets:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    return (
        np.stack([d.bbox_xyxy for d in dets]).astype(np.float32),
        np.array([d.score for d in dets], np.float32),
        np.array([d.cls for d in dets], int),
    )


def match_image(
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_cls: np.ndarray,
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy same-class matching by descending score (COCO style, one GT per prediction).

    Returns (gt_hit, pred_tp) boolean arrays.
    """
    gt_hit = np.zeros(len(gt_boxes), bool)
    pred_tp = np.zeros(len(pred_boxes), bool)
    if not len(gt_boxes) or not len(pred_boxes):
        return gt_hit, pred_tp
    iou = box_iou(pred_boxes, gt_boxes)
    iou[np.asarray(pred_cls)[:, None] != np.asarray(gt_cls)[None, :]] = 0.0
    for p in np.argsort(-np.asarray(pred_scores), kind="stable"):
        cand = np.flatnonzero(~gt_hit & (iou[p] >= iou_thr))
        if cand.size:
            gt_hit[cand[np.argmax(iou[p, cand])]] = True
            pred_tp[p] = True
    return gt_hit, pred_tp


def size_buckets(boxes: np.ndarray, small: float, medium: float) -> np.ndarray:
    """'small' / 'medium' / 'large' per box by pixel area (COCO-style thresholds)."""
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return np.where(area < small, SIZE_KEYS[0], np.where(area < medium, SIZE_KEYS[1], SIZE_KEYS[2]))


class Breakdown:
    """Counts GT recall and prediction precision per (group, key), e.g. ("size", "small")."""

    def __init__(self) -> None:
        self.gt: dict[tuple[str, str], int] = defaultdict(int)
        self.gt_hit: dict[tuple[str, str], int] = defaultdict(int)
        self.pred: dict[tuple[str, str], int] = defaultdict(int)
        self.pred_tp: dict[tuple[str, str], int] = defaultdict(int)

    def add_gt(self, group: str, keys: Iterable[str], hits: Iterable[bool]) -> None:
        for k, h in zip(keys, hits, strict=True):
            self.gt[(group, str(k))] += 1
            self.gt_hit[(group, str(k))] += int(h)

    def add_pred(self, group: str, key: str, n_pred: int, n_tp: int) -> None:
        self.pred[(group, key)] += int(n_pred)
        self.pred_tp[(group, key)] += int(n_tp)

    def rows(self) -> list[dict[str, Any]]:
        out = []
        for group, key in sorted(set(self.gt) | set(self.pred)):
            n_gt, n_pred = self.gt.get((group, key), 0), self.pred.get((group, key), 0)
            out.append(
                {
                    "group": group,
                    "key": key,
                    "gt": n_gt,
                    "recall": self.gt_hit.get((group, key), 0) / n_gt if n_gt else None,
                    "pred": n_pred,
                    "precision": self.pred_tp.get((group, key), 0) / n_pred if n_pred else None,
                }
            )
        return out


FAILURE_KEYS = {
    "night": lambda r: r["fn"] + r["fp"],
    "small": lambda r: r["fn_small"],
    "occlusion": lambda r: r["fn_occluded"],
}


def pick_failures(
    records: Sequence[dict[str, Any]], case: str, k: int, min_gt: int = 0
) -> list[dict[str, Any]]:
    """Worst ``k`` images for a gallery case. Only images with at least one error are kept."""
    key = FAILURE_KEYS[case]
    pool = [r for r in records if r["n_gt"] >= min_gt and key(r) > 0]
    if case == "night":
        pool = [r for r in pool if r["timeofday"] == "night"]
    return sorted(pool, key=lambda r: (-key(r), r["name"]))[:k]


def latency_stats(ms: Sequence[float]) -> dict[str, float]:
    if not ms:
        return {}
    a = np.asarray(ms, np.float64)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "fps": float(1000.0 / a.mean()),
        "frames": int(a.size),
    }


# ---------- drawing ----------
def _ints(box: np.ndarray) -> list[int]:
    return [int(round(float(v))) for v in box]


def draw_errors(
    img: np.ndarray,
    gt: GroundTruth,
    gt_hit: np.ndarray,
    pred: tuple[np.ndarray, np.ndarray, np.ndarray],
    pred_tp: np.ndarray,
    names: dict[int, str],
    title: str,
) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, hit in zip(gt.boxes, gt_hit, strict=True):
        x1, y1, x2, y2 = _ints(box)
        color, thick = (COLOR_HIT, 1) if hit else (COLOR_MISS, 3)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    boxes, scores, cls = pred
    for box, s, c, tp in zip(boxes, scores, cls, pred_tp, strict=True):
        if tp:
            continue
        x1, y1, x2, y2 = _ints(box)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_FP, 2)
        label = f"{names.get(int(c), c)} {s:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 3, 12)), font, 0.5, COLOR_FP, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 24), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_grid(tiles: list[np.ndarray], cols: int, tile_w: int) -> np.ndarray:
    tiles = [cv2.resize(t, (tile_w, int(round(t.shape[0] * tile_w / t.shape[1])))) for t in tiles]
    tile_h = max(t.shape[0] for t in tiles)
    padded = [
        cv2.copyMakeBorder(t, 0, tile_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles
    ]
    while len(padded) % cols:
        padded.append(np.zeros_like(padded[0]))
    rows = [np.hstack(padded[i : i + cols]) for i in range(0, len(padded), cols)]
    return np.vstack(rows)


# ---------- Kaggle-side steps ----------
def fetch_weights(run: Any, ref: str, dest: Path) -> Path:
    """Download ``best.pt`` from a W&B artifact and record it as an input of ``run``."""
    art = run.use_artifact(ref)
    path = Path(art.download(root=str(dest))) / "best.pt"
    print(f"weights {art.name} -> {path} {art.metadata}", flush=True)
    return path


def stage_val(cfg: dict[str, Any]) -> Path:
    from scripts.data.check_data_yaml import find_dataset_root
    from scripts.train_yolo import copy_tree

    try:
        src = find_dataset_root(Path(cfg["search"]))
    except FileNotFoundError:  # mount layout changed: fall back to a full (slower) search
        src = find_dataset_root(Path(cfg["search_fallback"]))
    dst = Path(cfg["stage_dir"])
    t0 = time.time()
    n = sum(copy_tree(src / sub, dst / sub) for sub in ("images/val", "labels/val"))
    meta = src / "meta" / "frames_val.jsonl.gz"
    if meta.exists():
        (dst / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(meta, dst / "meta" / meta.name)
    print(f"staged {n} val files from {src} in {time.time() - t0:.0f} s", flush=True)
    return dst


def load_meta(path: Path) -> dict[str, dict[str, Any]]:
    """Frame attributes and GT boxes from data-prep, keyed by image stem."""
    if not path.exists():
        print(f"WARNING: {path} missing; using YOLO labels, no timeofday/occlusion", flush=True)
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {Path(r["name"]).stem: r for r in rows}


def ground_truth(
    img_path: Path, meta: dict[str, dict[str, Any]], shape: tuple, min_px: float
) -> GroundTruth:
    m = meta.get(img_path.stem)
    if m is not None:  # original pixels; val images are not resized (prep.max_side = 1280)
        return gt_from_meta(m["boxes"], shape[1], shape[0], min_px)
    from scripts.train_yolo import label_path

    return gt_from_yolo(label_path(img_path).read_text(), shape[1], shape[0])


def run_val(model: Any, names: dict[int, str], paths: list[Path], cfg: dict, name: str,
            plots: bool) -> Any:
    from scripts.train_yolo import data_yaml

    out = Path(cfg["out_dir"])
    lst = Path(cfg["stage_dir"]).parent / "eval_lists" / f"{name}.txt"
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text("\n".join(str(p) for p in paths) + "\n")
    yml = data_yaml(names, str(lst), str(lst), lst.with_suffix(".yaml"))
    v = cfg["val"]
    return model.val(
        data=str(yml), imgsz=v["imgsz"], batch=v["batch"], project=str(out), name=f"val_{name}",
        exist_ok=True, plots=plots, verbose=False,
    )


def per_class_rows(metrics: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    box = metrics.box
    rows = []
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        rows.append(
            {"class": names[int(c)], "precision": float(p), "recall": float(r),
             "ap50": float(ap50), "ap50_95": float(ap)}
        )
    return rows


def analyse(
    detector: Any, paths: list[Path], meta: dict, names: dict[int, str], a: dict[str, Any]
) -> tuple[list[dict[str, Any]], Breakdown, list[float]]:
    bd, records, lat = Breakdown(), [], []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        t0 = time.perf_counter()
        dets = detector.detect(img)
        if i >= a["fps_warmup"]:
            lat.append((time.perf_counter() - t0) * 1000.0)
        gt = ground_truth(p, meta, img.shape, a["min_box_px"])
        pb, ps, pc = dets_to_arrays(dets)
        gt_hit, pred_tp = match_image(gt.boxes, gt.cls, pb, ps, pc, a["iou_match"])
        sizes = size_buckets(gt.boxes, a["small_area_px"], a["medium_area_px"])
        tod = (meta.get(p.stem) or {}).get("timeofday", "undefined")
        n = len(gt.cls)
        bd.add_gt("all", ["all"] * n, gt_hit)
        bd.add_gt("size", sizes, gt_hit)
        bd.add_gt("timeofday", [tod] * n, gt_hit)
        bd.add_gt("occlusion", np.where(gt.occluded, "occluded", "visible"), gt_hit)
        bd.add_gt("class", [names.get(int(c), str(c)) for c in gt.cls], gt_hit)
        bd.add_pred("all", "all", len(pc), pred_tp.sum())
        bd.add_pred("timeofday", tod, len(pc), pred_tp.sum())
        records.append(
            {
                "name": p.name,
                "path": str(p),
                "timeofday": tod,
                "n_gt": n,
                "fn": int((~gt_hit).sum()),
                "fp": int((~pred_tp).sum()),
                "fn_small": int((~gt_hit & (sizes == "small")).sum()),
                "fn_occluded": int((~gt_hit & gt.occluded).sum()),
            }
        )
        if (i + 1) % 1000 == 0:
            print(f"analysed {i + 1}/{len(paths)} images", flush=True)
    return records, bd, lat


def gallery(
    detector: Any, records: list[dict], meta: dict, names: dict[int, str], cfg: dict,
    a: dict[str, Any], case: str,
) -> Path | None:
    g = cfg["gallery"]
    picks = pick_failures(records, case, g["per_case"], g["min_gt"] if case == "night" else 0)
    if not picks:
        print(f"gallery {case}: no failing images", flush=True)
        return None
    tiles = []
    for r in picks:
        p = Path(r["path"])
        img = cv2.imread(str(p))
        gt = ground_truth(p, meta, img.shape, a["min_box_px"])
        pred = dets_to_arrays(detector.detect(img))
        gt_hit, pred_tp = match_image(gt.boxes, gt.cls, *pred, a["iou_match"])
        title = f"{p.stem[:17]} {r['timeofday']} miss {r['fn']}/{r['n_gt']} fp {r['fp']}"
        tiles.append(draw_errors(img, gt, gt_hit, pred, pred_tp, names, title))
    path = Path(cfg["out_dir"]) / f"gallery_{case}.jpg"
    cv2.imwrite(str(path), make_grid(tiles, g["cols"], g["tile_width"]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/yolo_eval.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    t_start = time.time()

    cfg = yaml.safe_load(args.config.read_text())
    names = load_names({"data_config": cfg["data_config"]}) or {}
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # Same GT filtering as the YOLO labels: min_box_px comes from the data-prep config.
    a = {**cfg["analysis"], "min_box_px": load_min_box_px(cfg["data_config"])}
    limit = cfg["smoke"]["limit"] if args.smoke else a["limit"]

    import wandb
    from ultralytics import YOLO, settings

    from perception.detection import Detector, load_config

    settings.update({"wandb": False})  # this script logs to W&B itself
    w = cfg["wandb"]
    run = wandb.init(
        project=w["project"], group=w["group"], job_type="smoke" if args.smoke else "eval",
        name=w["name"] + ("-smoke" if args.smoke else ""), config=cfg,
    )
    weights = fetch_weights(run, w["weights_artifact"], Path(cfg["weights_dir"]))
    root = stage_val(cfg)
    meta = load_meta(root / "meta" / "frames_val.jsonl.gz")
    paths = sorted((root / "images" / "val").glob("*.jpg"))[:limit]
    results: dict[str, Any] = {"weights": w["weights_artifact"], "images": len(paths)}

    # 1. Ultralytics val: overall + per time of day.
    from scripts.train_yolo import summarise

    model = YOLO(str(weights))
    m_all = run_val(model, names, paths, cfg, "all", plots=True)
    results["val"] = {"all": {**summarise(m_all, names), "speed_ms": dict(m_all.speed)}}
    results["per_class"] = per_class_rows(m_all, names)
    cm = getattr(m_all, "confusion_matrix", None)
    if cm is not None:
        results["confusion_matrix"] = {
            "labels": [names[i] for i in sorted(names)] + ["background"],
            "matrix": np.asarray(cm.matrix).astype(int).tolist(),  # rows: predicted, cols: true
        }
    for cond in cfg["val"]["conditions"]:
        sub = [p for p in paths if (meta.get(p.stem) or {}).get("timeofday") == cond]
        if not sub:
            continue
        m = run_val(model, names, sub, cfg, cond.replace("/", "_"), plots=False)
        results["val"][cond] = {"images": len(sub), "map50": float(m.box.map50),
                                "map50_95": float(m.box.map)}
    print("val:", json.dumps(results["val"], indent=1), flush=True)

    # 2. Error breakdown and latency with the Detector class.
    det_cfg = load_config(cfg["detector_config"], weights=str(weights), conf=a["conf"],
                          imgsz=cfg["val"]["imgsz"], agnostic_nms=a["agnostic_nms"])
    detector = Detector(det_cfg)
    records, bd, lat = analyse(detector, paths, meta, names, a)
    results["breakdown"] = bd.rows()
    results["detector_latency"] = {**latency_stats(lat), "backend": det_cfg["backend"],
                                   "imgsz": det_cfg["imgsz"], "batch": 1}
    print("breakdown:", json.dumps(results["breakdown"], indent=1), flush=True)
    print("latency:", results["detector_latency"], flush=True)

    # 3. Failure galleries.
    cases = cfg["gallery"]["cases"]
    galleries = {c: gallery(detector, records, meta, names, cfg, a, c) for c in cases}
    results["galleries"] = {c: str(p) for c, p in galleries.items() if p}

    # 4. Log.
    summary = {f"val/{k}": v for k, v in results["val"]["all"].items() if isinstance(v, float)}
    for cond, v in results["val"].items():
        if cond != "all":
            summary[f"val_{cond}/map50"] = v["map50"]
            summary[f"val_{cond}/map50_95"] = v["map50_95"]
    for r in results["breakdown"]:
        if r["recall"] is not None:
            summary[f"recall/{r['group']}/{r['key']}"] = r["recall"]
    summary.update({f"latency/{k}": v for k, v in results["detector_latency"].items()
                    if isinstance(v, float)})
    run.summary.update(summary)
    log: dict[str, Any] = {
        key: wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])
        for key in ("per_class", "breakdown")
        if (rows := results[key])
    }
    for png in sorted((out / "val_all").glob("*.png")):
        log[f"plots/{png.stem}"] = wandb.Image(str(png))
    for case, p in galleries.items():
        if p:
            log[f"gallery/{case}"] = wandb.Image(
                str(p), caption="red: missed GT, yellow: false positive, green: found GT"
            )
    run.log(log)
    results["minutes"] = (time.time() - t_start) / 60
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / 'results.json'}; W&B run {run.url}", flush=True)
    print(f"total time {results['minutes']:.1f} min", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
