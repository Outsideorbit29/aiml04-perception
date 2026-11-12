"""M2.6: MOT metrics (MOTA, IDF1, ID switches) for Detector + ByteTrack on KITTI tracking.

    python -m scripts.eval_mot --config configs/tracker_mot.yaml            # full sweep
    python -m scripts.eval_mot --config configs/tracker_mot.yaml --smoke    # 1 seq, 50 frames

Runs on Kaggle (job kaggle/eval_mot). Steps:
1. ``Detector`` runs once per frame of every sequence; detections are cached.
2. Each tracker config of the sweep replays the cached detections through ``Tracker``, so the
   sweep costs only tracker time (a few ms per frame).
3. Hypotheses and KITTI ``label_02`` ground truth are compared with ``motmetrics`` per
   sequence and overall, per evaluation group (all / vehicle / pedestrian), with IoU >= 0.5.
   DontCare, ignored classes, tiny and heavily occluded objects are ignore regions.
Writes results.json + m2_mot.md to ``out_dir`` and logs to W&B (group ``track``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from perception.detection.processing import box_iou

IGNORE = "ignore"
DONT_CARE = "DontCare"
MAX_SEARCH_DEPTH = 6
# motmetrics name -> short key used in results.json / tables
METRICS = {
    "num_frames": "frames",
    "mota": "mota",
    "motp": "motp",
    "idf1": "idf1",
    "num_switches": "idsw",
    "mostly_tracked": "mt",
    "partially_tracked": "pt",
    "mostly_lost": "ml",
    "num_false_positives": "fp",
    "num_misses": "fn",
    "num_objects": "gt_boxes",
    "num_unique_objects": "gt_ids",
    "num_fragmentations": "frag",
    "precision": "precision",
    "recall": "recall",
}


# ---------- KITTI labels (pure, unit-tested) ----------
@dataclass(frozen=True)
class KittiObject:
    track_id: int  # -1 for DontCare
    type: str
    truncated: int
    occluded: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class FrameGT:
    ids: np.ndarray  # (N,) int, evaluated objects
    boxes: np.ndarray  # (N, 4) xyxy
    ignore: np.ndarray  # (M, 4) xyxy ignore regions


def parse_kitti_labels(text: str) -> dict[int, list[KittiObject]]:
    """KITTI tracking ``label_02/<seq>.txt`` -> {frame: objects}.

    Columns: frame track_id type truncated occluded alpha x1 y1 x2 y2 h w l X Y Z ry [score].
    """
    out: dict[int, list[KittiObject]] = {}
    for n, line in enumerate(text.splitlines(), 1):
        f = line.split()
        if not f:
            continue
        if len(f) < 10:
            raise ValueError(f"line {n}: expected >= 10 columns, got {len(f)}")
        obj = KittiObject(
            track_id=int(f[1]),
            type=f[2],
            truncated=int(float(f[3])),
            occluded=int(float(f[4])),
            box=(float(f[6]), float(f[7]), float(f[8]), float(f[9])),
        )
        out.setdefault(int(f[0]), []).append(obj)
    return out


def frame_gt(
    objs: Sequence[KittiObject],
    class_map: Mapping[str, str],
    group_classes: Iterable[str],
    match_cfg: Mapping[str, Any],
) -> FrameGT:
    """Split one frame's KITTI objects into evaluated GT and ignore regions.

    Ignore regions: DontCare, types mapped to ``ignore`` (or unknown), classes outside the
    evaluated group, and objects that are too short, too occluded or too truncated.
    """
    group = set(group_classes)
    ids, boxes, ignore = [], [], []
    for o in objs:
        cls = IGNORE if o.type == DONT_CARE else class_map.get(o.type, IGNORE)
        too_hard = (
            o.box[3] - o.box[1] < match_cfg["min_height_px"]
            or o.occluded > match_cfg["max_occlusion"]
            or o.truncated > match_cfg["max_truncation"]
        )
        if cls == IGNORE or cls not in group or too_hard:
            ignore.append(o.box)
        else:
            ids.append(o.track_id)
            boxes.append(o.box)
    return FrameGT(
        np.asarray(ids, int),
        np.asarray(boxes, np.float32).reshape(-1, 4),
        np.asarray(ignore, np.float32).reshape(-1, 4),
    )


# ---------- matching helpers (pure, unit-tested) ----------
def box_ioa(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of each box in ``a`` covered by each box in ``b``: (N, 4) x (M, 4) -> (N, M)."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    return inter / np.maximum(area_a[:, None], 1e-9)


def keep_hypotheses(
    hyp_boxes: np.ndarray, gt: FrameGT, iou_thr: float, ignore_overlap: float
) -> np.ndarray:
    """Bool mask of hypotheses to score; drops unmatched ones lying on ignore regions.

    A hypothesis that can be matched one-to-one (Hungarian, IoU >= ``iou_thr``) to an evaluated
    GT box is always kept, so an ignore region next to a real object never hides a TP.
    """
    from scipy.optimize import linear_sum_assignment

    hyp_boxes = np.asarray(hyp_boxes, np.float32).reshape(-1, 4)
    keep = np.ones(len(hyp_boxes), bool)
    if not len(hyp_boxes) or not len(gt.ignore):
        return keep
    matched = np.zeros(len(hyp_boxes), bool)
    if len(gt.boxes):
        iou = box_iou(hyp_boxes, gt.boxes)
        rows, cols = linear_sum_assignment(-iou)
        matched[rows[iou[rows, cols] >= iou_thr]] = True
    on_ignore = box_ioa(hyp_boxes, gt.ignore).max(axis=1) >= ignore_overlap
    keep &= matched | ~on_ignore
    return keep


def iou_distance(gt_boxes: np.ndarray, hyp_boxes: np.ndarray, iou_thr: float) -> np.ndarray:
    """motmetrics distance matrix (1 - IoU), NaN where IoU < ``iou_thr`` (no match allowed).

    Own version because ``motmetrics.distances.iou_matrix`` uses ``np.asfarray`` (removed in
    NumPy 2) and xywh boxes.
    """
    iou = box_iou(gt_boxes, hyp_boxes).astype(np.float64)
    dist = 1.0 - iou
    dist[iou < iou_thr] = np.nan
    return dist


@dataclass
class Hyp:
    """Tracker output of one frame (only the fields the metrics need)."""

    ids: np.ndarray  # (K,) int
    boxes: np.ndarray  # (K, 4) xyxy
    cls_names: list[str]


def accumulate(
    frames: Sequence[int],
    gt: Mapping[int, FrameGT],
    hyps: Mapping[int, Hyp],
    pred_classes: Iterable[str],
    match_cfg: Mapping[str, Any],
) -> Any:
    """One ``motmetrics.MOTAccumulator`` for a sequence and an evaluation group."""
    import motmetrics as mm

    pred_set = set(pred_classes)
    empty_gt = FrameGT(np.zeros(0, int), np.zeros((0, 4), np.float32), np.zeros((0, 4)))
    acc = mm.MOTAccumulator(auto_id=False)
    for f in frames:
        g = gt.get(f, empty_gt)
        h = hyps.get(f)
        if h is None or not len(h.ids):
            h_ids, h_boxes = np.zeros(0, int), np.zeros((0, 4), np.float32)
        else:
            heights = h.boxes[:, 3] - h.boxes[:, 1]
            sel = np.array([c in pred_set for c in h.cls_names], bool)
            sel &= heights >= match_cfg["min_height_px"]
            h_ids, h_boxes = h.ids[sel], h.boxes[sel]
            keep = keep_hypotheses(
                h_boxes, g, match_cfg["iou_threshold"], match_cfg["ignore_overlap"]
            )
            h_ids, h_boxes = h_ids[keep], h_boxes[keep]
        dist = iou_distance(g.boxes, h_boxes, match_cfg["iou_threshold"])
        acc.update(g.ids.tolist(), h_ids.tolist(), dist, frameid=f)
    return acc


def _clean(v: Any) -> Any:
    v = v.item() if hasattr(v, "item") else v
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def mot_metrics(accs: Sequence[Any], names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """{sequence name | "overall": {short metric: value}}; NaN -> None."""
    import motmetrics as mm

    mh = mm.metrics.create()
    df = mh.compute_many(
        list(accs), metrics=list(METRICS), names=list(names), generate_overall=True
    )
    out: dict[str, dict[str, Any]] = {}
    for name, row in df.iterrows():
        key = "overall" if name == "OVERALL" else str(name)
        vals = {METRICS[m]: _clean(row[m]) for m in METRICS}
        # motmetrics' MOTP is the mean matched distance (1 - IoU); report mean IoU instead.
        vals["motp"] = None if vals["motp"] is None else 1.0 - vals["motp"]
        out[key] = vals
    return out


def pick_best(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Mapping[str, Any]:
    """Row with the highest ``overall`` metrics in ``keys`` order (None counts as -inf)."""

    def score(r: Mapping[str, Any]) -> tuple[float, ...]:
        o = r["metrics"]["all"]["overall"]
        return tuple(-math.inf if o.get(k) is None else float(o[k]) for k in keys)

    return max(rows, key=score)


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{100 * v:.1f}" if pct else f"{v:.2f}"
    return str(v)


def markdown_report(result: Mapping[str, Any]) -> str:
    """Markdown tables: the sweep (group all), per-group and per-sequence for the best config."""
    pct = {"mota", "idf1", "motp", "precision", "recall"}
    cols = ["mota", "idf1", "idsw", "mt", "ml", "fp", "fn", "frag", "precision", "recall"]
    head = "| " + " | ".join(c.upper() for c in cols) + " |"
    lines = [
        "# M2.6 MOT evaluation (KITTI tracking)",
        "",
        f"Sequences: {', '.join(result['sequences'])} ({result['frames']} frames). "
        f"IoU >= {result['matching']['iou_threshold']}; MOTA/IDF1/MOTP(IoU)/P/R in %.",
        f"Detector {result['det_ms']:.1f} ms/frame (mean) on {result['device']}.",
        "",
        "## Tracker sweep (group: all)",
        "",
        "| Config | Params " + head + " Unique ids | Short <3 | Track ms | FPS det+track |",
        "|---" * (len(cols) + 6) + "|",
    ]
    for r in result["sweep"]:
        o = r["metrics"]["all"]["overall"]
        params = ", ".join(f"{k}={v}" for k, v in r["params"].items()) or "base"
        cells = [_fmt(o.get(c), c in pct) for c in cols]
        extra = [r["unique_ids"], r["short_tracks_lt3"], _fmt(r["track_ms"]), _fmt(r["fps"])]
        lines.append(f"| {r['name']} | {params} | " + " | ".join(cells + [str(e) for e in extra])
                     + " |")
    best = next(r for r in result["sweep"] if r["name"] == result["best"])
    lines += ["", f"Best by {'/'.join(result['select_by'])}: **{best['name']}**", "",
              f"## Groups ({best['name']})", "", "| Group " + head, "|---" * (len(cols) + 1) + "|"]
    for g, m in best["metrics"].items():
        lines.append(f"| {g} | " + " | ".join(_fmt(m["overall"].get(c), c in pct) for c in cols)
                     + " |")
    lines += ["", f"## Sequences ({best['name']}, group all)", "", "| Sequence " + head,
              "|---" * (len(cols) + 1) + "|"]
    for seq, m in best["metrics"]["all"].items():
        lines.append(f"| {seq} | " + " | ".join(_fmt(m.get(c), c in pct) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# ---------- Kaggle-side steps ----------
def find_kitti_dirs(cfg: dict[str, Any]) -> tuple[Path, Path]:
    """KITTI tracking ``image_02`` and ``label_02``: paths from data_config, else a search."""
    from perception.detection.detector import resolve_path
    from scripts.eval_tracking import find_image_root

    kitti = yaml.safe_load(resolve_path(cfg["data_config"]).read_text())
    labels = Path(kitti["input_root"]) / kitti["tracking"]["label_dir"]
    images = find_image_root(cfg)
    if not labels.is_dir():
        labels = images.parent / "label_02"
    if not labels.is_dir():
        raise FileNotFoundError(f"no label_02 next to {images}; this KITTI upload has no labels")
    return images, labels


def detect_sequence(
    seq_dir: Path, max_frames: int | None, detector: Any, warmup: int
) -> tuple[list[int], dict[int, list[Any]], list[float]]:
    """Run the detector on every frame; returns frame ids, detections and timings (ms)."""
    import cv2

    from scripts.eval_tracking import frame_paths

    paths = frame_paths(seq_dir, max_frames)
    frames, dets, ms = [], {}, []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        fid = int(p.stem)
        t0 = time.perf_counter()
        dets[fid] = detector.detect(img)
        if i >= warmup:
            ms.append((time.perf_counter() - t0) * 1000.0)
        frames.append(fid)
    return frames, dets, ms


def track_sequence(
    tracker: Any, frames: Sequence[int], dets: Mapping[int, list[Any]], warmup: int
) -> tuple[dict[int, Hyp], list[list[int]], list[float]]:
    tracker.reset()
    hyps, ids_per_frame, ms = {}, [], []
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        tracks = tracker.update(dets[f], None)
        if i >= warmup:
            ms.append((time.perf_counter() - t0) * 1000.0)
        ids = [t.track_id for t in tracks]
        ids_per_frame.append(ids)
        hyps[f] = Hyp(
            np.asarray(ids, int),
            np.asarray([t.det.bbox_xyxy for t in tracks], np.float32).reshape(-1, 4),
            [t.det.cls_name for t in tracks],
        )
    return hyps, ids_per_frame, ms


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/tracker_mot.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    t_start = time.time()

    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        cfg.update(cfg["smoke"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    match_cfg = cfg["matching"]
    sweep = cfg["sweep"][: cfg["max_sweep"]] if cfg.get("max_sweep") else cfg["sweep"]

    import wandb

    from perception.detection import Detector, load_config
    from perception.tracking import Tracker
    from scripts.eval_detection import fetch_weights
    from scripts.eval_tracking import track_stats

    w = cfg["wandb"]
    run = wandb.init(
        project=w["project"], group=w["group"], job_type="smoke" if args.smoke else "eval",
        name=w["name"] + ("-smoke" if args.smoke else ""), config=cfg,
    )
    weights = fetch_weights(run, w["weights_artifact"], Path(cfg["weights_dir"]))
    det_cfg = load_config(cfg["detector_config"], weights=str(weights), **cfg["detector_overrides"])
    detector = Detector(det_cfg)
    run.config.update({"detector": det_cfg}, allow_val_change=True)

    image_root, label_root = find_kitti_dirs(cfg)
    print("KITTI images:", image_root, "labels:", label_root, flush=True)

    # 1. detections + GT per sequence
    data: dict[str, dict[str, Any]] = {}
    det_ms: list[float] = []
    for seq in cfg["sequences"]:
        seq_dir, label_file = image_root / seq, label_root / f"{seq}.txt"
        if not seq_dir.is_dir() or not label_file.is_file():
            print(f"WARNING: sequence {seq} (images or labels) missing; skipped", flush=True)
            continue
        frames, dets, ms = detect_sequence(seq_dir, cfg["max_frames"], detector,
                                           cfg["latency_warmup"])
        det_ms += ms
        data[seq] = {"frames": frames, "dets": dets,
                     "labels": parse_kitti_labels(label_file.read_text())}
        print(f"{seq}: {len(frames)} frames detected", flush=True)
    if not data:
        raise SystemExit("no KITTI sequence evaluated")
    seqs = list(data)
    gt = {
        g: {s: {f: frame_gt(objs, cfg["kitti_class_map"], spec["gt"], match_cfg)
                for f, objs in data[s]["labels"].items()} for s in seqs}
        for g, spec in cfg["groups"].items()
    }

    # 2. tracker sweep on cached detections, 3. metrics
    tracker_base = load_config(cfg["tracker_config"], frame_rate=cfg["fps"])
    det_mean = float(np.mean(det_ms)) if det_ms else float("nan")
    rows = []
    for entry in sweep:
        tracker = Tracker({**tracker_base, **entry["params"]})
        hyps, ids, trk_ms = {}, [], []
        for s in seqs:
            hyps[s], seq_ids, ms = track_sequence(tracker, data[s]["frames"], data[s]["dets"],
                                                  cfg["latency_warmup"])
            ids += [[f"{s}:{i}" for i in fr] for fr in seq_ids]  # ids are per sequence
            trk_ms += ms
        metrics = {
            g: mot_metrics(
                [accumulate(data[s]["frames"], gt[g][s], hyps[s], spec["pred"], match_cfg)
                 for s in seqs],
                seqs,
            )
            for g, spec in cfg["groups"].items()
        }
        trk_mean = float(np.mean(trk_ms)) if trk_ms else float("nan")
        stats = track_stats(ids)
        row = {
            "name": entry["name"], "params": entry["params"], "metrics": metrics,
            "unique_ids": stats["unique_ids"], "short_tracks_lt3": stats["short_tracks_lt3"],
            "track_ms": trk_mean, "fps": 1000.0 / (det_mean + trk_mean),
        }
        rows.append(row)
        o = metrics["all"]["overall"]
        print(f"{entry['name']}: MOTA {o['mota']} IDF1 {o['idf1']} IDSW {o['idsw']} "
              f"unique ids {stats['unique_ids']}", flush=True)

    best = pick_best(rows, cfg["select_by"])
    import torch

    result = {
        "sequences": seqs,
        "frames": sum(len(data[s]["frames"]) for s in seqs),
        "matching": match_cfg,
        "groups": cfg["groups"],
        "det_ms": det_mean,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "sweep": rows,
        "best": best["name"],
        "select_by": cfg["select_by"],
        "wandb_url": run.url,
    }
    md = markdown_report(result)
    result["minutes"] = (time.time() - t_start) / 60
    (out / "results.json").write_text(json.dumps(result, indent=2))
    (out / "m2_mot.md").write_text(md)
    print(md, flush=True)

    # W&B: sweep table, per-sequence table of the best config, summary of default + best
    cols = ["config", "group", "sequence", *METRICS.values()]
    table = wandb.Table(columns=cols)
    for r in rows:
        for g, per_seq in r["metrics"].items():
            for s, m in per_seq.items():
                table.add_data(r["name"], g, s, *[m[k] for k in METRICS.values()])
    run.log({"mot_metrics": table})
    summary: dict[str, Any] = {"best_config": best["name"], "det_ms": det_mean}
    default = next((r for r in rows if r["name"] == "default"), rows[0])
    for tag, r in (("baseline", rows[0]), ("default", default), ("best", best)):
        for k in ("mota", "idf1", "idsw", "mt", "ml", "fp", "fn"):
            summary[f"{tag}/{k}"] = r["metrics"]["all"]["overall"][k]
        summary[f"{tag}/fps_det_track"] = r["fps"]
        summary[f"{tag}/track_ms"] = r["track_ms"]
    run.summary.update({k: v for k, v in summary.items() if v is not None})
    print("summary:", summary, f"W&B run {run.url}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
