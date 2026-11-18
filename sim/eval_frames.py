"""M5.6 sim eval: the perception stack on the recorded CARLA frames. Runs on Kaggle (sim_eval).

    python -m sim.eval_frames --config configs/carla_eval.yaml            # all 3000 frames
    python -m sim.eval_frames --config configs/carla_eval.yaml --smoke    # every 30th frame

ROS-free and CARLA-free: it reads the files written by sim/record_dataset.py (Kaggle dataset
``anish2906/aiml04-carla-frames``): images/<town>_<idx>.jpg, labels/*.txt (YOLO, BDD ids),
gt/*.json (boxes, GT distance = nearest 3D-box vertex depth, visibility, weather), depth/*.png
(GT depth, uint16 cm, every 10th frame) and meta_<town>.json (intrinsics, weather ranges).
The upload may keep the subfolders as .zip files; everything is staged flat to ``stage_dir``.

1. Detection (Ultralytics ``model.val``): the fine-tuned YOLO11s (W&B artifact) on all frames,
   mAP@0.5 and mAP@0.5:0.95 over the classes that have CARLA GT (car, truck, bus, person,
   bicycle, motorcycle; Ultralytics averages only over classes present in the labels), per
   class and per weather; then the COCO yolo11s zero-shot with labels remapped to COCO ids.
2. Distance, with the live stack's settings (configs/pipeline_carla.yaml, torch backend):
   per frame the Detector and MiDaS-small run once, then ``DistanceFusion.measure`` gives the
   geometric, aligned-MiDaS and fused distance of every road-user box (stateless, no Kalman:
   frames are 4 Hz samples). Two box sources, both scored against the GT distance:
   ``det``  detections with score >= ``det_conf`` matched to GT (class-agnostic IoU), i.e. what
            the live stack sees; unmatched GT counts against ``det_recall``.
   ``gt``   the GT boxes themselves (score 1), which isolates distance from detection and is
            comparable to the M4.4 KITTI table.
   ``fused`` uses leave-one-out alignment (as M4.4); ``fused_live`` is the live call (anchors'
   MiDaS fitted with themselves). Also the camera height implied by the GT (checks the
   flat-ground model: configs/camera.yaml carla height_m / pitch).
3. MiDaS vs GT depth on the depth PNGs, with a per-frame least-squares affine fit in inverse
   depth (oracle alignment, the usual relative-depth eval): AbsRel, RMSE, delta < 1.25.

Outputs in ``out_dir``: results.json, sim_eval.md (tables), per_object.jsonl, snapshots/*.jpg
(GT green with GT distance, matched detections blue with fused distance, misses red, unmatched
detections yellow), Ultralytics val folders; one W&B run (group ``sim_eval``).
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import time
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from perception.detection.processing import box_iou
from perception.types import Detection

PARTS = {"images": (".jpg",), "labels": (".txt",), "gt": (".json",), "depth": (".png",)}
METHODS = ("geometric", "midas_small", "fused", "fused_live")
COLOR_GT = (0, 200, 0)  # BGR
COLOR_MATCH = (255, 160, 0)
COLOR_MISS = (0, 0, 255)
COLOR_FP = (0, 230, 255)


# ---------------------------------------------------------------- pure helpers (unit-tested)
def find_meta(search: str | Path, depth: int) -> Path:
    """First ``meta_*.json`` 1..``depth`` levels below ``search`` (shallow globs only)."""
    for d in range(1, depth + 1):
        hits = sorted(glob.glob(str(Path(search, *(["*"] * d), "meta_*.json"))))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"no meta_*.json under {search} (depth {depth})")


def locate_parts(root: Path) -> dict[str, Path]:
    """Each dataset part as a non-empty folder ``root/<part>`` or a file ``root/<part>.zip``
    (``kaggle datasets create --dir-mode zip`` uploads folders as zips; Kaggle may or may not
    extract them)."""
    out = {}
    for part in PARTS:
        folder, zipped = root / part, root / f"{part}.zip"
        if folder.is_dir() and any(folder.iterdir()):
            out[part] = folder
        elif zipped.is_file():
            out[part] = zipped
    return out


def _extract(src: Path, dst: Path, suffixes: tuple[str, ...]) -> int:
    n = 0
    with zipfile.ZipFile(src) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name.lower().endswith(suffixes):
                continue
            with zf.open(info) as fin, open(dst / name, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            n += 1
    return n


def stage_part(src: Path, dst: Path, suffixes: Sequence[str], workers: int = 16) -> int:
    """Copy (folder) or extract (zip) every file with one of ``suffixes`` flat into ``dst``.

    Files are matched by suffix at any depth, so ``images/x.jpg``, ``images/images/x.jpg`` and
    a zip holding either stage the same; zips inside a folder are extracted too. Returns the
    number of files."""
    dst.mkdir(parents=True, exist_ok=True)
    suffixes = tuple(s.lower() for s in suffixes)
    if src.is_file():
        return _extract(src, dst, suffixes) if zipfile.is_zipfile(src) else 0
    found = [p for p in src.rglob("*") if p.is_file()]
    files = [p for p in found if p.suffix.lower() in suffixes]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda p: shutil.copyfile(p, dst / p.name), files))
    zips = [p for p in found if p.suffix.lower() == ".zip" and zipfile.is_zipfile(p)]
    return len(files) + sum(_extract(z, dst, suffixes) for z in zips)


def spread_subset(items: Sequence[Any], every_n: int) -> list[Any]:
    """Every ``every_n``-th item starting at 0 (spans all weather blocks of the recording)."""
    return list(items[:: max(1, int(every_n))])


def match_pairs(
    gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_scores: np.ndarray, iou_thr: float
) -> list[tuple[int, int, float]]:
    """Class-agnostic greedy matching by descending score: ``[(gt_idx, pred_idx, iou)]``.

    Each prediction takes the unmatched GT box with the highest IoU >= ``iou_thr``."""
    gt_boxes = np.asarray(gt_boxes, np.float32).reshape(-1, 4)
    pred_boxes = np.asarray(pred_boxes, np.float32).reshape(-1, 4)
    if not len(gt_boxes) or not len(pred_boxes):
        return []
    iou = box_iou(pred_boxes, gt_boxes)
    taken = np.zeros(len(gt_boxes), bool)
    out = []
    for p in np.argsort(-np.asarray(pred_scores, float), kind="stable"):
        cand = np.flatnonzero(~taken & (iou[p] >= iou_thr))
        if cand.size:
            g = int(cand[np.argmax(iou[p, cand])])
            taken[g] = True
            out.append((g, int(p), float(iou[p, g])))
    return out


def gt_detections(objects: Sequence[dict[str, Any]], name_to_id: dict[str, int]) -> list[Detection]:
    """GT objects from gt/<frame>.json as score-1 Detections (for the GT-box distance path)."""
    return [
        Detection(
            np.asarray(o["bbox_xyxy"], np.float32),
            1.0,
            int(name_to_id.get(o["cls_name"], -1)),
            str(o["cls_name"]),
        )
        for o in objects
    ]


def implied_height(distance_m: float, v_bottom: float, cy: float, fy: float) -> float | None:
    """Camera height that puts a ground point at image row ``v_bottom`` at ``distance_m``
    (flat ground, zero pitch): ``Z * (v - cy) / fy``. None above the horizon."""
    if v_bottom <= cy:
        return None
    return float(distance_m * (v_bottom - cy) / fy)


def fit_inverse_affine(m: np.ndarray, z: np.ndarray) -> tuple[float, float] | None:
    """Least-squares ``1/z = s*m + t`` (relative inverse depth -> metric). None if degenerate."""
    m, z = np.asarray(m, float).ravel(), np.asarray(z, float).ravel()
    ok = np.isfinite(m) & np.isfinite(z) & (z > 0)
    if ok.sum() < 2 or np.ptp(m[ok]) <= 0:
        return None
    a = np.stack([m[ok], np.ones(int(ok.sum()))], axis=1)
    (s, t), *_ = np.linalg.lstsq(a, 1.0 / z[ok], rcond=None)
    return float(s), float(t)


def depth_map_metrics(
    inv_rel: np.ndarray, gt_m: np.ndarray, min_m: float, max_m: float, stride: int = 1
) -> dict[str, float] | None:
    """Oracle-aligned MiDaS vs GT depth on pixels with ``min_m < gt < max_m``: the affine fit
    in inverse depth uses every ``stride``-th pixel; scores use all valid pixels whose aligned
    inverse depth is positive. Returns AbsRel, RMSE (m), delta < 1.25, coverage, s, t."""
    if inv_rel.shape != gt_m.shape:
        raise ValueError(f"depth map {inv_rel.shape} does not match GT {gt_m.shape}")
    valid = (gt_m > min_m) & (gt_m < max_m) & np.isfinite(inv_rel)
    sub = np.zeros_like(valid)
    sub[::stride, ::stride] = True
    fit = fit_inverse_affine(inv_rel[valid & sub], gt_m[valid & sub])
    if fit is None:
        return None
    s, t = fit
    inv = s * inv_rel[valid] + t
    pos = inv > 1e-6
    if not pos.any():
        return None
    pred, gt = 1.0 / inv[pos], gt_m[valid][pos]
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "absrel": float(np.mean(np.abs(pred - gt) / gt)),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "coverage": float(pos.mean()),
        "s": s,
        "t": t,
    }


def mean_rows(rows: Sequence[dict[str, float]], keys: Sequence[str]) -> dict[str, Any]:
    """Mean of each key over per-frame metric dicts (frames are weighted equally)."""
    out: dict[str, Any] = {"frames": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


# ---------------------------------------------------------------- drawing
def _put(img: np.ndarray, text: str, x: int, y: int, color: tuple) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    y = max(th + 2, y)
    cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y + 1), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_snapshot(
    img: np.ndarray,
    objects: list[dict[str, Any]],
    dets: list[Detection],
    pairs: list[tuple[int, int, float]],
    fused: list[float | None],
    title: str,
) -> np.ndarray:
    """GT green (GT distance), matched detections blue (fused distance), missed GT red,
    unmatched detections yellow."""
    vis = img.copy()
    gmatch = {g: p for g, p, _ in pairs}
    pmatched = {p for _, p, _ in pairs}
    for gi, o in enumerate(objects):
        x1, y1, x2, y2 = (int(round(v)) for v in o["bbox_xyxy"])
        color = COLOR_GT if gi in gmatch else COLOR_MISS
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        _put(vis, f"{o['cls_name']} GT {o['distance_m']:.1f}m", x1, y1 - 2, color)
    for pi, d in enumerate(dets):
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox_xyxy)
        color = COLOR_MATCH if pi in pmatched else COLOR_FP
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        z = fused[pi]
        _put(vis, f"{d.cls_name} {d.score:.2f} " + (f"{z:.1f}m" if z else "-"), x1, y2 + 12,
             color)
    _put(vis, title, 4, 14, (255, 255, 255))
    return vis


# ---------------------------------------------------------------- Kaggle-side steps
def stage_dataset(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Find the mount, stage images/labels/gt/depth flat into ``stage_dir``.
    Returns (stage root, meta, layout info)."""
    meta_path = find_meta(cfg["search"], int(cfg["search_depth"]))
    root = meta_path.parent
    print(f"dataset root {root}: {sorted(p.name for p in root.iterdir())}", flush=True)
    parts = locate_parts(root)
    dst = Path(cfg["stage_dir"])
    t0 = time.time()
    layout: dict[str, Any] = {"root": str(root), "meta": meta_path.name, "parts": {}}
    for part, src in parts.items():
        n = stage_part(src, dst / part, PARTS[part])
        layout["parts"][part] = {"source": str(src), "zip": src.is_file(), "files": n}
        print(f"staged {part}: {n} files from {src}", flush=True)
    missing = [p for p in ("images", "labels", "gt") if p not in parts]
    if missing:
        raise FileNotFoundError(f"dataset parts {missing} not found under {root}")
    layout["stage_s"] = time.time() - t0
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return dst, meta, layout


def detection_eval(
    model_path: str, names: dict[int, str], paths: list[Path], weathers: dict[str, str],
    cfg: dict[str, Any], tag: str, classes: list[int] | None = None, plots: bool = False,
) -> dict[str, Any]:
    """Ultralytics val on ``paths`` (+ per weather). ``classes`` filters predictions."""
    from ultralytics import YOLO

    from scripts.eval_detection import per_class_rows, run_val
    from scripts.train_yolo import summarise

    model = YOLO(model_path)
    d = cfg["detection"]
    vcfg = {"out_dir": cfg["out_dir"], "stage_dir": cfg["stage_dir"],
            "val": {"imgsz": d["imgsz"], "batch": d["batch"]}}
    kw = {"classes": classes} if classes else {}

    def val(sub: list[Path], name: str, with_plots: bool) -> Any:
        if not kw:
            return run_val(model, names, sub, vcfg, name, with_plots)
        from scripts.train_yolo import data_yaml

        lst = Path(cfg["stage_dir"]).parent / "eval_lists" / f"{name}.txt"
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_text("\n".join(str(p) for p in sub) + "\n")
        yml = data_yaml(names, str(lst), str(lst), lst.with_suffix(".yaml"))
        return model.val(data=str(yml), imgsz=d["imgsz"], batch=d["batch"],
                         project=cfg["out_dir"], name=f"val_{name}", exist_ok=True,
                         plots=with_plots, verbose=False, **kw)

    m = val(paths, f"{tag}_all", plots)
    s = summarise(m, names)
    res: dict[str, Any] = {
        "images": len(paths),
        "all": {"map50": s["map50"], "map50_95": s["map50_95"],
                "precision": float(m.box.mp), "recall": float(m.box.mr)},
        "per_class": per_class_rows(m, names),
        "per_weather": {},
    }
    if d.get("per_weather", True):
        for w in sorted(set(weathers.values())):
            sub = [p for p in paths if weathers.get(p.stem) == w]
            if sub:
                mw = val(sub, f"{tag}_{w}", False)
                res["per_weather"][w] = {
                    "images": len(sub), "map50": float(mw.box.map50),
                    "map50_95": float(mw.box.map),
                    "per_class_ap50": {r["class"]: r["ap50"] for r in per_class_rows(mw, names)},
                }
    return res


def zeroshot_eval(
    stage: Path, paths: list[Path], weathers: dict[str, str], names: dict[int, str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """COCO yolo11s on the same frames, CARLA labels remapped to COCO ids in a side folder."""
    from scripts.train_yolo import bdd_to_coco_ids, get_weights, remap_label_text

    zs = cfg["detection"]["zeroshot"]
    weights = get_weights(zs["model"], Path(cfg["weights_dir"]))
    mapping = bdd_to_coco_ids(names, zs["bdd_to_coco"])
    side = stage.parent / "carla_coco"
    (side / "labels").mkdir(parents=True, exist_ok=True)
    if not (side / "images").exists():
        (side / "images").symlink_to(stage / "images", target_is_directory=True)
    for p in paths:
        text = (stage / "labels" / f"{p.stem}.txt").read_text(encoding="utf-8")
        (side / "labels" / f"{p.stem}.txt").write_text(remap_label_text(text, mapping))
    from ultralytics import YOLO

    coco_names = {int(k): v for k, v in YOLO(weights).names.items()}
    side_paths = [side / "images" / p.name for p in paths]
    res = detection_eval(weights, coco_names, side_paths, weathers, cfg, "zeroshot",
                         classes=sorted(set(mapping.values())))
    res["note"] = "COCO yolo11s; CARLA labels remapped to COCO ids (bdd_to_coco)"
    return res


def summarize_distance(
    records: list[dict[str, Any]], bins: list[list[float]]
) -> dict[str, Any]:
    from scripts.eval_depth import summarize

    out: dict[str, Any] = {"overall": summarize(records, METHODS, "gt", bins)}
    by_w: dict[str, list[dict]] = defaultdict(list)
    by_c: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_w[r["weather"]].append(r)
        by_c["person" if r["cls"] == "person" else "vehicle"].append(r)
    out["per_weather"] = {w: summarize(rs, METHODS, "gt", bins) for w, rs in sorted(by_w.items())}
    out["per_group"] = {c: summarize(rs, METHODS, "gt", bins) for c, rs in sorted(by_c.items())}
    return out


def _f(v: Any, nd: int = 3) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def markdown(results: dict[str, Any]) -> str:
    lines = ["# M5.6 sim eval (CARLA Town01)", ""]
    det = results.get("detection", {})
    for tag in ("finetuned", "zeroshot"):
        if tag not in det:
            continue
        r = det[tag]
        lines += [f"## Detection: {tag} ({r['images']} frames)", "",
                  f"mAP@0.5 {_f(r['all']['map50'])}, mAP@0.5:0.95 {_f(r['all']['map50_95'])}, "
                  f"P {_f(r['all']['precision'])}, R {_f(r['all']['recall'])}", "",
                  "| class | P | R | AP50 | AP50-95 |", "|---|---|---|---|---|"]
        lines += [f"| {c['class']} | {_f(c['precision'])} | {_f(c['recall'])} | "
                  f"{_f(c['ap50'])} | {_f(c['ap50_95'])} |" for c in r["per_class"]]
        lines += ["", "| weather | frames | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|"]
        lines += [f"| {w} | {v['images']} | {_f(v['map50'])} | {_f(v['map50_95'])} |"
                  for w, v in r["per_weather"].items()]
        lines.append("")
    for src in ("det", "gt"):
        dist = results.get("distance", {}).get(src)
        if not dist:
            continue
        lines += [f"## Distance, {src} boxes", "",
                  "| method | bin | n | coverage | AbsRel | RMSE (m) | bias |",
                  "|---|---|---|---|---|---|---|"]
        for m in METHODS:
            for b, e in dist["overall"].items():
                st = e["methods"][m]
                lines.append(f"| {m} | {b} | {st['n']} | {_f(st['coverage'], 2)} | "
                             f"{_f(st['absrel'])} | {_f(st['rmse'], 2)} | {_f(st['bias'])} |")
        lines += ["", "| weather | n | fused AbsRel | fused bias | geometric AbsRel | coverage "
                  "fused |", "|---|---|---|---|---|---|"]
        for w, s in dist["per_weather"].items():
            a = s["all"]["methods"]
            lines.append(f"| {w} | {a['fused']['n']} | {_f(a['fused']['absrel'])} | "
                         f"{_f(a['fused']['bias'])} | {_f(a['geometric']['absrel'])} | "
                         f"{_f(a['fused']['coverage'], 2)} |")
        lines.append("")
    mg = results.get("midas_gt")
    if mg:
        lines += ["## MiDaS-small vs GT depth (per-frame oracle affine fit, inverse depth)", "",
                  "| weather | frames | AbsRel | RMSE (m) | delta<1.25 |", "|---|---|---|---|---|"]
        for w, v in {"all": mg["all"], **mg["per_weather"]}.items():
            lines.append(f"| {w} | {v['frames']} | {_f(v['absrel'])} | {_f(v['rmse'], 2)} | "
                         f"{_f(v['delta1'])} |")
    return "\n".join(lines) + "\n"


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) or v is None:
            out[key] = v
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/carla_eval.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    t_start = time.time()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = Path(cfg["out_dir"]) / ("smoke" if args.smoke else "full")
    cfg["out_dir"] = str(out)
    out.mkdir(parents=True, exist_ok=True)

    import wandb

    from perception.depth import DepthEstimator, load_camera
    from perception.detection import Detector
    from perception.detection.detector import load_names
    from perception.fusion import DistanceFusion
    from perception.pipeline import load_pipeline_config, stage_configs
    from scripts.eval_detection import fetch_weights
    from scripts.eval_lanes import git_sha, gpu_name

    w = cfg["wandb"]
    run = wandb.init(
        project=w["project"], entity=w.get("entity"), group=w["group"],
        job_type="smoke" if args.smoke else "eval",
        name=w["name"] + ("-smoke" if args.smoke else ""),
        config={**cfg, "git_sha": git_sha(), "gpu": gpu_name()},
    )
    weights = fetch_weights(run, w["weights_artifact"], Path(cfg["weights_dir"]))
    stage, meta, layout = stage_dataset(cfg)
    names = load_names({"data_config": cfg["data_config"]}) or {}
    name_to_id = {v: k for k, v in names.items()}

    paths = sorted((stage / "images").glob("*.jpg"))
    if args.smoke:
        paths = spread_subset(paths, int(cfg["smoke"]["every_n"]))
    gts = {p.stem: json.loads((stage / "gt" / f"{p.stem}.json").read_text("utf-8"))
           for p in paths}
    weathers = {k: g.get("weather", "unknown") for k, g in gts.items()}
    counts: dict[str, int] = defaultdict(int)
    for g in gts.values():
        for o in g["objects"]:
            counts[o["cls_name"]] += 1
    results: dict[str, Any] = {
        "mode": "smoke" if args.smoke else "full", "git_sha": git_sha(), "gpu": gpu_name(),
        "weights": w["weights_artifact"], "layout": layout,
        "dataset": {"frames": len(paths), "boxes_per_class": dict(counts),
                    "empty_frames": sum(not g["objects"] for g in gts.values()),
                    "frames_per_weather": {x: sum(v == x for v in weathers.values())
                                           for x in sorted(set(weathers.values()))},
                    "meta_intrinsics": meta.get("intrinsics"), "town": meta.get("town")},
    }
    print("dataset:", json.dumps(results["dataset"], indent=1), flush=True)

    # 1. Detection mAP.
    t0 = time.time()
    results["detection"] = {
        "finetuned": detection_eval(str(weights), names, paths, weathers, cfg, "finetuned",
                                    plots=True)
    }
    if cfg["detection"]["zeroshot"].get("enabled", True):
        results["detection"]["zeroshot"] = zeroshot_eval(stage, paths, weathers, names, cfg)
    results["detection"]["seconds"] = time.time() - t0
    print("detection:", json.dumps({k: v["all"] for k, v in results["detection"].items()
                                    if isinstance(v, dict)}, indent=1), flush=True)

    # 2. Distance with the live stack's detector / MiDaS / fusion settings (torch backend).
    pcfg = load_pipeline_config(cfg["pipeline_config"])
    # Only detector, MiDaS and fusion are built. The lane detector is never built (lanes play no
    # part in distance), so lanes_carla.yaml's laptop-only yolop .engine path is never opened.
    # `backend` (torch) switches detector + depth only; the detector gets the W&B best.pt.
    sc = stage_configs(pcfg, cfg["backend"], cfg["camera"])
    sc["detector"]["weights"] = str(weights)
    detector = Detector(sc["detector"])
    est = DepthEstimator(sc["depth"])
    cam = load_camera(pcfg["camera_config"], cfg["camera"])
    intr = meta.get("intrinsics") or {}
    if cfg.get("camera_from_meta") and intr:
        cfg_k = (cam.fx, cam.fy, cam.cx, cam.cy)
        p = np.array([[intr["fx"], 0, intr["cx"], 0], [0, intr["fy"], intr["cy"], 0],
                      [0, 0, 1, 0]], float)
        cam = cam.with_intrinsics(p, (cam.height, cam.width))
        if not np.allclose(cfg_k, (cam.fx, cam.fy, cam.cx, cam.cy), atol=0.5):
            print(f"WARNING: meta intrinsics {intr} differ from camera.yaml {cfg_k}", flush=True)
    fusion = DistanceFusion(sc["depth"], cam)
    results["setup"] = {"camera": repr(cam), "detector": {k: sc["detector"].get(k) for k in (
        "backend", "imgsz", "conf", "iou", "agnostic_nms")},
        "depth": {"model": est.name, "input_hw": est.mcfg["input_hw"],
                  "backend": type(est.backend).__name__}}
    dcfg, mcfg = cfg["distance"], cfg["midas_gt"]
    det_conf, iou_thr = float(dcfg["det_conf"]), float(dcfg["iou_match"])
    loo = bool(dcfg["leave_one_out"])
    snap_ids = {paths[i].stem for i in np.linspace(0, len(paths) - 1,
                                                   min(int(cfg["snapshots"]["count"]),
                                                       len(paths))).round().astype(int)}
    (out / "snapshots").mkdir(exist_ok=True)
    recs: dict[str, list[dict]] = {"det": [], "gt": []}
    det_hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # bin -> [matched, total]
    heights: list[float] = []
    depth_rows: list[dict] = []
    timing: dict[str, list[float]] = defaultdict(list)
    bins = dcfg["bins"]
    from scripts.eval_depth import bin_label

    def measure(dets: list[Detection], hw: tuple[int, int], inv: np.ndarray) -> tuple[list, list]:
        m_loo, _ = fusion.measure(dets, hw, inv, leave_one_out=loo)
        m_live, _ = fusion.measure(dets, hw, inv)
        return m_loo, m_live

    def rec(stem: str, o: dict, ms: Any, ml: Any, extra: dict) -> dict:
        return {
            "frame": stem, "weather": weathers[stem], "cls": o["cls_name"], "id": o["id"],
            "gt": float(o["distance_m"]), "visible_frac": o.get("visible_frac"),
            "geometric": ms.geometric[0] if ms.geometric else None,
            "midas_small": ms.midas[0] if ms.midas else None,
            "fused": ms.fused[0] if ms.fused else None,
            "fused_live": ml.fused[0] if ml.fused else None,
            "anchor": bool(ms.anchor), "align_mode": ms.align_mode, **extra,
        }

    t0 = time.time()
    for k, p in enumerate(paths):
        img = cv2.imread(str(p))
        hw = (int(img.shape[0]), int(img.shape[1]))
        objs = [o for o in gts[p.stem]["objects"] if o["cls_name"] in fusion.classes]
        t1 = time.perf_counter()
        dets = [d for d in detector.detect(img) if d.score >= det_conf and fusion.keeps(d)]
        t2 = time.perf_counter()
        inv = est.estimate(img)
        t3 = time.perf_counter()
        if k >= 5:
            timing["detect_ms"].append(1e3 * (t2 - t1))
            timing["depth_ms"].append(1e3 * (t3 - t2))
        # detection boxes, matched to GT
        d_loo, d_live = measure(dets, hw, inv)
        gboxes = np.array([o["bbox_xyxy"] for o in objs], np.float32).reshape(-1, 4)
        pairs = match_pairs(gboxes, np.array([d.bbox_xyxy for d in dets]).reshape(-1, 4),
                            np.array([d.score for d in dets]), iou_thr)
        for g, pi, iou in pairs:
            recs["det"].append(rec(p.stem, objs[g], d_loo[pi], d_live[pi], {
                "iou": iou, "score": dets[pi].score, "det_cls": dets[pi].cls_name}))
        matched = {g for g, _, _ in pairs}
        for gi, o in enumerate(objs):
            b = bin_label(float(o["distance_m"]), bins)
            for key in ((b, "all") if b else ()):
                det_hits[key][0] += gi in matched
                det_hits[key][1] += 1
        # GT boxes as detections
        g_loo, g_live = measure(gt_detections(objs, name_to_id), hw, inv)
        for o, ms, ml in zip(objs, g_loo, g_live, strict=True):
            recs["gt"].append(rec(p.stem, o, ms, ml, {}))
            v = float(o["bbox_xyxy"][3])
            z = float(o["distance_m"])
            hc = dcfg["height_check"]
            if v < hw[0] - 1 - fusion.bottom_margin and hc["min_m"] <= z <= hc["max_m"]:
                h = implied_height(z, v, cam.cy, cam.fy)
                if h is not None:
                    heights.append(h)
        # MiDaS vs GT depth
        dpng = stage / "depth" / f"{p.stem}.png"
        if dpng.exists():
            gt_cm = cv2.imread(str(dpng), cv2.IMREAD_UNCHANGED)
            mm = depth_map_metrics(inv, gt_cm.astype(np.float32) / 100.0, float(mcfg["min_m"]),
                                   float(mcfg["max_m"]), int(mcfg["stride"]))
            if mm is not None:
                depth_rows.append({"frame": p.stem, "weather": weathers[p.stem], **mm})
        if p.stem in snap_ids:
            fz = [m.fused[0] if m.fused else None for m in d_loo]
            title = f"{p.stem} {weathers[p.stem]}  green GT / blue det (fused) / red miss"
            cv2.imwrite(str(out / "snapshots" / f"{p.stem}.jpg"),
                        draw_snapshot(img, objs, dets, pairs, fz, title),
                        [cv2.IMWRITE_JPEG_QUALITY, int(cfg["snapshots"]["jpg_quality"])])
        if (k + 1) % 500 == 0:
            print(f"distance: {k + 1}/{len(paths)} frames ({time.time() - t0:.0f} s)", flush=True)

    results["distance"] = {src: summarize_distance(rs, bins) for src, rs in recs.items()}
    results["distance"]["det_recall"] = {
        b: {"matched": m, "gt": n, "recall": m / n if n else None} for b, (m, n) in det_hits.items()
    }
    results["distance"]["implied_camera_height_m"] = {
        "n": len(heights), "configured": cam.height_m,
        "median": float(np.median(heights)) if heights else None,
        "p10": float(np.percentile(heights, 10)) if heights else None,
        "p90": float(np.percentile(heights, 90)) if heights else None,
    }
    results["distance"]["timing_ms"] = {k: float(np.mean(v)) for k, v in timing.items() if v}
    keys = ("absrel", "rmse", "delta1", "coverage")
    by_w: dict[str, list[dict]] = defaultdict(list)
    for r in depth_rows:
        by_w[r["weather"]].append(r)
    results["midas_gt"] = {
        "all": mean_rows(depth_rows, keys),
        "per_weather": {x: mean_rows(rs, keys) for x, rs in sorted(by_w.items())},
        "fit": "per frame least squares 1/Z = s*m + t on GT pixels "
               f"{mcfg['min_m']}-{mcfg['max_m']} m (oracle alignment)",
    }
    results["minutes"] = (time.time() - t_start) / 60
    results["wandb_url"] = run.url

    with open(out / "per_object.jsonl", "w", encoding="utf-8") as fh:
        for src, rs in recs.items():
            for r in rs:
                fh.write(json.dumps({"boxes": src, **r}) + "\n")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str),
                                      encoding="utf-8")
    md = markdown(results)
    (out / "sim_eval.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    print("implied camera height:", results["distance"]["implied_camera_height_m"], flush=True)
    print("det recall:", json.dumps(results["distance"]["det_recall"]), flush=True)

    summary = {}
    for tag, r in results["detection"].items():
        if isinstance(r, dict):
            summary[f"det_{tag}/map50"] = r["all"]["map50"]
            summary[f"det_{tag}/map50_95"] = r["all"]["map50_95"]
            for wname, v in r["per_weather"].items():
                summary[f"det_{tag}_weather/{wname}/map50"] = v["map50"]
    for src in ("det", "gt"):
        for b, e in results["distance"][src]["overall"].items():
            for m in METHODS:
                st = e["methods"][m]
                for key in ("absrel", "rmse", "bias", "coverage"):
                    summary[f"dist_{src}/{m}/{b}/{key}"] = st[key]
    summary.update(flat(results["midas_gt"]["all"], "midas_gt/"))
    summary.update(flat(results["distance"]["implied_camera_height_m"], "camera_height/"))
    run.summary.update(summary)
    log: dict[str, Any] = {}
    for tag, r in results["detection"].items():
        if isinstance(r, dict) and r["per_class"]:
            rows = r["per_class"]
            log[f"per_class_{tag}"] = wandb.Table(columns=list(rows[0]),
                                                  data=[list(x.values()) for x in rows])
    snaps = sorted((out / "snapshots").glob("*.jpg"))
    if snaps:
        log["snapshots"] = [wandb.Image(str(s), caption=s.stem) for s in snaps]
    for png in sorted((out / "val_finetuned_all").glob("*.png")):
        log[f"plots/{png.stem}"] = wandb.Image(str(png))
    run.log(log)
    run.save(str(out / "results.json"), base_path=str(out))
    run.save(str(out / "sim_eval.md"), base_path=str(out))
    print(f"wrote {out}; W&B {run.url}; {results['minutes']:.1f} min", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
