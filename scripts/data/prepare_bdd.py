"""Build the BDD100K YOLO dataset: 20k stratified train subset + full val split.

Runs on Kaggle (CPU job ``kaggle/data_prep``); never on the laptop. Idempotent: images that
already exist in the output are not copied again; labels and manifests are rewritten, and files
left over from a previous run with a different subset are removed.

    python -m scripts.data.prepare_bdd --config configs/data_bdd.yaml \
        --input-root /kaggle/input --out /kaggle/working/bdd_yolo

Output layout (Ultralytics):
    <out>/images/{train,val}/*.jpg   <out>/labels/{train,val}/*.txt
    <out>/data.yaml                  (no ``path`` key: resolves relative to its own folder)
    <out>/meta/frames_{train_full,train,val}.jsonl.gz   compact labels for EDA (original px)
    <out>/meta/prep_summary.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from scripts.data.bdd_yolo import (
    BDD_TO_CLASS,
    CLASSES,
    class_id,
    format_yolo_rows,
    frame_attributes,
    frame_to_yolo,
    iter_boxes,
    stratified_sample,
    stratum_counts,
)


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(root: Path, rel: str) -> Path:
    """Join ``rel`` to ``root`` unless ``rel`` is absolute."""
    p = Path(rel)
    return p if p.is_absolute() else root / p


def load_frames(path: Path) -> list[dict[str, Any]]:
    """Load BDD frames from one big JSON list or a directory of per-image JSON files."""
    print(f"loading labels from {path}", flush=True)
    if path.is_dir():
        frames = []
        for f in sorted(path.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                fr = json.load(fh)
            if not fr["name"].endswith(".jpg"):
                fr["name"] += ".jpg"
            frames.append(fr)
        return frames
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fill_attributes(
    frames: list[dict[str, Any]], root: Path, rel: str | None, max_missing: float = 0.01
) -> int:
    """Copy frame attributes (weather/timeofday/scene) from another label file if missing.

    det_20 releases on some mirrors drop the frame attributes; the 2018 release keeps them.
    Returns the number of frames that were filled in.
    """
    missing = [f for f in frames if not (f.get("attributes") or {}).get("timeofday")]
    print(f"frames without timeofday: {len(missing)}/{len(frames)}", flush=True)
    if len(missing) <= max_missing * len(frames) or not rel:
        return 0
    by_name = {f["name"]: f.get("attributes") or {} for f in load_frames(resolve(root, rel))}
    filled = 0
    for f in missing:
        attrs = by_name.get(f["name"])
        if attrs:
            f["attributes"] = {**(f.get("attributes") or {}), **attrs}
            filled += 1
    print(f"filled attributes for {filled} frames from {rel}", flush=True)
    return filled


def compact(frame: dict[str, Any]) -> dict[str, Any]:
    """Keep only what EDA needs: attributes and mapped boxes in original pixels.

    Box rows are ``[cls, x1, y1, x2, y2, occluded, truncated]``.
    """
    tod, weather = frame_attributes(frame)
    boxes = []
    for lab in iter_boxes(frame):
        cid = class_id(lab.get("category", ""))
        if cid is None:
            continue
        b = lab["box2d"]
        attrs = lab.get("attributes") or {}
        boxes.append(
            [
                cid,
                round(b["x1"], 2),
                round(b["y1"], 2),
                round(b["x2"], 2),
                round(b["y2"], 2),
                int(bool(attrs.get("occluded"))),
                int(bool(attrs.get("truncated"))),
            ]
        )
    scene = (frame.get("attributes") or {}).get("scene") or "undefined"
    return {
        "name": frame["name"],
        "timeofday": tod,
        "weather": weather,
        "scene": scene,
        "boxes": boxes,
    }


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def export_one(
    frame: dict[str, Any],
    src_dir: Path,
    img_dir: Path,
    lbl_dir: Path,
    max_side: int,
    min_size_px: float,
    jpeg_quality: int,
) -> int:
    """Copy (or resize) one image and write its YOLO label file. Returns the number of boxes."""
    import cv2

    name = frame["name"]
    src = src_dir / name
    dst = img_dir / name
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(f"cannot read image {src}")
    height, width = img.shape[:2]
    if not (dst.exists() and dst.stat().st_size > 0):
        if max(width, height) > max_side:
            scale = max_side / max(width, height)
            new = (round(width * scale), round(height * scale))
            small = cv2.resize(img, new, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        else:
            shutil.copyfile(src, dst)
    # YOLO coordinates are normalised, so they are valid for the resized image as well.
    rows = frame_to_yolo(frame, width, height, min_size_px=min_size_px)
    (lbl_dir / (Path(name).stem + ".txt")).write_text(format_yolo_rows(rows))
    return len(rows)


def export_split(
    frames: list[dict[str, Any]],
    split: str,
    src_dir: Path,
    out: Path,
    prep: dict[str, Any],
    workers: int,
) -> int:
    img_dir, lbl_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    def job(fr: dict[str, Any]) -> int:
        return export_one(
            fr,
            src_dir,
            img_dir,
            lbl_dir,
            int(prep["max_side"]),
            float(prep["min_box_px"]),
            int(prep["jpeg_quality"]),
        )

    total = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, n in enumerate(pool.map(job, frames)):
            total += n
            if i % 2000 == 0:
                print(f"  {split}: {i}/{len(frames)} images", flush=True)
    keep = {fr["name"] for fr in frames}
    keep_lbl = {Path(n).stem + ".txt" for n in keep}
    for p in img_dir.iterdir():
        if p.name not in keep:
            p.unlink()
    for p in lbl_dir.iterdir():
        if p.name not in keep_lbl:
            p.unlink()
    return total


def write_data_yaml(out: Path) -> None:
    data = {
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASSES),
        "names": dict(enumerate(CLASSES)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/data_bdd.yaml"))
    ap.add_argument("--input-root", type=Path, help="override prep.input_root")
    ap.add_argument("--out", type=Path, help="override prep.out_dir")
    ap.add_argument("--train-size", type=int, help="override prep.train_size")
    ap.add_argument("--val-size", type=int, help="cap val images (debug only); default all")
    ap.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 2) * 4))
    args = ap.parse_args(argv)

    prep = load_config(args.config)["prep"]
    root = args.input_root or Path(prep["input_root"])
    out = args.out or Path(prep["out_dir"])
    src = prep["source"]
    train_size = args.train_size if args.train_size is not None else int(prep["train_size"])
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "classes": list(CLASSES),
        "class_map": BDD_TO_CLASS,
        "seed": prep["seed"],
        "train_size": train_size,
        "max_side": prep["max_side"],
        "source": src,
    }
    for split in ("train", "val"):
        frames = load_frames(resolve(root, src[f"labels_{split}"]))
        img_src = resolve(root, src[f"images_{split}"])
        on_disk = set(os.listdir(img_src))
        n_json = len(frames)
        n_filled = fill_attributes(frames, root, src.get(f"attributes_{split}"))
        frames = [f for f in frames if f["name"] in on_disk]
        print(f"{split}: {n_json} labelled frames, {len(frames)} with an image", flush=True)
        if split == "train":
            write_jsonl_gz(
                out / "meta" / "frames_train_full.jsonl.gz", [compact(f) for f in frames]
            )
            frames = stratified_sample(
                frames,
                key=frame_attributes,
                total=train_size,
                seed=int(prep["seed"]),
                min_per_stratum=int(prep["min_per_stratum"]),
            )
        elif args.val_size:
            frames = frames[: args.val_size]
        n_boxes = export_split(frames, split, img_src, out, prep, args.workers)
        write_jsonl_gz(out / "meta" / f"frames_{split}.jsonl.gz", [compact(f) for f in frames])
        summary[split] = {
            "labelled_frames_in_source": n_json,
            "attributes_filled_from_fallback": n_filled,
            "images": len(frames),
            "boxes": n_boxes,
            "strata": {f"{t}|{w}": c for (t, w), c in sorted(stratum_counts(frames).items())},
        }
        print(f"{split}: wrote {len(frames)} images, {n_boxes} boxes", flush=True)
        del frames
    write_data_yaml(out)
    (out / "meta" / "prep_summary.json").write_text(json.dumps(summary, indent=1))
    print("done:", out, flush=True)


if __name__ == "__main__":
    main()
