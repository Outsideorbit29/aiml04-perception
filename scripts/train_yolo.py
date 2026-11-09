"""M2.1-M2.3: zero-shot baseline, YOLO11s fine-tune and evaluation on BDD100K. Runs on Kaggle.

    python -m scripts.train_yolo --config configs/yolo_bdd.yaml            # full run
    python -m scripts.train_yolo --config configs/yolo_bdd.yaml --smoke    # ~10 min check
    python -m scripts.train_yolo --config configs/yolo_bdd.yaml --resume   # next session

Steps: find the data-prep output, copy it to local disk, write the image lists (repeat-factor
sampled train list), score the COCO model zero-shot, fine-tune with a W&B checkpoint every epoch,
then evaluate best.pt and save metrics to ``out_dir`` and W&B.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

# Classes both models know; used to compare zero-shot and fine-tuned on equal terms.
SHARED = ("car", "truck", "bus", "person", "bicycle", "motorcycle", "traffic light")


# ---------- pure helpers (unit-tested) ----------
def label_classes(text: str) -> set[int]:
    """Class ids present in one YOLO label file's text."""
    return {int(line.split()[0]) for line in text.splitlines() if line.strip()}


def repeat_factors(image_classes: Sequence[set[int]], threshold: float) -> list[float]:
    """LVIS repeat factor per image: max over its classes of max(1, sqrt(t / f_c)).

    f_c is the fraction of images that contain class c.
    """
    n = len(image_classes)
    counts: dict[int, int] = {}
    for classes in image_classes:
        for c in classes:
            counts[c] = counts.get(c, 0) + 1
    class_r = {c: max(1.0, math.sqrt(threshold / (k / n))) for c, k in counts.items()}
    return [max((class_r[c] for c in classes), default=1.0) for classes in image_classes]


def repeat_list(items: Sequence[str], factors: Sequence[float], seed: int) -> list[str]:
    """Repeat each item floor(r) times, plus once more with probability frac(r) (seeded)."""
    rng = random.Random(seed)
    out: list[str] = []
    for item, r in zip(items, factors, strict=True):
        k = int(r) + (rng.random() < r - int(r))
        out.extend([item] * k)
    return out


def remap_label_text(text: str, mapping: dict[int, int]) -> str:
    """Rewrite class ids with ``mapping``; rows whose class is not in it are dropped."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if parts and int(parts[0]) in mapping:
            rows.append(" ".join([str(mapping[int(parts[0])]), *parts[1:]]))
    return "\n".join(rows) + ("\n" if rows else "")


def bdd_to_coco_ids(bdd_names: dict[int, str], name_map: dict[str, int]) -> dict[int, int]:
    return {i: name_map[n] for i, n in bdd_names.items() if n in name_map}


def label_path(img: str | Path) -> Path:
    """``<root>/images/<split>/x.jpg`` -> ``<root>/labels/<split>/x.txt`` (Ultralytics layout)."""
    p = Path(img)
    return p.parents[2] / "labels" / p.parent.name / (p.stem + ".txt")


def shared_mean(per_class: dict[str, dict[str, float]], key: str) -> float:
    vals = [per_class[n][key] for n in SHARED if n in per_class]
    return sum(vals) / len(vals) if vals else 0.0


# ---------- Kaggle-side steps ----------
def copy_tree(src: Path, dst: Path, workers: int = 32) -> int:
    """Copy a folder with many threads (the kernel-output mount is latency bound)."""
    files = [p for p in src.rglob("*") if p.is_file()]
    for d in {dst / p.parent.relative_to(src) for p in files}:
        d.mkdir(parents=True, exist_ok=True)

    def one(p: Path) -> None:
        target = dst / p.relative_to(src)
        if not target.exists():
            shutil.copyfile(p, target)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, files, chunksize=64))
    return len(files)


def stage(cfg: dict[str, Any]) -> Path:
    from scripts.data.check_data_yaml import find_dataset_root

    src = find_dataset_root(Path(cfg["search"]))
    dst = Path(cfg["stage_dir"])
    t0 = time.time()
    n = sum(
        copy_tree(src / sub, dst / sub)
        for sub in ("images/train", "images/val", "labels/train", "labels/val")
    )
    print(f"staged {n} files from {src} to {dst} in {time.time() - t0:.0f} s", flush=True)
    return dst


def write_lists(cfg: dict[str, Any], root: Path, smoke: bool) -> dict[str, Any]:
    lists = Path(cfg["lists_dir"])
    lists.mkdir(parents=True, exist_ok=True)
    train = sorted(str(p) for p in (root / "images/train").glob("*.jpg"))
    val = sorted(str(p) for p in (root / "images/val").glob("*.jpg"))
    if smoke:
        train, val = train[: cfg["smoke"]["train_limit"]], val[: cfg["smoke"]["val_limit"]]
    info: dict[str, Any] = {"train_images": len(train), "val_images": len(val)}
    if cfg["rfs"]["enabled"]:
        classes = [label_classes(label_path(p).read_text()) for p in train]
        factors = repeat_factors(classes, cfg["rfs"]["threshold"])
        train = repeat_list(train, factors, cfg["rfs"]["seed"])
        info["train_entries_rfs"] = len(train)
    (lists / "train.txt").write_text("\n".join(train) + "\n")
    (lists / "val.txt").write_text("\n".join(val) + "\n")
    print("lists:", info, flush=True)
    return info


def data_yaml(names: dict[int, str], train: str, val: str, out: Path) -> Path:
    out.write_text(yaml.safe_dump({"train": train, "val": val, "nc": len(names), "names": names}))
    return out


def summarise(metrics: Any, names: dict[int, str]) -> dict[str, Any]:
    box = metrics.box
    cls = {
        names[int(c)]: {"ap50": float(box.all_ap[i, 0]), "ap50_95": float(box.all_ap[i].mean())}
        for i, c in enumerate(box.ap_class_index)
    }
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "shared7_map50": shared_mean(cls, "ap50"),
        "shared7_map50_95": shared_mean(cls, "ap50_95"),
        "per_class": cls,
    }


def zeroshot(cfg: dict[str, Any], root: Path, bdd_names: dict[int, str], out: Path) -> dict:
    """M2.1: COCO model on BDD val, with labels remapped to COCO ids in a side folder."""
    from ultralytics import YOLO

    zs = cfg["zeroshot"]
    model = YOLO(zs["model"])
    mapping = bdd_to_coco_ids(bdd_names, zs["bdd_to_coco"])
    side = Path(cfg["stage_dir"]).parent / "bdd_zeroshot"
    (side / "images").mkdir(parents=True, exist_ok=True)
    (side / "labels/val").mkdir(parents=True, exist_ok=True)
    if not (side / "images/val").exists():
        (side / "images/val").symlink_to(root / "images/val", target_is_directory=True)
    vals = Path(cfg["lists_dir"], "val.txt").read_text().split()
    for img in vals:
        text = remap_label_text(label_path(img).read_text(), mapping)
        (side / "labels/val" / (Path(img).stem + ".txt")).write_text(text)
    zs_list = side / "val.txt"
    zs_list.write_text("\n".join(str(side / "images/val" / Path(p).name) for p in vals) + "\n")
    coco_names = {int(k): v for k, v in model.names.items()}
    yml = data_yaml(coco_names, str(zs_list), str(zs_list), out / "data_zeroshot.yaml")
    m = model.val(
        data=str(yml),
        imgsz=cfg["train"]["imgsz"],
        batch=32,
        classes=sorted(set(mapping.values())),
        project=str(out),
        name="zeroshot",
        exist_ok=True,
        plots=False,
    )
    res = summarise(m, coco_names)
    res["note"] = "COCO yolo11s; rider scored as person; traffic sign excluded"
    return res


def add_callbacks(model: Any, cfg: dict[str, Any], t_start: float) -> None:
    import wandb

    every, limit_h = cfg["wandb"]["ckpt_every"], cfg["time_limit_h"]

    def on_model_save(trainer: Any) -> None:
        # last.pt still has optimizer state here; final_eval strips it later.
        if wandb.run is None or (trainer.epoch + 1) % every:
            return
        art = wandb.Artifact(
            cfg["wandb"]["ckpt_artifact"],
            type="model",
            metadata={"epoch": trainer.epoch + 1, "run_id": wandb.run.id},
        )
        art.add_file(str(trainer.last), name="last.pt")
        wandb.run.log_artifact(art, aliases=["latest", f"epoch{trainer.epoch + 1}"])

    def on_fit_epoch_end(trainer: Any) -> None:
        hours = (time.time() - t_start) / 3600
        if hours > limit_h and trainer.epoch + 1 < trainer.epochs:
            print(
                f"time limit {limit_h} h hit after epoch {trainer.epoch + 1}; stopping. "
                "Run again with --resume.",
                flush=True,
            )
            trainer.stop = True

    model.add_callback("on_model_save", on_model_save)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def fetch_checkpoint(cfg: dict[str, Any], dest: Path) -> tuple[Path, str | None]:
    """Download the newest last.pt; also return the W&B run id it came from."""
    import wandb

    art = wandb.Api().artifact(f"{cfg['wandb']['project']}/{cfg['wandb']['ckpt_artifact']}:latest")
    path = Path(art.download(root=str(dest))) / "last.pt"
    print("resuming from", art.name, art.metadata, flush=True)
    return path, art.metadata.get("run_id")


def ensure_run(project: str, run_id: str) -> Any:
    """Ultralytics' W&B callback finishes the run at train end; reopen it for the final eval."""
    import wandb

    return wandb.run or wandb.init(project=project, id=run_id, resume="allow")


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/yolo_bdd.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    t_start = time.time()

    cfg = yaml.safe_load(args.config.read_text())
    names_raw = yaml.safe_load(Path(cfg["data_config"]).read_text())["names"]
    bdd_names = {int(k): v for k, v in names_raw.items()}
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tr = dict(cfg["train"])
    if args.smoke:
        tr.update(epochs=cfg["smoke"]["epochs"], imgsz=cfg["smoke"]["imgsz"])
        tr["name"] += "-smoke"
        for k in ("ckpt_artifact", "best_artifact"):  # keep smoke files out of the real lineage
            cfg["wandb"][k] += "-smoke"

    import wandb
    from ultralytics import YOLO, settings

    settings.update({"wandb": True})  # Ultralytics logs metrics into the run started here
    project = cfg["wandb"]["project"]
    ckpt, run_id = (None, None)
    if args.resume:
        ckpt, run_id = fetch_checkpoint(cfg, Path("/kaggle/temp/ckpt"))
    run = wandb.init(
        project=project,
        name=tr["name"],
        id=run_id,
        resume="allow",
        job_type="smoke" if args.smoke else "train",
        config={"train": tr, "rfs": cfg["rfs"]},
        allow_val_change=True,
    )
    run_id = run.id

    root = stage(cfg)
    lists = write_lists(cfg, root, args.smoke)
    run.config.update({"lists": lists}, allow_val_change=True)
    yml = data_yaml(
        bdd_names,
        str(Path(cfg["lists_dir"], "train.txt")),
        str(Path(cfg["lists_dir"], "val.txt")),
        out / "data_bdd_train.yaml",
    )
    results: dict[str, Any] = {"lists": lists}

    if cfg["zeroshot"]["enabled"] and not args.resume:
        results["zeroshot"] = zeroshot(cfg, root, bdd_names, out)
        print("zeroshot:", json.dumps(results["zeroshot"], indent=1), flush=True)
        run.summary.update(
            {f"zeroshot/{k}": v for k, v in results["zeroshot"].items() if isinstance(v, float)}
        )

    if args.resume:
        model = YOLO(str(ckpt))
        add_callbacks(model, cfg, t_start)
        model.train(resume=True)
    else:
        model = YOLO(tr.pop("model"))
        add_callbacks(model, cfg, t_start)
        model.train(data=str(yml), project=str(out / "runs"), exist_ok=True, **tr)

    trainer = model.trainer
    run = ensure_run(project, run_id)
    best = Path(trainer.best)
    m = YOLO(str(best)).val(
        data=str(yml),
        imgsz=tr["imgsz"],
        batch=32,
        project=str(out),
        name="final_val",
        exist_ok=True,
        plots=True,
    )
    ft = summarise(m, bdd_names)
    ft["epochs_done"] = trainer.epoch + 1
    early = bool(getattr(trainer.stopper, "possible_stop", False))
    ft["finished"] = bool(trainer.epoch + 1 >= trainer.epochs or early)
    results["finetuned"] = ft
    run.summary.update({f"final/{k}": v for k, v in ft.items() if isinstance(v, float)})
    shutil.copy(best, out / "best.pt")
    art = wandb.Artifact(
        cfg["wandb"]["best_artifact"],
        type="model",
        metadata={k: v for k, v in ft.items() if k != "per_class"},
    )
    art.add_file(str(best), name="best.pt")
    run.log_artifact(art, aliases=["latest", "smoke" if args.smoke else "candidate"])

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("results:", json.dumps(results, indent=1), flush=True)
    print(f"total time {(time.time() - t_start) / 60:.1f} min", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
