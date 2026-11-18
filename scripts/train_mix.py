"""M5.9: fine-tune the BDD YOLO11s for a few epochs on a BDD + CARLA mix. Runs on Kaggle.

    python -m scripts.train_mix --config configs/yolo_mix.yaml            # full run
    python -m scripts.train_mix --config configs/yolo_mix.yaml --smoke    # ~15 min check
    python -m scripts.train_mix --config configs/yolo_mix.yaml --resume   # next session

Steps (kaggle/train_mix/runner.py):
1. Stage BDD (data-prep kernel output, as scripts/train_yolo.py) and the CARLA frames
   (``sim.eval_frames.stage_dataset``).
2. Split CARLA by contiguous blocks: in every weather segment one ``block_len`` block is
   CARLA-val, with ``guard`` frames dropped on both sides (consecutive frames are near-duplicates).
3. Pseudo-label rider / traffic light / traffic sign on CARLA-train with the base model (CARLA has
   no GT for them; unlabeled they would be trained as background).
4. Train list: BDD train (repeat-factor sampled as M2.2) + CARLA-train frames with boxes repeated
   to ``carla_share`` of the list + a fraction of the empty frames once. Val list: BDD val.
5. Fine-tune from W&B ``yolo11s-bdd-best`` (checkpoint artifact every epoch, ``--resume``).
6. Evaluate base and new model on BDD val (9 classes) and on CARLA-val (6 classes, per weather,
   ``sim.eval_frames.detection_eval``, the M5.6 scoring).
7. Log the weights as ``yolo11s-mix-best`` and the ONNX export (``deploy.export.export_yolo``,
   opset 17, 1x3x640x640) as ``yolo11s-mix-onnx``. Outputs: ``out_dir``/results.json + mix_eval.md.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from scripts.train_yolo import (
    add_callbacks,
    data_yaml,
    ensure_run,
    label_classes,
    label_path,
    repeat_factors,
    repeat_list,
    stage,
    summarise,
)

CARLA_CLASSES = ("car", "truck", "bus", "person", "bicycle", "motorcycle")


# ---------- pure helpers (unit-tested) ----------
def frame_index(stem: str) -> int:
    """``Town01_00042`` -> 42 (the recorder's running frame counter)."""
    return int(stem.rsplit("_", 1)[-1])


def weather_segments(weather_log: Sequence[dict[str, Any]], n_frames: int) -> list[tuple]:
    """``[(weather, start, end)]`` from meta ``weathers`` (``first_index`` of each change)."""
    log = sorted(weather_log, key=lambda w: int(w["first_index"]))
    if not log:
        return [("unknown", 0, n_frames)]
    starts = [int(w["first_index"]) for w in log] + [n_frames]
    return [(w["weather"], starts[i], starts[i + 1]) for i, w in enumerate(log)]


def block_split(
    segments: Sequence[tuple], n_frames: int, block_len: int, val_block: int, guard: int
) -> list[str]:
    """Role of every frame index: ``"train"``, ``"val"`` or ``"guard"``.

    Per segment the ``val_block``-th block of ``block_len`` frames (negative counts from the
    segment end) is val; ``guard`` frames before and after it are dropped from both sets, also
    across a segment boundary (the ego keeps driving when the weather changes)."""
    roles = ["train"] * n_frames
    vals = []
    for _, start, end in segments:
        if val_block < 0:
            v1 = end + (val_block + 1) * block_len
            v0 = v1 - block_len
        else:
            v0 = start + val_block * block_len
            v1 = v0 + block_len
        v0, v1 = max(start, v0), min(end, v1)
        if v1 <= v0:
            raise ValueError(f"segment {start}-{end} has no block {val_block} of {block_len}")
        vals.append((v0, v1))
        for i in range(v0, v1):
            roles[i] = "val"
    for v0, v1 in vals:
        for i in [*range(v0 - guard, v0), *range(v1, v1 + guard)]:
            if 0 <= i < n_frames and roles[i] == "train":
                roles[i] = "guard"
    return roles


def split_ranges(roles: Sequence[str], role: str) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` index ranges with ``role`` (for the report)."""
    out: list[tuple[int, int]] = []
    for i, r in enumerate(roles):
        if r != role:
            continue
        if out and out[-1][1] == i:
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


def pos_repeat(n_base: int, n_pos: int, n_neg: int, share: float) -> float:
    """Repeat factor for CARLA frames with boxes so that CARLA (``n_pos`` x r + ``n_neg``) makes
    up ``share`` of the train list next to ``n_base`` BDD entries (at least 1)."""
    if n_pos == 0:
        return 0.0
    target = share / (1.0 - share) * n_base
    return max(1.0, (target - n_neg) / n_pos)


def keep_fraction(items: Sequence[str], frac: float, seed: int) -> list[str]:
    """A seeded random ``frac`` of ``items`` (sorted, so the list is reproducible)."""
    k = round(len(items) * max(0.0, min(1.0, frac)))
    return sorted(random.Random(seed).sample(list(items), k))


def pseudo_rows(
    cls: Sequence[int], xywhn: Sequence[Sequence[float]], conf: Sequence[float],
    keep: Iterable[int], conf_thr: float,
) -> list[str]:
    """YOLO label rows for predictions of the ``keep`` classes with conf >= ``conf_thr``."""
    keep = set(keep)
    return [
        f"{int(c)} " + " ".join(f"{float(v):.6f}" for v in box)
        for c, box, s in zip(cls, xywhn, conf, strict=True)
        if int(c) in keep and float(s) >= conf_thr
    ]


def merge_label_text(gt_text: str, rows: Sequence[str]) -> str:
    lines = [ln for ln in gt_text.splitlines() if ln.strip()] + list(rows)
    return "\n".join(lines) + ("\n" if lines else "")


def delta_rows(base: dict[str, float], new: dict[str, float]) -> list[tuple[str, float, float]]:
    """``[(key, base, new)]`` for keys in either dict (missing -> nan), base order first."""
    keys = list(base) + [k for k in new if k not in base]
    nan = float("nan")
    return [(k, base.get(k, nan), new.get(k, nan)) for k in keys]


def md_table(title: str, rows: Sequence[tuple[str, float, float]], col: str) -> list[str]:
    out = [f"### {title}", "", f"| {col} | base | mix | delta |", "|---|---|---|---|"]
    for k, b, n in rows:
        out.append(f"| {k} | {b:.3f} | {n:.3f} | {n - b:+.3f} |")
    return [*out, ""]


# ---------- Kaggle-side steps ----------
def stage_carla(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any], list[Path]]:
    from sim.eval_frames import stage_dataset

    root, meta, layout = stage_dataset(cfg["carla"])
    paths = sorted((root / "images").glob("*.jpg"), key=lambda p: frame_index(p.stem))
    print("carla layout:", json.dumps(layout), flush=True)
    return root, meta, paths


def carla_split(
    cfg: dict[str, Any], root: Path, meta: dict, paths: list[Path], names: dict[int, str]
) -> dict:
    s = cfg["split"]
    n = max(frame_index(p.stem) for p in paths) + 1
    segs = weather_segments(meta.get("weathers") or [], n)
    roles = block_split(segs, n, int(s["block_len"]), int(s["val_block"]), int(s["guard"]))
    by_role: dict[str, list[Path]] = {"train": [], "val": [], "guard": []}
    for p in paths:
        by_role[roles[frame_index(p.stem)]].append(p)
    weather = {}
    for w, a, b in segs:
        for i in range(a, b):
            weather[i] = w

    def stats(ps: list[Path]) -> dict[str, Any]:
        boxes: Counter = Counter()
        empty = 0
        for p in ps:
            ids = [ln.split()[0] for ln in (root / "labels" / f"{p.stem}.txt").read_text()
                   .splitlines() if ln.strip()]
            boxes.update(names[int(i)] for i in ids)
            empty += not ids
        per_w = Counter(weather.get(frame_index(p.stem), "unknown") for p in ps)
        return {"frames": len(ps), "empty": empty, "boxes": dict(boxes),
                "frames_per_weather": dict(per_w)}

    info = {
        "segments": [{"weather": w, "start": a, "end": b} for w, a, b in segs],
        "val_ranges": split_ranges(roles, "val"),
        "guard_ranges": split_ranges(roles, "guard"),
        "train": stats(by_role["train"]),
        "val": stats(by_role["val"]),
        "guard_frames": len(by_role["guard"]),
    }
    return {"info": info, "paths": by_role,
            "weathers": {p.stem: weather.get(frame_index(p.stem), "unknown") for p in paths}}


def pseudo_label(
    cfg: dict[str, Any], base: Path, root: Path, train: list[Path], names: dict[int, str]
) -> tuple[list[Path], dict[str, Any]]:
    """Write CARLA-train labels (GT + pseudo boxes) to ``carla.train_dir``; return image paths
    there (``images/`` is a symlink to the staged frames)."""
    tdir = Path(cfg["carla"]["train_dir"])
    (tdir / "labels").mkdir(parents=True, exist_ok=True)
    if not (tdir / "images").exists():
        (tdir / "images").symlink_to(root / "images", target_is_directory=True)
    pc = cfg["pseudo"]
    name_to_id = {v: k for k, v in names.items()}
    keep = [name_to_id[n] for n in pc["classes"]]
    rows: dict[str, list[str]] = {p.stem: [] for p in train}
    if pc["enabled"] and train:
        from ultralytics import YOLO

        model = YOLO(str(base))
        for r in model.predict(
            source=[str(p) for p in train], stream=True, conf=float(pc["conf"]), classes=keep,
            imgsz=int(pc["imgsz"]), verbose=False,
        ):
            b = r.boxes
            rows[Path(r.path).stem] = pseudo_rows(
                b.cls.tolist(), b.xywhn.tolist(), b.conf.tolist(), keep, float(pc["conf"])
            )
    counts: Counter = Counter()
    for p in train:
        counts.update(names[int(r.split()[0])] for r in rows[p.stem])
        gt = (root / "labels" / f"{p.stem}.txt").read_text()
        (tdir / "labels" / f"{p.stem}.txt").write_text(merge_label_text(gt, rows[p.stem]))
    info = {"boxes": dict(counts), "frames_with_pseudo": sum(bool(v) for v in rows.values()),
            "conf": pc["conf"], "enabled": pc["enabled"]}
    print("pseudo labels:", info, flush=True)
    return [tdir / "images" / p.name for p in train], info


def write_mix_lists(
    cfg: dict[str, Any], bdd_root: Path, carla_train: list[Path], root: Path, smoke: bool
) -> dict[str, Any]:
    lists = Path(cfg["lists_dir"])
    lists.mkdir(parents=True, exist_ok=True)
    train = sorted(str(p) for p in (bdd_root / "images/train").glob("*.jpg"))
    val = sorted(str(p) for p in (bdd_root / "images/val").glob("*.jpg"))
    if smoke:
        train, val = train[: cfg["smoke"]["train_limit"]], val[: cfg["smoke"]["val_limit"]]
    info: dict[str, Any] = {"bdd_train_images": len(train), "bdd_val_images": len(val)}
    if cfg["rfs"]["enabled"]:
        classes = [label_classes(label_path(p).read_text()) for p in train]
        train = repeat_list(train, repeat_factors(classes, cfg["rfs"]["threshold"]),
                            cfg["rfs"]["seed"])
    info["bdd_train_entries"] = len(train)
    # Positives / negatives by CARLA GT (before pseudo labels).
    pos = [str(p) for p in carla_train if (root / "labels" / f"{p.stem}.txt").read_text().strip()]
    pos_set = set(pos)
    neg_all = [str(p) for p in carla_train if str(p) not in pos_set]
    m = cfg["mix"]
    neg = keep_fraction(neg_all, float(m["neg_keep"]), int(m["seed"]))
    r = pos_repeat(len(train), len(pos), len(neg), float(m["carla_share"]))
    carla = repeat_list(pos, [r] * len(pos), int(m["seed"])) + neg
    info.update(carla_pos_frames=len(pos), carla_neg_frames=len(neg_all),
                carla_neg_kept=len(neg), carla_pos_repeat=round(r, 3),
                carla_entries=len(carla))
    full = train + carla
    info["train_entries"] = len(full)
    info["carla_share"] = round(len(carla) / max(1, len(full)), 4)
    (lists / "train.txt").write_text("\n".join(full) + "\n")
    (lists / "val.txt").write_text("\n".join(val) + "\n")
    print("lists:", info, flush=True)
    return info


def fetch_artifact(run: Any, ref: str, dest: Path) -> Path:
    from scripts.eval_detection import fetch_weights

    return fetch_weights(run, ref, dest)


def bdd_eval(weights: Path, yml: Path, names: dict[int, str], cfg: dict, tag: str) -> dict:
    from ultralytics import YOLO

    from scripts.eval_detection import per_class_rows

    e = cfg["eval"]
    m = YOLO(str(weights)).val(data=str(yml), imgsz=e["imgsz"], batch=e["batch"],
                               project=cfg["out_dir"], name=f"bdd_{tag}", exist_ok=True,
                               plots=False, verbose=False)
    s = summarise(m, names)
    s.pop("per_class")
    s.update(precision=float(m.box.mp), recall=float(m.box.mr),
             per_class=per_class_rows(m, names))
    return s


def carla_eval(weights: Path, names: dict[int, str], paths: list[Path], weathers: dict,
               cfg: dict, tag: str) -> dict:
    from sim.eval_frames import detection_eval

    ecfg = {"out_dir": cfg["out_dir"], "stage_dir": cfg["carla"]["stage_dir"],
            "detection": {"imgsz": cfg["eval"]["imgsz"], "batch": cfg["eval"]["batch"],
                          "per_weather": cfg["eval"]["per_weather"]}}
    return detection_eval(str(weights), names, paths, weathers, ecfg, f"carlaval_{tag}")


def export_onnx(cfg: dict[str, Any], run: Any, best_ref: str, smoke: bool) -> dict[str, Any]:
    """Ultralytics ONNX export + ORT parity with deploy/export.py's own code and settings."""
    import wandb

    from deploy.common import (
        blob_fn,
        fixture_frames,
        load_bench_config,
        onnx_path,
        runtime_configs,
    )
    from deploy.export import export_yolo

    ex = cfg["export"]
    bcfg = load_bench_config(ex["bench_config"])
    bcfg["wandb"]["detector_artifact"] = best_ref
    bcfg["weights_dir"] = ex["weights_dir"]
    bcfg["files"]["yolo"] = ex["file"]
    rc = runtime_configs(bcfg)
    blobs = [blob_fn("yolo", rc)(f) for f in fixture_frames(bcfg)]
    # export_yolo fetches the weights with run.use_artifact; this run logged them itself, so
    # read them through the public API instead (no use-after-log lineage cycle).
    w = cfg["wandb"]
    api_run = SimpleNamespace(use_artifact=lambda ref: wandb.Api().artifact(
        f"{w['entity']}/{w['project']}/{ref}"))
    info = export_yolo(bcfg, rc, api_run, blobs)
    name = w["onnx_artifact"]
    art = wandb.Artifact(
        name, type="model", description=f"M5.9 ONNX export of {best_ref} (deploy/export.py)",
        metadata={k: v for k, v in info.items() if k != "torch_weights"},
    )
    art.add_file(str(onnx_path(bcfg, "yolo")), name=ex["file"])
    run.log_artifact(art, aliases=["latest", "smoke" if smoke else "m5.9"])
    art.wait()
    info["artifact"] = f"{name}:{art.version}"
    return info


def markdown(res: dict[str, Any]) -> str:
    ev = res["eval"]
    lines = ["# M5.9 BDD + CARLA mix fine-tune", "",
             f"mode {res['mode']}, git {res.get('git_sha')}, weights {res.get('final_weights')}",
             "", "## Split", "", "```", json.dumps(res["split"], indent=1), "```", "",
             "## Lists", "", "```", json.dumps(res["lists"], indent=1), "```", ""]
    b0, b1 = ev["bdd"]["base"], ev["bdd"]["mix"]
    lines += md_table("BDD val, overall", delta_rows(
        {k: b0[k] for k in ("map50", "map50_95", "precision", "recall")},
        {k: b1[k] for k in ("map50", "map50_95", "precision", "recall")}), "metric")
    lines += md_table("BDD val, AP50 per class", delta_rows(
        {r["class"]: r["ap50"] for r in b0["per_class"]},
        {r["class"]: r["ap50"] for r in b1["per_class"]}), "class")
    c0, c1 = ev["carla_val"]["base"], ev["carla_val"]["mix"]
    lines += md_table("CARLA-val, overall (6 classes)", delta_rows(c0["all"], c1["all"]), "metric")
    lines += md_table("CARLA-val, AP50 per class", delta_rows(
        {r["class"]: r["ap50"] for r in c0["per_class"]},
        {r["class"]: r["ap50"] for r in c1["per_class"]}), "class")
    lines += md_table("CARLA-val, mAP@0.5 per weather", delta_rows(
        {w: v["map50"] for w, v in c0["per_weather"].items()},
        {w: v["map50"] for w, v in c1["per_weather"].items()}), "weather")
    lines += md_table("CARLA-val, car AP50 per weather", delta_rows(
        {w: v["per_class_ap50"].get("car", float("nan")) for w, v in c0["per_weather"].items()},
        {w: v["per_class_ap50"].get("car", float("nan")) for w, v in c1["per_weather"].items()}),
        "weather")
    return "\n".join(lines)


def flat(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flat(v, key))
        elif isinstance(v, int | float) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/yolo_mix.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    t_start = time.time()

    cfg = yaml.safe_load(args.config.read_text())
    names_raw = yaml.safe_load(Path(cfg["data_config"]).read_text())["names"]
    names = {int(k): v for k, v in names_raw.items()}
    out = Path(cfg["out_dir"]) / ("smoke" if args.smoke else "full")
    cfg["out_dir"] = str(out)
    out.mkdir(parents=True, exist_ok=True)
    tr = dict(cfg["train"])
    w = cfg["wandb"]
    if args.smoke:
        tr.update(epochs=cfg["smoke"]["epochs"], imgsz=cfg["smoke"]["imgsz"])
        tr["name"] += "-smoke"
        for k in ("ckpt_artifact", "best_artifact", "onnx_artifact"):
            w[k] += "-smoke"  # keep smoke files out of the real lineage

    import wandb
    from ultralytics import YOLO, settings

    from scripts.eval_lanes import git_sha
    from scripts.train_yolo import fetch_checkpoint

    settings.update({"wandb": True})
    ckpt, run_id = (None, None)
    if args.resume:
        ckpt, run_id = fetch_checkpoint(cfg, Path("/kaggle/temp/ckpt"))
    run = wandb.init(
        project=w["project"], entity=w.get("entity"), group=w["group"], name=tr["name"],
        id=run_id, resume="allow", job_type="smoke" if args.smoke else "train",
        config={"train": tr, "mix": cfg["mix"], "split": cfg["split"], "pseudo": cfg["pseudo"],
                "rfs": cfg["rfs"], "git_sha": git_sha()},
        allow_val_change=True,
    )
    run_id = run.id
    base = fetch_artifact(run, w["base_artifact"], Path(cfg["weights_dir"]))

    bdd_root = stage(cfg)
    croot, meta, cpaths = stage_carla(cfg)
    sp = carla_split(cfg, croot, meta, cpaths, names)
    ctrain, cval = sp["paths"]["train"], sp["paths"]["val"]
    if args.smoke:
        k = int(cfg["smoke"]["carla_every"])
        ctrain, cval = ctrain[::k], cval[::k]
    print("split:", json.dumps(sp["info"], indent=1), flush=True)
    ctrain_mix, pseudo = pseudo_label(cfg, base, croot, ctrain, names)
    lists = write_mix_lists(cfg, bdd_root, ctrain_mix, croot, args.smoke)
    run.config.update({"lists": lists, "split_info": sp["info"], "pseudo_info": pseudo},
                      allow_val_change=True)
    yml = data_yaml(names, str(Path(cfg["lists_dir"], "train.txt")),
                    str(Path(cfg["lists_dir"], "val.txt")), out / "data_mix.yaml")
    results: dict[str, Any] = {"mode": "smoke" if args.smoke else "full", "git_sha": git_sha(),
                               "base": w["base_artifact"], "split": sp["info"], "lists": lists,
                               "pseudo": pseudo}

    # Fine-tune.
    t0 = time.time()
    if args.resume:
        model = YOLO(str(ckpt))
        add_callbacks(model, cfg, t_start)
        model.train(resume=True)
    else:
        model = YOLO(str(base))
        add_callbacks(model, cfg, t_start)
        model.train(data=str(yml), project=str(out / "runs"), exist_ok=True, **tr)
    trainer = model.trainer
    results["train"] = {"epochs_done": trainer.epoch + 1, "minutes": (time.time() - t0) / 60,
                        "finished": trainer.epoch + 1 >= trainer.epochs}
    run = ensure_run(w["project"], run_id)
    final = Path(trainer.last if cfg["select"] == "last" else trainer.best)
    shutil.copy(final, out / "mix.pt")
    results["final_weights"] = f"{cfg['select']}.pt"

    # Evaluate base and mix on the same data.
    t0 = time.time()
    ev: dict[str, Any] = {"bdd": {}, "carla_val": {}}
    for tag, wts in (("base", base), ("mix", out / "mix.pt")):
        ev["bdd"][tag] = bdd_eval(wts, yml, names, cfg, tag)
        ev["carla_val"][tag] = carla_eval(wts, names, cval, sp["weathers"], cfg, tag)
        print(tag, "bdd", ev["bdd"][tag]["map50"], "carla-val", ev["carla_val"][tag]["all"],
              flush=True)
    ev["seconds"] = time.time() - t0
    results["eval"] = ev
    run.summary.update({f"m59/{k}": v for k, v in flat(
        {"bdd": {t: {"map50": ev["bdd"][t]["map50"], "map50_95": ev["bdd"][t]["map50_95"],
                     **{r["class"]: r["ap50"] for r in ev["bdd"][t]["per_class"]}}
                 for t in ev["bdd"]},
         "carla_val": {t: {**ev["carla_val"][t]["all"],
                           **{r["class"]: r["ap50"] for r in ev["carla_val"][t]["per_class"]},
                           **{f"weather_{x}": v["map50"]
                              for x, v in ev["carla_val"][t]["per_weather"].items()}}
                       for t in ev["carla_val"]}}).items()})

    # Weights artifact, then the ONNX export from that artifact.
    meta_art = {"bdd_map50": ev["bdd"]["mix"]["map50"],
                "carla_val_map50": ev["carla_val"]["mix"]["all"]["map50"],
                "base": w["base_artifact"], "select": cfg["select"], **results["train"]}
    art = wandb.Artifact(
        w["best_artifact"], type="model", metadata=meta_art,
        description="M5.9 YOLO11s fine-tuned on BDD + CARLA (configs/yolo_mix.yaml)",
    )
    art.add_file(str(out / "mix.pt"), name="best.pt")
    run.log_artifact(art, aliases=["latest", "smoke" if args.smoke else "candidate"])
    art.wait()
    best_ref = f"{w['best_artifact']}:{art.version}"
    results["artifact"] = best_ref
    try:
        info = export_onnx(cfg, run, best_ref, args.smoke)
        results["onnx"] = {k: v for k, v in info.items() if k != "torch_weights"}
    except Exception as exc:  # noqa: BLE001 - weights and metrics are already logged
        import traceback

        traceback.print_exc()
        results["onnx"] = {"error": f"{type(exc).__name__}: {exc}"}

    results["minutes"] = (time.time() - t_start) / 60
    results["wandb_url"] = run.url
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "mix_eval.md").write_text(markdown(results))
    run.save(str(out / "results.json"), base_path=str(out), policy="now")
    run.save(str(out / "mix_eval.md"), base_path=str(out), policy="now")
    print(markdown(results), flush=True)
    print(f"total time {results['minutes']:.1f} min, W&B {run.url}", flush=True)
    wandb.finish()
    if "error" in results["onnx"]:
        raise SystemExit("ONNX export failed (weights and metrics are logged)")


if __name__ == "__main__":
    main()
