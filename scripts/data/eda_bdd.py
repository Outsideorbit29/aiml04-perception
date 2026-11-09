"""EDA for the BDD100K detection data (runs on Kaggle after ``prepare_bdd``; CPU only).

Reads the compact label files written by ``prepare_bdd`` (boxes at the original resolution) and
writes PNG figures plus ``eda_summary.json`` / ``eda_summary.md``:

    python -m scripts.data.eda_bdd --data /kaggle/working/bdd_yolo --out /kaggle/working/eda

Figures: eda_class_counts.png, eda_instances_per_image.png, eda_box_sizes.png,
eda_conditions.png, eda_samples.png.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.data.bdd_yolo import CLASSES, area_bucket

# Reference categorical palette slots 1-3 (light mode) and text tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SPLITS = {"train_full": "train (all 70k)", "train": "train subset (20k)", "val": "val (10k)"}
BUCKETS = ("small", "medium", "large")


def read_frames(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def split_stats(frames: list[dict[str, Any]]) -> dict[str, Any]:
    cls = Counter()
    size = {c: Counter() for c in CLASSES}
    per_image = Counter()
    occluded = Counter()
    cond = Counter()
    for fr in frames:
        per_image[len(fr["boxes"])] += 1
        cond[(fr["timeofday"], fr["weather"])] += 1
        for cid, x1, y1, x2, y2, occ, _trunc in fr["boxes"]:
            name = CLASSES[cid]
            cls[name] += 1
            size[name][area_bucket(abs(x2 - x1) * abs(y2 - y1))] += 1
            occluded[name] += occ
    n_img = len(frames)
    n_box = sum(cls.values())
    counts = [k for k, v in per_image.items() for _ in range(v)]
    counts.sort()
    return {
        "images": n_img,
        "boxes": n_box,
        "class_counts": {c: cls[c] for c in CLASSES},
        "class_share": {c: round(cls[c] / max(n_box, 1), 4) for c in CLASSES},
        "images_with_class": None,  # filled below
        "instances_per_image": {
            "mean": round(n_box / max(n_img, 1), 2),
            "median": counts[len(counts) // 2] if counts else 0,
            "p95": counts[int(len(counts) * 0.95)] if counts else 0,
            "max": counts[-1] if counts else 0,
            "empty_images": per_image[0],
            "hist": {str(k): per_image[k] for k in sorted(per_image)},
        },
        "box_size": {c: {b: size[c][b] for b in BUCKETS} for c in CLASSES},
        "box_size_all": {b: sum(size[c][b] for c in CLASSES) for b in BUCKETS},
        "occluded_share": {c: round(occluded[c] / max(cls[c], 1), 3) for c in CLASSES},
        "timeofday": dict(Counter(fr["timeofday"] for fr in frames).most_common()),
        "weather": dict(Counter(fr["weather"] for fr in frames).most_common()),
        "conditions": {f"{t}|{w}": n for (t, w), n in cond.most_common()},
    }


def images_with_class(frames: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter()
    for fr in frames:
        for cid in {b[0] for b in fr["boxes"]}:
            c[CLASSES[cid]] += 1
    return {k: c[k] for k in CLASSES}


def style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "text.color": TEXT,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def plot_class_counts(plt, stats: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    keys = [k for k in SPLITS if k in stats]
    width = 0.8 / len(keys)
    for i, k in enumerate(keys):
        vals = [stats[k]["class_counts"][c] for c in CLASSES]
        xs = [j + (i - (len(keys) - 1) / 2) * width for j in range(len(CLASSES))]
        ax.bar(xs, vals, width=width * 0.92, color=SERIES[i], label=SPLITS[k])
    ax.set_yscale("log")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=20)
    ax.set_ylabel("boxes (log scale)")
    ax.set_title("Boxes per class")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_instances(plt, stats: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, k in enumerate(k for k in ("train", "val") if k in stats):
        hist = {int(a): b for a, b in stats[k]["instances_per_image"]["hist"].items()}
        n = stats[k]["images"]
        xs = list(range(0, min(max(hist), 80) + 1))
        ax.plot(
            xs,
            [hist.get(x, 0) / n for x in xs],
            color=SERIES[i + 1],
            lw=2,
            label=f"{SPLITS[k]} (mean {stats[k]['instances_per_image']['mean']})",
        )
    ax.set_xlabel("boxes per image (9 classes)")
    ax.set_ylabel("share of images")
    ax.set_title("Instances per image")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_box_sizes(plt, st: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    left = [0.0] * len(CLASSES)
    for i, b in enumerate(BUCKETS):
        shares = []
        for c in CLASSES:
            tot = sum(st["box_size"][c].values()) or 1
            shares.append(st["box_size"][c][b] / tot)
        ax.barh(
            range(len(CLASSES)),
            shares,
            left=left,
            color=SERIES[i],
            height=0.7,
            edgecolor=SURFACE,
            linewidth=2,
            label=f"{b}",
        )
        left = [a + s for a, s in zip(left, shares, strict=True)]
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of boxes (COCO buckets at 1280x720: small < 32², large ≥ 96²)")
    ax.set_title("Box size per class (train subset)")
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.06))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_conditions(plt, stats: dict[str, Any], out: Path) -> None:
    keys = [k for k in ("train_full", "train", "val") if k in stats]
    tods = list(stats[keys[0]]["timeofday"])
    weathers = list(stats[keys[0]]["weather"])
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, k in zip(axes[0], keys, strict=True):
        n = stats[k]["images"]
        grid = [
            [stats[k]["conditions"].get(f"{t}|{w}", 0) / n * 100 for w in weathers] for t in tods
        ]
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=40, aspect="auto")
        for r, row in enumerate(grid):
            for c, v in enumerate(row):
                ax.text(
                    c,
                    r,
                    f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > 25 else TEXT,
                )
        ax.set_xticks(range(len(weathers)), weathers, rotation=35, ha="right")
        ax.set_yticks(range(len(tods)), tods)
        ax.set_title(f"{SPLITS[k]}: % of images")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_samples(plt, frames: list[dict[str, Any]], img_dir: Path, out: Path, seed: int) -> None:
    import cv2

    rng = random.Random(seed)
    wanted = [
        ("daytime", "clear"),
        ("night", "clear"),
        ("daytime", "rainy"),
        ("night", "rainy"),
        ("daytime", "snowy"),
        ("dawn/dusk", "partly cloudy"),
        ("daytime", "overcast"),
        ("night", "snowy"),
        ("daytime", "foggy"),
    ]
    picks = []
    for tod, w in wanted:
        cand = [f for f in frames if f["timeofday"] == tod and f["weather"] == w and f["boxes"]]
        if cand:
            picks.append(rng.choice(cand))
    colors = [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
        "#52514e",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    for ax, fr in zip(axes.flat, picks, strict=False):
        img = cv2.imread(str(img_dir / fr["name"]))
        if img is None:
            continue
        sh, sw = img.shape[:2]
        sx, sy = sw / 1280.0, sh / 720.0
        for cid, x1, y1, x2, y2, _o, _t in fr["boxes"]:
            hexc = colors[cid].lstrip("#")
            bgr = tuple(int(hexc[i : i + 2], 16) for i in (4, 2, 0))
            cv2.rectangle(img, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), bgr, 2)
        ax.imshow(img[:, :, ::-1])
        ax.set_title(f"{fr['timeofday']} / {fr['weather']} - {len(fr['boxes'])} boxes", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(picks) :]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=colors[i], lw=3, label=c) for i, c in enumerate(CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncols=9, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=90)
    plt.close(fig)


def recommend(st: dict[str, Any]) -> dict[str, Any]:
    """Class-handling numbers: imbalance ratios and inverse-sqrt-frequency weights."""
    counts = st["class_counts"]
    top = max(counts.values())
    inv_sqrt = {c: (top / max(n, 1)) ** 0.5 for c, n in counts.items()}
    mean_w = sum(inv_sqrt.values()) / len(inv_sqrt)
    return {
        "imbalance_vs_car": {c: round(top / max(n, 1), 1) for c, n in counts.items()},
        "inv_sqrt_freq_weights": {c: round(w / mean_w, 3) for c, w in inv_sqrt.items()},
        "images_with_class": st["images_with_class"],
    }


def write_markdown(summary: dict[str, Any], out: Path) -> None:
    lines = ["# BDD100K EDA summary", ""]
    for k, label in SPLITS.items():
        if k not in summary:
            continue
        st = summary[k]
        ipi = st["instances_per_image"]
        lines += [
            f"## {label}",
            f"- images {st['images']}, boxes {st['boxes']}, boxes/image mean {ipi['mean']}, "
            f"median {ipi['median']}, p95 {ipi['p95']}, max {ipi['max']}, "
            f"empty {ipi['empty_images']}",
            f"- box sizes (all classes): {st['box_size_all']}",
            f"- timeofday: {st['timeofday']}",
            f"- weather: {st['weather']}",
            "",
            "| class | boxes | share | images with class | small | medium | large | occluded |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in CLASSES:
            bs = st["box_size"][c]
            lines.append(
                f"| {c} | {st['class_counts'][c]} | {st['class_share'][c]:.2%} | "
                f"{st['images_with_class'][c]} | {bs['small']} | {bs['medium']} | "
                f"{bs['large']} | {st['occluded_share'][c]:.0%} |"
            )
        lines.append("")
    rec = summary.get("recommendation")
    if rec:
        lines += [
            "## Class-handling numbers (train subset)",
            "",
            "| class | car/class ratio | inv-sqrt-freq weight |",
            "|---|---|---|",
        ]
        for c in CLASSES:
            lines.append(
                f"| {c} | {rec['imbalance_vs_car'][c]} | {rec['inv_sqrt_freq_weights'][c]} |"
            )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="prepare_bdd output folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    args.out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {}
    frames_by_split: dict[str, list] = {}
    for k in SPLITS:
        p = args.data / "meta" / f"frames_{k}.jsonl.gz"
        if p.exists():
            frames = read_frames(p)
            stats[k] = split_stats(frames)
            stats[k]["images_with_class"] = images_with_class(frames)
            frames_by_split[k] = frames
            print(f"{k}: {stats[k]['images']} images, {stats[k]['boxes']} boxes", flush=True)
    sub = "train" if "train" in stats else next(iter(stats))
    stats["recommendation"] = recommend(stats[sub])
    plot_class_counts(plt, stats, args.out / "eda_class_counts.png")
    plot_instances(plt, stats, args.out / "eda_instances_per_image.png")
    plot_box_sizes(plt, stats[sub], args.out / "eda_box_sizes.png")
    plot_conditions(plt, stats, args.out / "eda_conditions.png")
    plot_samples(
        plt,
        frames_by_split[sub],
        args.data / "images" / sub,
        args.out / "eda_samples.png",
        args.seed,
    )
    (args.out / "eda_summary.json").write_text(json.dumps(stats, indent=1))
    write_markdown(stats, args.out / "eda_summary.md")
    print("wrote", sorted(p.name for p in args.out.iterdir()), flush=True)


if __name__ == "__main__":
    main()
