"""Locate the data-prep output on Kaggle and write a resolved Ultralytics data YAML.

The data-prep kernel output is attached to later kernels with
``kernel_sources: ["anish2906/aiml04-data-prep"]``; this finds where Kaggle mounted it, checks
that images and labels line up, and writes a copy of ``configs/data_bdd.yaml`` whose ``path``
points at the real mount (prints a warning if the committed ``path`` is wrong).

    python -m scripts.data.check_data_yaml --config configs/data_bdd.yaml \
        --search /kaggle/input --out /kaggle/working/data_bdd_resolved.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def find_dataset_root(search: Path, marker: str = "data.yaml", folder: str = "bdd_yolo") -> Path:
    """Return the first ``<...>/bdd_yolo`` folder under ``search`` that contains ``data.yaml``."""
    hits = sorted(p.parent for p in search.glob(f"**/{folder}/{marker}"))
    if not hits:
        raise FileNotFoundError(f"no {folder}/{marker} under {search}")
    return hits[0]


def check_split(root: Path, split: str) -> dict[str, int]:
    imgs = {p.stem for p in (root / "images" / split).glob("*.jpg")}
    lbls = {p.stem for p in (root / "labels" / split).glob("*.txt")}
    return {"images": len(imgs), "labels": len(lbls), "unmatched": len(imgs ^ lbls)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/data_bdd.yaml"))
    ap.add_argument("--search", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/data_bdd_resolved.yaml"))
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    root = find_dataset_root(args.search)
    print("dataset root:", root, flush=True)
    if Path(cfg["path"]) != root:
        print(f"WARNING: configs path {cfg['path']} != mounted {root}", flush=True)
    for split in ("train", "val"):
        stats = check_split(root, split)
        print(split, stats, flush=True)
        if stats["images"] == 0 or stats["unmatched"]:
            raise SystemExit(f"{split}: images and labels do not line up: {stats}")
    data = {k: v for k, v in cfg.items() if k != "prep"}
    data["path"] = str(root)
    args.out.write_text(yaml.safe_dump(data, sort_keys=False))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
