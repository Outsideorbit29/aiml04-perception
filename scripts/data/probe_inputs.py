"""Describe the layout of attached Kaggle datasets (runs on Kaggle, CPU only).

Prints, per directory: file count, total size and extension histogram; lists every non-image
file (capped per directory) and peeks into JSON label files so the converter can be pointed at
the right paths. Nothing is copied.

    python scripts/data/probe_inputs.py --root /kaggle/input --out /kaggle/working/probe.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bin"}


def peek_json(path: Path, max_bytes: int = 1500) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read(max_bytes)


def probe(root: Path, max_listed: int = 30) -> dict:
    report: dict = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        ext = Counter(Path(f).suffix.lower() for f in filenames)
        size = 0
        for f in filenames:
            try:
                size += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
        others = sorted(f for f in filenames if Path(f).suffix.lower() not in IMAGE_EXT)
        entry = {
            "n_files": len(filenames),
            "size_mb": round(size / 1e6, 1),
            "ext": dict(ext),
            "non_image_files": others[:max_listed],
        }
        jsons = [f for f in others if f.endswith(".json")]
        if jsons:
            entry["json_peek"] = {jsons[0]: peek_json(Path(dirpath) / jsons[0])}
            big = [f for f in jsons if os.path.getsize(os.path.join(dirpath, f)) > 1e6]
            if big:
                entry["json_peek"][big[0]] = peek_json(Path(dirpath) / big[0])
        txts = [f for f in others if f.endswith(".txt")]
        if txts:
            entry["txt_peek"] = {txts[0]: peek_json(Path(dirpath) / txts[0], 600)}
        if filenames or len(dirnames) > 0:
            report[dirpath] = entry
        if filenames:
            print(f"{dirpath}: {len(filenames)} files, {entry['size_mb']} MB, {dict(ext)}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("probe.json"))
    args = ap.parse_args()
    print("top level:", sorted(os.listdir(args.root)), flush=True)
    report = probe(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
