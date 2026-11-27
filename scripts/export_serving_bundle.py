"""F5: bundle everything the serving stack needs into one folder (runs on Kaggle).

Inputs (attached kernel outputs): F1 splits, F3 model, F4 client risk / rings / summary.
Output folder (default /kaggle/working/serving_bundle):
    models/   model.json, spec.json, threshold.json, blend.json, client_risk.parquet
    data/     history.parquet (train+val, columns the feature state needs),
              replay.parquet (test period, raw columns the scorer needs + isFraud),
              rings.parquet
Download it once to the laptop (scripts/fetch_serving_bundle.sh); nothing here is committed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from fraud.config import load_config


def find_one(name: str, root: str = "/kaggle/input", depth: int = 6) -> Path:
    for d in range(1, depth + 1):
        hits = sorted(glob.glob(os.path.join(root, *["*"] * d, name)))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"{name} not found under {root}")


def needed_columns(feat_cfg: dict) -> tuple[list[str], list[str]]:
    state_cols = [
        feat_cfg["id_column"],
        feat_cfg["time_column"],
        feat_cfg["amount_column"],
        feat_cfg["d1_column"],
        feat_cfg["addr_column"],
        feat_cfg["email_column"],
        *feat_cfg["card_key_columns"],
        *[c for c in feat_cfg["client_key_columns"] if c != "d1n"],
        *feat_cfg["device_key_columns"],
    ]
    replay_cols = [
        *state_cols,
        *feat_cfg["passthrough_numeric"],
        *feat_cfg["passthrough_categorical"],
        feat_cfg["target"],
    ]
    dedup = lambda cols: list(dict.fromkeys(cols))  # noqa: E731
    return dedup(state_cols), dedup(replay_cols)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-config", default="configs/features.yaml")
    ap.add_argument("--root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/serving_bundle")
    args = ap.parse_args(argv)

    feat_cfg = load_config(args.features_config)
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    model_dir = find_one("spec.json", args.root).parent
    for f in ("model.json", "spec.json", "threshold.json"):
        shutil.copy(model_dir / f, out / "models" / f)
    gnn_summary = json.loads(find_one("gnn_summary.json", args.root).read_text())
    serving = next(r for r in gnn_summary["results"] if r["name"] == "serving_blend")
    blend = {
        "weight_xgb": gnn_summary["serving_blend_weight_xgb"],
        "threshold": serving["threshold"],
        "test_pr_auc": serving["test_pr_auc"],
        "test_recall": serving["test_recall"],
        "test_precision": serving["test_precision"],
    }
    (out / "models" / "blend.json").write_text(json.dumps(blend, indent=2))
    shutil.copy(find_one("client_risk.parquet", args.root), out / "models" / "client_risk.parquet")
    shutil.copy(find_one("rings.parquet", args.root), out / "data" / "rings.parquet")

    split_dir = find_one("train.parquet", args.root).parent
    if split_dir.name != "splits":  # the F2 features also have train.parquet; want F1 splits
        split_dir = next(
            p.parent
            for p in map(
                Path,
                glob.glob(os.path.join(args.root, "**", "splits", "train.parquet"), recursive=True),
            )
        )
    state_cols, replay_cols = needed_columns(feat_cfg)
    parts = {s: pd.read_parquet(split_dir / f"{s}.parquet") for s in ("train", "val", "test")}
    history = pd.concat([parts["train"], parts["val"]], ignore_index=True)
    history = history[[c for c in state_cols if c in history]]
    replay = parts["test"][[c for c in replay_cols if c in parts["test"]]]
    history.to_parquet(out / "data" / "history.parquet", index=False)
    replay.to_parquet(out / "data" / "replay.parquet", index=False)
    info = {
        "history_rows": len(history),
        "replay_rows": len(replay),
        "replay_columns": list(replay.columns),
        "blend": blend,
        "model": json.loads((model_dir / "threshold.json").read_text()),
        "sizes_mb": {
            str(p.relative_to(out)): round(p.stat().st_size / 1e6, 2) for p in out.rglob("*.*")
        },
    }
    (out / "bundle_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
