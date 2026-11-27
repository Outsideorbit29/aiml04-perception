"""F2: build client/card/device features for all splits, check offline/online parity, write outputs.

Runs on Kaggle (kaggle/features) with the F1 output attached. Locally, on a prepared split dir:
    python -m scripts.build_features --input-dir <dir with splits/> --out-dir out
Outputs in out_dir:
    features/{train,val,test}.parquet  model-ready rows (id, time, label, passthrough, features)
    client_features.parquet            Feast offline source (client_key, event_timestamp, features)
    features_summary.json              timings, parity report, per-feature stats on train
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.config import load_config
from fraud.data import SPLIT_NAMES
from fraud.features import KEY_COLUMNS, build_features, feature_names, online_features, parity


def find_splits(input_dir: str, search_root: str = "/kaggle/input") -> Path:
    if input_dir != "auto":
        return Path(input_dir) / "splits"
    for depth in range(1, 6):
        pattern = os.path.join(search_root, *["*"] * depth, "splits", "train.parquet")
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError(f"splits/train.parquet not found under {search_root}")


def auc(y: np.ndarray, x: np.ndarray) -> float | None:
    """ROC-AUC of a single score via ranks (Mann-Whitney); NaN rows dropped."""
    m = ~np.isnan(x)
    y, x = y[m], x[m]
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def feature_stats(df: pd.DataFrame, names: list[str], target: str) -> dict:
    y = df[target].to_numpy()
    stats = {}
    for n in names:
        x = df[n].to_numpy(dtype="float64")
        a = auc(y, x)
        stats[n] = {
            "missing": float(np.isnan(x).mean()),
            "mean_legit": float(np.nanmean(x[y == 0])) if (y == 0).any() else None,
            "mean_fraud": float(np.nanmean(x[y == 1])) if (y == 1).any() else None,
            # direction-free separation: max(AUC, 1 - AUC)
            "auc": a,
            "separation": None if a is None else max(a, 1 - a),
        }
    return stats


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/features.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--input-dir", help="override config input_dir (folder containing splits/)")
    ap.add_argument("--out-dir", help="override config out_dir")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    split_dir = find_splits(args.input_dir or cfg["input_dir"])
    out_dir = Path(args.out_dir or cfg["out_dir"])
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    t_col, target, id_col = cfg["time_column"], cfg["target"], cfg["id_column"]
    names = feature_names(cfg["windows"])

    parts = []
    for name in SPLIT_NAMES:
        part = pd.read_parquet(split_dir / f"{name}.parquet")
        parts.append(part.assign(split=name))
    df = pd.concat(parts, ignore_index=True).sort_values(t_col, kind="stable")
    df = df.reset_index(drop=True)
    if args.smoke:
        df = df.iloc[: cfg["smoke_rows"]].copy()
    print(f"{len(df):,} rows from {split_dir}", flush=True)

    t0 = time.time()
    feats = build_features(df, cfg)
    t_offline = time.time() - t0
    print(f"offline features: {t_offline:.1f} s", flush=True)

    n_par = min(len(df), cfg["smoke_rows"] if args.smoke else cfg["parity_rows"])
    t0 = time.time()
    online = online_features(df.iloc[:n_par], cfg)
    t_online = time.time() - t0
    report = parity(feats.iloc[:n_par], online, names)
    head = feats.iloc[:n_par]
    key_match = float(
        np.mean(
            [(head[k].fillna("<none>") == online[k].fillna("<none>")).mean() for k in KEY_COLUMNS]
        )
    )
    print(f"parity on {n_par:,} rows: max rel diff {max(report.values()):.2e}", flush=True)

    keep_num = [c for c in cfg["passthrough_numeric"] if c in df]
    keep_cat = [c for c in cfg["passthrough_categorical"] if c in df]
    model_df = pd.concat(
        [df[[id_col, t_col, target, "split", *keep_num, *keep_cat]], feats], axis=1
    )
    for c in keep_cat:
        model_df[c] = model_df[c].astype("string")
    for c in names:
        model_df[c] = model_df[c].astype("float32")
    for name in SPLIT_NAMES:
        part = model_df[model_df["split"] == name].drop(columns="split")
        part.to_parquet(out_dir / "features" / f"{name}.parquet", index=False)

    event_ts = pd.Timestamp(data_cfg["start_date"], tz="UTC") + pd.to_timedelta(
        df[t_col].astype("int64"), unit="s"
    )
    feast_df = feats[["client_key", *names]].astype({n: "float32" for n in names})
    feast_df.insert(1, "event_timestamp", event_ts.values)
    feast_df.insert(2, id_col, df[id_col].values)
    feast_df.to_parquet(out_dir / "client_features.parquet", index=False)

    train = model_df[model_df["split"] == "train"]
    summary = {
        "smoke": bool(args.smoke),
        "rows": int(len(df)),
        "rows_per_split": {n: int((model_df["split"] == n).sum()) for n in SPLIT_NAMES},
        "clients": int(feats["client_key"].nunique()),
        "cards": int(feats["card_key"].nunique()),
        "devices": int(feats["device_key"].nunique()),
        "client_tx_median": float(feats.groupby("client_key").size().median()),
        "offline_seconds": round(t_offline, 1),
        "online_seconds": round(t_online, 1),
        "online_us_per_tx": round(1e6 * t_online / max(n_par, 1), 1),
        "parity_rows": int(n_par),
        "parity_key_match": key_match,
        "parity_max_rel_diff": {k: (None if np.isinf(v) else v) for k, v in report.items()},
        "parity_ok": bool(key_match == 1.0 and max(report.values()) < 1e-5),
        "feature_stats_train": feature_stats(train, names, target),
        "model_columns": {"numeric": keep_num, "categorical": keep_cat, "features": names},
    }
    (out_dir / "features_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in list(summary)[:10]}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
