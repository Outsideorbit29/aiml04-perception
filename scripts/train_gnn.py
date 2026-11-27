"""F4: GNN on the transaction/client/card/device graph, fraud rings, blend with XGBoost.

Runs on Kaggle GPU (kaggle/gnn) with the F2 features, the F3 model and the secrets attached:
    python -m scripts.train_gnn --config configs/gnn.yaml [--smoke]
Reports, on val (selection) and test (once):
- xgb: the F3 serving model; gnn: GraphSAGE; blend: w*logit(xgb) + (1-w)*logit(gnn), w on val
- serving blend: xgb blended with a per-client GNN risk computed from EARLIER periods only
  (train for val, train+val for test), which is what the live scorer can look up
Outputs in out_dir: gnn_summary.json, scores.parquet, rings.parquet, ring_stats.json,
client_risk.parquet (train+val, for serving), gnn_state.pt.
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
from fraud.graph import best_blend, blend_logits, encode_entities, find_rings
from fraud.metrics import evaluate, threshold_at_fpr
from fraud.model import FeatureSpec
from fraud.tracking import configure
from scripts.train import find_features


def find_model_dir(model_dir: str, search_root: str = "/kaggle/input") -> Path:
    if model_dir != "auto":
        return Path(model_dir)
    for depth in range(1, 6):
        hits = sorted(glob.glob(os.path.join(search_root, *["*"] * depth, "spec.json")))
        hits = [h for h in hits if Path(h).with_name("model.json").exists()]
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError(f"spec.json + model.json not found under {search_root}")


def scored(name: str, y: dict, p: dict, max_fpr: float) -> dict:
    thr = threshold_at_fpr(y["val"], p["val"], max_fpr)
    return {
        "name": name,
        "threshold": thr,
        **evaluate(y["val"], p["val"], thr, "val_"),
        **evaluate(y["test"], p["test"], thr, "test_"),
    }


def client_risk(keys: pd.Series, probs: np.ndarray) -> pd.Series:
    """Mean GNN probability per client over the given rows."""
    return pd.Series(probs, index=keys.values).groupby(level=0).mean()


def ring_stats(df: pd.DataFrame, rings: pd.DataFrame, target: str) -> dict:
    in_ring = df["client_key"].isin(set(rings["client_key"]))
    stats = {"rings": int(rings["ring_id"].nunique()), "ring_clients": int(len(rings))}
    for split in ("train", "val", "test"):
        part = df["split"] == split
        stats[split] = {
            "tx_in_rings": int((part & in_ring).sum()),
            "fraud_rate_in_rings": float(df.loc[part & in_ring, target].mean()),
            "fraud_rate_outside": float(df.loc[part & ~in_ring, target].mean()),
        }
    train = df[df["split"] == "train"].merge(rings, on="client_key")
    top = (
        train.groupby("ring_id")
        .agg(
            size=("ring_size", "first"),
            devices=("ring_devices", "first"),
            tx=(target, "size"),
            fraud_rate=(target, "mean"),
        )
        .sort_values(["fraud_rate", "tx"], ascending=False)
    )
    stats["ring_size_hist"] = rings.groupby("ring_id")["ring_size"].first().describe().to_dict()
    stats["rings_with_train_fraud_rate_over_50pct"] = int((top["fraud_rate"] > 0.5).sum())
    stats["top_rings_train"] = top.head(10).reset_index().to_dict("records")
    return stats


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gnn.yaml")
    ap.add_argument("--mlflow-config", default="configs/mlflow.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    import torch
    import xgboost as xgb

    from fraud.gnn import build_hetero, prepare_tx_features, train_gnn

    cfg = load_config(args.config)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    target, max_fpr = cfg["target"], cfg["max_fpr"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    feat_dir = find_features(cfg["features_dir"])
    model_dir = find_model_dir(cfg["model_dir"])
    parts = []
    for split in ("train", "val", "test"):
        part = pd.read_parquet(feat_dir / "features" / f"{split}.parquet")
        if args.smoke:
            part = part.iloc[: cfg["smoke"]["rows_per_split"][split]]
        parts.append(part.assign(split=split))
    df = pd.concat(parts, ignore_index=True).sort_values(cfg["time_column"], kind="stable")
    df = df.reset_index(drop=True)
    idx = {s: np.flatnonzero(df["split"].to_numpy() == s) for s in ("train", "val", "test")}
    y_all = df[target].to_numpy().astype(int)
    y = {s: y_all[i] for s, i in idx.items()}
    print({s: (len(i), int(y[s].sum())) for s, i in idx.items()}, flush=True)

    # F3 model scores and the shared input matrix.
    spec = FeatureSpec.load(model_dir / "spec.json")
    booster = xgb.XGBClassifier()
    booster.load_model(model_dir / "model.json")
    x_all = spec.transform(df)
    p_xgb = booster.predict_proba(x_all)[:, 1]
    xgb_meta = json.loads((model_dir / "threshold.json").read_text())

    # Graph + GNN.
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[idx["train"]] = True
    x_tx = prepare_tx_features(x_all, train_mask)
    enc = encode_entities(df, cfg["entities"])
    data = build_hetero(x_tx, enc)
    graph_info = {
        "tx_nodes": int(len(df)),
        **{f"{c}_nodes": int(n) for c, (_, _, n) in enc.items()},
        **{f"{c}_edges": int(len(t)) for c, (t, _, _) in enc.items()},
    }
    print(graph_info, flush=True)
    gnn_cfg = dict(cfg["gnn"])
    if args.smoke:
        gnn_cfg["epochs"] = cfg["smoke"]["epochs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model, p_gnn, info = train_gnn(
        data, y_all, idx["train"], idx["val"], gnn_cfg, device, log=lambda r: print(r, flush=True)
    )
    gnn_seconds = time.time() - t0

    split_scores = lambda p: {s: p[i] for s, i in idx.items()}  # noqa: E731
    w, grid = best_blend(y["val"], p_xgb[idx["val"]], p_gnn[idx["val"]], cfg["blend_grid"])
    p_blend = blend_logits(p_xgb, p_gnn, w)

    # Serving-style blend: client GNN risk from earlier periods only.
    keys = df["client_key"]
    risk_train = client_risk(keys.iloc[idx["train"]], p_gnn[idx["train"]])
    trval = np.concatenate([idx["train"], idx["val"]])
    risk_trval = client_risk(keys.iloc[trval], p_gnn[trval])
    p_risk = np.full(len(df), np.nan)
    p_risk[idx["val"]] = keys.iloc[idx["val"]].map(risk_train).to_numpy()
    p_risk[idx["test"]] = keys.iloc[idx["test"]].map(risk_trval).to_numpy()
    p_risk = np.where(np.isnan(p_risk), p_xgb, p_risk)  # unseen client -> xgb alone
    w_s, grid_s = best_blend(y["val"], p_xgb[idx["val"]], p_risk[idx["val"]], cfg["blend_grid"])
    p_serving = blend_logits(p_xgb, p_risk, w_s)

    results = [
        scored("xgb", y, split_scores(p_xgb), max_fpr),
        scored("gnn", y, split_scores(p_gnn), max_fpr),
        scored("blend", y, split_scores(p_blend), max_fpr),
        scored("serving_blend", y, split_scores(p_serving), max_fpr),
    ]
    for r in results:
        print(r["name"], {k: round(v, 4) for k, v in r.items() if k != "name"}, flush=True)

    rings = find_rings(
        df["client_key"],
        df["device_key"],
        cfg["rings"]["max_clients_per_device"],
        cfg["rings"]["min_size"],
    )
    r_stats = ring_stats(df, rings, target)
    print(
        "rings",
        {k: r_stats[k] for k in ("rings", "ring_clients", "train", "val", "test")},
        flush=True,
    )

    pd.DataFrame(
        {
            cfg["id_column"]: df[cfg["id_column"]].values,
            "split": df["split"].values,
            "p_xgb": p_xgb,
            "p_gnn": p_gnn,
            "p_blend": p_blend,
            "p_serving": p_serving,
        }
    ).to_parquet(out_dir / "scores.parquet", index=False)
    rings.to_parquet(out_dir / "rings.parquet", index=False)
    risk_trval.rename("gnn_risk").rename_axis("client_key").reset_index().to_parquet(
        out_dir / "client_risk.parquet", index=False
    )
    (out_dir / "ring_stats.json").write_text(json.dumps(r_stats, indent=2, default=float))
    torch.save(model.state_dict(), out_dir / "gnn_state.pt")

    summary = {
        "smoke": bool(args.smoke),
        "device": device,
        "gnn_seconds": round(gnn_seconds, 1),
        "graph": graph_info,
        "gnn": {k: v for k, v in info.items() if k != "history"},
        "xgb_model": xgb_meta.get("experiment"),
        "blend_weight_xgb": w,
        "blend_grid_val_pr_auc": grid,
        "serving_blend_weight_xgb": w_s,
        "serving_grid_val_pr_auc": grid_s,
        "results": results,
        "rings": {k: r_stats[k] for k in ("rings", "ring_clients", "train", "val", "test")},
    }

    tracking_uri = configure(load_config(args.mlflow_config))
    try:
        import mlflow

        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
        with mlflow.start_run(run_name="gnn_blend" + ("-smoke" if args.smoke else "")):
            mlflow.set_tags({"milestone": "F4", "git_sha": os.environ.get("GIT_SHA", "unknown")})
            mlflow.log_params(
                {
                    **{f"gnn_{k}": v for k, v in gnn_cfg.items()},
                    "blend_w": w,
                    "serving_blend_w": w_s,
                    **graph_info,
                }
            )
            for r in results:
                mlflow.log_metrics({f"{r['name']}_{k}": v for k, v in r.items() if k != "name"})
            for row in info["history"]:
                mlflow.log_metric("gnn_val_pr_auc_curve", row["val_pr_auc"], step=row["epoch"])
            mlflow.log_metrics({"rings": r_stats["rings"], "ring_clients": r_stats["ring_clients"]})
            (out_dir / "gnn_summary.json").write_text(json.dumps(summary, indent=2, default=float))
            mlflow.log_artifact(str(out_dir / "gnn_summary.json"))
            mlflow.log_artifact(str(out_dir / "ring_stats.json"))
        summary["tracking_uri"] = tracking_uri
    except Exception as exc:
        print(f"MLflow logging failed: {type(exc).__name__}: {exc}", flush=True)
        summary["tracking_uri"] = None
    (out_dir / "gnn_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    main()
