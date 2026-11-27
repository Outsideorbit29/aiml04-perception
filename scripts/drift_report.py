"""F6: weekly drift report over val + test against the training split (runs on Kaggle).

    python -m scripts.drift_report --config configs/drift.yaml [--smoke]

Inputs (attached kernel outputs): F2 features (features/{train,val,test}.parquet,
features_summary.json) and, for prediction drift and performance, the F3 model (model.json,
spec.json, threshold.json). Per window of `window_days`:
- PSI of every model input (fraud/drift.py) and the share of drifted inputs,
- PSI of the XGBoost score vs the val scores, label drift (fraud rate),
- PR-AUC, precision, recall, FPR at the serving threshold, and the retraining decision,
- an Evidently DataDriftPreset report (JSON for every window, HTML for some).
Outputs in out_dir: drift_summary.json, drift_reference.json (for services/monitor),
evidently/<window>.{json,html}. Logged to MLflow if configured.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud.config import load_config
from fraud.drift import (
    build_reference,
    column_psi,
    decide_retrain,
    drift_table,
    live_performance,
    time_windows,
)
from fraud.metrics import average_precision

SCORE = "score"


def find_dir(setting: str, marker: str, root: str = "/kaggle/input") -> Path | None:
    """`setting` if not 'auto', else the parent folder of the first `marker` under `root`."""
    if setting != "auto":
        return Path(setting)
    for depth in range(1, 7):
        hits = sorted(glob.glob(os.path.join(root, *["*"] * depth, marker)))
        if hits:
            return Path(hits[0]).parent
    return None


def load_features(input_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    summary = json.loads((input_dir / "features_summary.json").read_text())
    splits = {
        n: pd.read_parquet(input_dir / "features" / f"{n}.parquet")
        for n in ("train", "val", "test")
    }
    return splits, summary


def score_frames(model_dir: Path, frames: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    import xgboost as xgb

    from fraud.model import FeatureSpec

    spec = FeatureSpec.load(model_dir / "spec.json")
    booster = xgb.Booster()
    booster.load_model(model_dir / "model.json")
    meta = json.loads((model_dir / "threshold.json").read_text())
    scores = {
        n: booster.inplace_predict(spec.transform(df)).astype("float64") for n, df in frames.items()
    }
    return scores, meta


def evidently_columns(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Plain dtypes for Evidently: float64 numerics, string categoricals with 'NA' for missing."""
    out = pd.DataFrame(index=df.index)
    for c in numeric:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in categorical:
        out[c] = df[c].astype("object").where(df[c].notna(), "NA").astype(str)
    return out


def run_evidently(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    features_num: list[str],
    features_cat: list[str],
    extra_num: list[str],
    extra_cat: list[str],
    drift_share: float,
):
    """DataDriftPreset on the inputs + ValueDrift on the score (prediction) and label (target)."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset

    num, cat = [*features_num, *extra_num], [*features_cat, *extra_cat]
    definition = DataDefinition(numerical_columns=num, categorical_columns=cat)
    report = Report(
        [
            DataDriftPreset(columns=[*features_num, *features_cat], drift_share=drift_share),
            *[ValueDrift(column=c) for c in [*extra_num, *extra_cat]],
        ]
    )
    return report.run(
        Dataset.from_pandas(evidently_columns(cur, num, cat), data_definition=definition),
        Dataset.from_pandas(evidently_columns(ref, num, cat), data_definition=definition),
    )


def evidently_summary(snapshot) -> dict[str, Any]:
    """Drifted-column count/share and each column's drift score from a snapshot's dict()."""
    out: dict[str, Any] = {"drifted_count": None, "drift_share": None, "columns": {}}
    for m in snapshot.dict()["metrics"]:
        name = str(m.get("metric_name") or m.get("id") or "")
        value = m.get("value")
        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            out["drifted_count"] = value.get("count")
            out["drift_share"] = value.get("share")
        elif name.startswith("ValueDrift"):
            config = m.get("config") or {}
            col = config.get("column")
            if col is None:
                found = re.search(r"column=([^,)]+)", name)
                col = found.group(1) if found else name
            method = config.get("method")
            if method is None:
                found = re.search(r"method=([^,)]+)", name)
                method = found.group(1) if found else None
            out["columns"][col] = {"score": value, "method": method}
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/drift.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default="/kaggle/input")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    off, pcfg, rcfg = cfg["offline"], cfg["psi"], cfg["retrain"]
    out_dir = Path(off["out_dir"])
    (out_dir / "evidently").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    input_dir = find_dir(off["input_dir"], "features_summary.json", args.root)
    if input_dir is None:
        raise FileNotFoundError(f"features_summary.json not found under {args.root}")
    splits, fsum = load_features(input_dir)
    target, t_col = "isFraud", "TransactionDT"
    cols = fsum["model_columns"]
    numeric = [*cols["numeric"], *cols["features"]]
    categorical = list(cols["categorical"])
    current = pd.concat([splits["val"], splits["test"]], ignore_index=True)
    current = current.sort_values(t_col, kind="stable").reset_index(drop=True)
    train = splits["train"]
    if args.smoke:
        train = train.sample(n=min(len(train), off["smoke"]["train_rows"]), random_state=0)
        current = current.iloc[: off["smoke"]["current_rows"]].reset_index(drop=True)
    print(f"reference {len(train):,} rows, current {len(current):,} rows", flush=True)

    model_dir = find_dir(off["model_dir"], "threshold.json", args.root)
    threshold, model_meta = None, {}
    if model_dir is not None and (model_dir / "model.json").exists():
        scores, model_meta = score_frames(model_dir, {"val": splits["val"], "current": current})
        threshold = float(model_meta["threshold"])
        current[SCORE] = scores["current"]
        val_scores = scores["val"]
        print(f"scored with {model_dir} (threshold {threshold:.5f})", flush=True)
    else:
        val_scores = None
        print("no F3 model found: skipping score drift and performance", flush=True)

    reference = build_reference(train, numeric, categorical, pcfg["bins"], pcfg["max_categories"])
    reference[t_col] = build_reference(train, [t_col], [], pcfg["bins"])[t_col]
    # Live monitor reference: the inputs it sees in the case store + the score on val.
    live_ref = build_reference(
        train, ["TransactionAmt", "hour"], ["ProductCD"], pcfg["bins"], pcfg["max_categories"]
    )
    expected_alert_rate = None
    if val_scores is not None:
        score_ref = build_reference(pd.DataFrame({SCORE: val_scores}), [SCORE], [], pcfg["bins"])
        reference[SCORE] = score_ref[SCORE]
        live_ref["p_xgb"] = {**score_ref[SCORE], "column": "p_xgb"}  # case-store name
        expected_alert_rate = float((val_scores > threshold).mean())
    (out_dir / "drift_reference.json").write_text(
        json.dumps(
            {
                "columns": live_ref,
                "expected_alert_rate": expected_alert_rate,
                "threshold_xgb": threshold,
                "source": "F6 kaggle/drift: train inputs, val XGBoost scores",
            },
            indent=2,
        )
    )

    exclude = list(cfg["exclude_from_share"])
    features = [c for c in [*numeric, *categorical] if c in current]
    windows = [
        {"name": "all", "day_from": None, "day_to": None, "index": np.arange(len(current))},
        *time_windows(current[t_col], off["window_days"], off["min_window_rows"]),
    ]
    ev_cfg = off["evidently"]
    ev_ref, ev_version = None, None
    if ev_cfg["enabled"]:
        ev_ref = train.sample(n=min(len(train), ev_cfg["reference_rows"]), random_state=0)
        if val_scores is not None:
            # The score column for the Evidently reference comes from val (see score_reference).
            ref_scores = np.random.default_rng(0).choice(val_scores, len(ev_ref))
            ev_ref = ev_ref.assign(**{SCORE: ref_scores})
        try:
            import evidently

            ev_version = evidently.__version__
        except ImportError:
            print("evidently not installed: skipping Evidently reports", flush=True)
    html_names = set(ev_cfg["html"])
    if "first" in html_names and len(windows) > 1:
        html_names.add(windows[1]["name"])
    if "last" in html_names:
        html_names.add(windows[-1]["name"])

    results = []
    for w in windows:
        t0 = time.time()
        cur = current.iloc[w["index"]]
        table = drift_table(
            cur, reference, pcfg["feature_threshold"], pcfg["eps"], [*features, t_col], exclude
        )
        row: dict[str, Any] = {
            "window": w["name"],
            "day_from": w["day_from"]
            if w["day_from"] is not None
            else int(current[t_col].min() // 86400),
            "day_to": w["day_to"]
            if w["day_to"] is not None
            else int(current[t_col].max() // 86400),
            "rows": int(len(cur)),
            "fraud_rate": float(cur[target].mean()),
            "drift_share": round(table["drift_share"], 4),
            "drifted": table["drifted"],
            "top_psi": dict(list(table["psi"].items())[:8]),
        }
        metrics: dict[str, Any] = {"n": len(cur), "drift_share": table["drift_share"]}
        if val_scores is not None:
            y, s = cur[target].to_numpy(), cur[SCORE].to_numpy()
            perf = live_performance(s > threshold, y, rcfg["alert_budget"], expected_alert_rate)
            row["score_psi"] = round(column_psi(s, reference[SCORE], pcfg["eps"]), 5)
            row["pr_auc"] = round(average_precision(y, s), 4)
            row.update(
                {
                    k: (round(v, 5) if isinstance(v, float) else v)
                    for k, v in perf.items()
                    if k
                    in (
                        "alert_rate",
                        "precision",
                        "recall",
                        "fpr",
                        "fpr_budget_ratio",
                        "alert_rate_ratio",
                    )
                }
            )
            metrics.update(perf)
            metrics.update(score_psi=row["score_psi"], pr_auc=row["pr_auc"])
        row["decision"] = decide_retrain(metrics, rcfg)
        if ev_ref is not None and ev_version:
            ev_num = [c for c in numeric if c in cur and c not in exclude]
            ev_cat = [c for c in categorical if c in cur and c not in exclude]
            extra_num = [SCORE] if SCORE in cur else []
            snap = run_evidently(
                ev_ref, cur, ev_num, ev_cat, extra_num, [target], ev_cfg["drift_share"]
            )
            snap.save_json(str(out_dir / "evidently" / f"{w['name']}.json"))
            if w["name"] in html_names:
                snap.save_html(str(out_dir / "evidently" / f"{w['name']}.html"))
            row["evidently"] = evidently_summary(snap)
        row["seconds"] = round(time.time() - t0, 1)
        results.append(row)
        ev = row.get("evidently", {})
        print(
            f"{w['name']}: rows {row['rows']:,}, fraud {row['fraud_rate']:.4f}, PSI drift share "
            f"{row['drift_share']:.2f}, Evidently share {ev.get('drift_share')}, score PSI "
            f"{row.get('score_psi')}, PR-AUC {row.get('pr_auc')}, FPR {row.get('fpr')}, "
            f"retrain {row['decision']['retrain']} {row['decision']['reasons']}",
            flush=True,
        )

    summary = {
        "smoke": bool(args.smoke),
        "git_sha": os.environ.get("GIT_SHA"),
        "evidently_version": ev_version,
        "reference_rows": int(len(train)),
        "current_rows": int(len(current)),
        "model": model_meta,
        "expected_alert_rate": expected_alert_rate,
        "exclude_from_share": exclude,
        "thresholds": {"psi_feature": pcfg["feature_threshold"], **rcfg},
        "windows": results,
        "retrain_windows": [r["window"] for r in results if r["decision"]["retrain"]],
        "seconds": round(time.time() - t_start, 1),
    }
    (out_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if off.get("mlflow"):
        log_mlflow(summary, out_dir, args.smoke)
    return summary


def log_mlflow(summary: dict[str, Any], out_dir: Path, smoke: bool) -> None:
    try:
        import mlflow

        from fraud.tracking import configure

        mlflow.set_tracking_uri(configure(load_config("configs/mlflow.yaml")))
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
        with mlflow.start_run(run_name=f"f6-drift-{'smoke' if smoke else 'full'}"):
            mlflow.log_params(
                {
                    "git_sha": summary["git_sha"],
                    "smoke": smoke,
                    "evidently": summary["evidently_version"],
                }
            )
            for step, r in enumerate(r for r in summary["windows"] if r["window"] != "all"):
                vals = {
                    k: r[k]
                    for k in (
                        "drift_share",
                        "score_psi",
                        "pr_auc",
                        "fpr",
                        "precision",
                        "fraud_rate",
                    )
                    if r.get(k) is not None
                }
                vals["retrain"] = float(r["decision"]["retrain"])
                if r.get("evidently", {}).get("drift_share") is not None:
                    vals["evidently_drift_share"] = r["evidently"]["drift_share"]
                mlflow.log_metrics(vals, step=step)
            mlflow.log_artifact(str(out_dir / "drift_summary.json"))
            mlflow.log_artifact(str(out_dir / "drift_reference.json"))
            for html in sorted((out_dir / "evidently").glob("*.html")):
                mlflow.log_artifact(str(html), artifact_path="evidently")
            summary["mlflow_run_id"] = mlflow.active_run().info.run_id
        (out_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"MLflow run {summary['mlflow_run_id']}", flush=True)
    except Exception as exc:  # tracking must never fail the report
        print(f"MLflow logging failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
