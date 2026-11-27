"""F3: train fraud classifiers on the F2 features, log to MLflow, register the best XGBoost.

Runs on Kaggle (kaggle/train) with the F2 output and the secrets dataset attached:
    python -m scripts.train --config configs/train.yaml [--smoke]
Experiments (configs/train.yaml): logistic regression, XGBoost on raw fields only, XGBoost on
raw + F2 features with class weights, and the same with SMOTE. Each is an MLflow run; the alert
threshold is chosen on val for a 1% false-positive budget; the best val PR-AUC model is saved
to out_dir (model.json, spec.json, threshold.json) and registered in MLflow.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud.config import load_config
from fraud.metrics import evaluate, threshold_at_fpr
from fraud.model import FeatureSpec
from fraud.tracking import configure


def find_features(input_dir: str, search_root: str = "/kaggle/input") -> Path:
    if input_dir != "auto":
        return Path(input_dir)
    for depth in range(1, 6):
        pattern = os.path.join(search_root, *["*"] * depth, "features", "train.parquet")
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0]).parent.parent
    raise FileNotFoundError(f"features/train.parquet not found under {search_root}")


def columns_for(summary: dict, group: str) -> tuple[list[str], list[str]]:
    cols = summary["model_columns"]
    numeric = list(cols["numeric"])
    if group == "full":
        numeric += cols["features"]
    elif group != "raw":
        raise ValueError(f"unknown feature group {group}")
    return numeric, list(cols["categorical"])


def impute(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
    out = x.copy()
    rows, cols = np.where(np.isnan(out))
    out[rows, cols] = medians[cols]
    return out


def fit_logreg(xtr, ytr, spec: FeatureSpec, cfg: dict, seed: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    n_num = len(spec.numeric)
    signed_log = FunctionTransformer(lambda a: np.sign(a) * np.log1p(np.abs(a)))
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    signed_log,
                    StandardScaler(),
                ),
                list(range(n_num)),
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value=-1),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                list(range(n_num, xtr.shape[1])),
            ),
        ]
    )
    model = Pipeline(
        [
            ("pre", pre),
            (
                "lr",
                LogisticRegression(
                    C=cfg["C"], max_iter=cfg["max_iter"], class_weight="balanced", random_state=seed
                ),
            ),
        ]
    )
    model.fit(xtr, ytr)
    return model, lambda x: model.predict_proba(x)[:, 1], {}


def fit_xgb(xtr, ytr, xva, yva, cfg: dict, imbalance: str, smote_cfg: dict, seed: int):
    import xgboost as xgb

    params = {k: v for k, v in cfg.items() if k != "early_stopping_rounds"}
    info: dict[str, Any] = {}
    transform = None
    if imbalance == "weight":
        params["scale_pos_weight"] = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    elif imbalance == "smote":
        from imblearn.over_sampling import SMOTE

        medians = np.nanmedian(xtr, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        sm = SMOTE(
            sampling_strategy=smote_cfg["sampling_strategy"],
            k_neighbors=smote_cfg["k_neighbors"],
            random_state=seed,
        )
        xtr, ytr = sm.fit_resample(impute(xtr, medians), ytr)
        xva = impute(xva, medians)
        info["smote_rows"] = int(len(ytr))
        info["medians"] = medians

        def transform(x, m=medians):
            return impute(x, m)

    else:
        raise ValueError(f"unknown imbalance handling {imbalance}")
    model = xgb.XGBClassifier(
        **params,
        early_stopping_rounds=cfg["early_stopping_rounds"],
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=200)
    info["best_iteration"] = int(model.best_iteration)
    info["scale_pos_weight"] = params.get("scale_pos_weight")

    def predict(x):
        return model.predict_proba(transform(x) if transform else x)[:, 1]

    return model, predict, info


def pr_curve_figure(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (y, s) in curves.items():
        order = np.argsort(-s, kind="mergesort")
        tp = np.cumsum(y[order])
        precision = tp / np.arange(1, len(y) + 1)
        recall = tp / max(y.sum(), 1)
        ax.plot(recall, precision, lw=1.2, label=name)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall (val)")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--mlflow-config", default="configs/mlflow.yaml")
    ap.add_argument("--input-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these experiment names")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    feat_dir = find_features(args.input_dir or cfg["input_dir"])
    out_dir = Path(args.out_dir or cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    target, seed, max_fpr = cfg["target"], cfg["seed"], cfg["max_fpr"]
    summary_f2 = json.loads((feat_dir / "features_summary.json").read_text())

    splits = {
        n: pd.read_parquet(feat_dir / "features" / f"{n}.parquet") for n in ("train", "val", "test")
    }
    if args.smoke:
        splits["train"] = splits["train"].iloc[: cfg["smoke"]["train_rows"]]
    y = {n: d[target].to_numpy().astype(int) for n, d in splits.items()}
    print({n: (len(d), int(y[n].sum())) for n, d in splits.items()}, flush=True)

    tracking_uri = configure(load_config(args.mlflow_config))
    import mlflow

    try:
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
    except Exception as exc:  # remote unreachable: keep the run, log locally
        print(f"MLflow remote failed ({type(exc).__name__}: {exc}); using file store", flush=True)
        tracking_uri = load_config(args.mlflow_config)["fallback_uri"]
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    xgb_cfg = dict(cfg["xgb"])
    if args.smoke:
        xgb_cfg["n_estimators"] = cfg["smoke"]["n_estimators"]
    git_sha = os.environ.get("GIT_SHA", "unknown")
    results, curves, models = [], {}, {}
    for exp in cfg["experiments"]:
        if args.only and exp["name"] not in args.only:
            continue
        numeric, categorical = columns_for(summary_f2, exp["features"])
        spec = FeatureSpec.fit(splits["train"], numeric, categorical, cfg["category_min_count"])
        x = {n: spec.transform(d) for n, d in splits.items()}
        t0 = time.time()
        with mlflow.start_run(run_name=exp["name"] + ("-smoke" if args.smoke else "")):
            mlflow.set_tags({"milestone": "F3", "git_sha": git_sha, "smoke": str(args.smoke)})
            mlflow.log_params({**exp, "n_features": len(spec.columns), "max_fpr": max_fpr})
            if exp["model"] == "logreg":
                model, predict, info = fit_logreg(x["train"], y["train"], spec, cfg["logreg"], seed)
            else:
                mlflow.log_params({f"xgb_{k}": v for k, v in xgb_cfg.items()})
                model, predict, info = fit_xgb(
                    x["train"],
                    y["train"],
                    x["val"],
                    y["val"],
                    xgb_cfg,
                    exp["imbalance"],
                    cfg["smote"],
                    seed,
                )
            seconds = time.time() - t0
            scores = {n: predict(x[n]) for n in ("val", "test")}
            thr = threshold_at_fpr(y["val"], scores["val"], max_fpr)
            metrics = {
                **evaluate(y["val"], scores["val"], thr, "val_"),
                **evaluate(y["test"], scores["test"], thr, "test_"),
                "train_seconds": seconds,
                "threshold": thr,
            }
            metrics.update({k: v for k, v in info.items() if isinstance(v, (int, float))})
            if "medians" in info:  # serve with the same imputation as training
                spec.medians = [float(m) for m in info["medians"]]
            mlflow.log_metrics(metrics)
            run_id = mlflow.active_run().info.run_id
            if exp["model"] == "xgb":
                imp = pd.Series(model.feature_importances_, index=spec.columns)
                imp = imp.sort_values(ascending=False)
                imp_path = out_dir / f"importance_{exp['name']}.csv"
                imp.to_csv(imp_path, header=["gain_share"])
                mlflow.log_artifact(str(imp_path))
        print(exp["name"], json.dumps({k: round(v, 4) for k, v in metrics.items()}), flush=True)
        results.append({"name": exp["name"], "run_id": run_id, **metrics})
        curves[exp["name"]] = (y["val"], scores["val"])
        models[exp["name"]] = (model, spec, thr, exp)

    fig = pr_curve_figure(curves, out_dir / "pr_curve_val.png")
    best = max(results, key=lambda r: r["val_pr_auc"])
    # Serving model: best XGBoost on val PR-AUC (its spec carries any SMOTE imputation medians).
    xgb_runs = [r for r in results if models[r["name"]][3]["model"] == "xgb"]
    serving = max(xgb_runs, key=lambda r: r["val_pr_auc"]) if xgb_runs else None
    registered = None
    if serving:
        model, spec, thr, exp = models[serving["name"]]
        model.save_model(out_dir / "model.json")
        spec.save(out_dir / "spec.json")
        meta = {"threshold": thr, "max_fpr": max_fpr, "experiment": serving["name"]}
        (out_dir / "threshold.json").write_text(json.dumps(meta, indent=2))
        with mlflow.start_run(run_id=serving["run_id"]):
            mlflow.log_artifact(str(fig))
            for f in ("model.json", "spec.json", "threshold.json"):
                mlflow.log_artifact(str(out_dir / f), artifact_path="serving")
            try:
                import mlflow.xgboost

                name = None if args.smoke else cfg["register_name"]
                logged = mlflow.xgboost.log_model(
                    model, artifact_path="model", registered_model_name=name
                )
                registered = name
                print(f"logged model {logged.model_uri} (registered as {name})", flush=True)
            except Exception as exc:
                print(f"model logging/registration failed: {type(exc).__name__}: {exc}", flush=True)

    summary = {
        "smoke": bool(args.smoke),
        "tracking_uri": tracking_uri,
        "git_sha": git_sha,
        "rows": {n: int(len(d)) for n, d in splits.items()},
        "frauds": {n: int(v.sum()) for n, v in y.items()},
        "results": results,
        "best": best["name"],
        "serving": serving["name"] if serving else None,
        "registered": registered,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
