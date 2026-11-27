import json

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud.config import load_config


def _split(n, day0, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(day0 * 86_400, (day0 + 21) * 86_400, n))
    amt = rng.lognormal(4, 1.0, n) * scale
    y = (rng.random(n) < 0.04).astype(int)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n) + day0 * 100_000,
            "TransactionDT": t,
            "isFraud": y,
            "TransactionAmt": amt,
            "hour": (t // 3600) % 24,
            "C1": rng.poisson(1 + 3 * y).astype(float),
            "ProductCD": pd.array(rng.choice(["W", "C"], n, p=[0.8, 0.2]), dtype="string"),
            "cnt_7d": rng.poisson(2, n).astype("float32"),
        }
    )


@pytest.fixture
def features_dir(tmp_path):
    pytest.importorskip("pyarrow")
    root = tmp_path / "input" / "f2"
    (root / "features").mkdir(parents=True)
    _split(6000, 0, 0).to_parquet(root / "features" / "train.parquet")
    _split(3000, 21, 1).to_parquet(root / "features" / "val.parquet")
    _split(3000, 42, 2, scale=3.0).to_parquet(root / "features" / "test.parquet")  # drifted
    summary = {
        "model_columns": {
            "numeric": ["TransactionAmt", "hour", "C1"],
            "categorical": ["ProductCD"],
            "features": ["cnt_7d"],
        }
    }
    (root / "features_summary.json").write_text(json.dumps(summary))
    return tmp_path


def _config(tmp_path, **offline):
    cfg = load_config("configs/drift.yaml")
    cfg["offline"].update(
        out_dir=str(tmp_path / "out"), min_window_rows=200, mlflow=False, **offline
    )
    cfg["offline"]["evidently"]["enabled"] = False
    cfg["retrain"]["min_window"] = 100
    path = tmp_path / "drift.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_drift_report_without_model(features_dir):
    from scripts import drift_report

    cfg = _config(features_dir)
    s = drift_report.main(["--config", str(cfg), "--root", str(features_dir / "input")])
    names = [w["window"] for w in s["windows"]]
    assert names[0] == "all" and len(names) == 1 + 6  # 42 days of val + test in weekly windows
    first, last = s["windows"][1], s["windows"][-1]
    assert first["drift_share"] == 0.0 and not first["decision"]["retrain"]
    assert "TransactionAmt" in last["drifted"] and last["decision"]["retrain"]
    assert "score_psi" not in last
    ref = json.loads((features_dir / "out" / "drift_reference.json").read_text())
    assert set(ref["columns"]) == {"TransactionAmt", "hour", "ProductCD"}


def test_drift_report_with_model(features_dir):
    xgb = pytest.importorskip("xgboost")
    from fraud.model import FeatureSpec
    from scripts import drift_report

    train = pd.read_parquet(features_dir / "input" / "f2" / "features" / "train.parquet")
    spec = FeatureSpec.fit(train, ["TransactionAmt", "hour", "C1", "cnt_7d"], ["ProductCD"], 1)
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3).fit(
        spec.transform(train), train["isFraud"]
    )
    mdir = features_dir / "input" / "f3"
    mdir.mkdir()
    model.save_model(mdir / "model.json")
    spec.save(mdir / "spec.json")
    (mdir / "threshold.json").write_text(json.dumps({"threshold": 0.3, "experiment": "xgb"}))

    s = drift_report.main(
        ["--config", str(_config(features_dir)), "--root", str(features_dir / "input")]
    )
    w1 = s["windows"][1]
    for key in ("score_psi", "pr_auc", "precision", "fpr", "alert_rate"):
        assert key in w1
    assert w1["score_psi"] < 0.1  # val period: same distribution as the score reference
    assert s["expected_alert_rate"] is not None
    ref = json.loads((features_dir / "out" / "drift_reference.json").read_text())
    assert "p_xgb" in ref["columns"] and ref["threshold_xgb"] == 0.3


class _Snap:
    def dict(self):
        return {
            "metrics": [
                {
                    "metric_name": "DriftedColumnsCount(drift_share=0.3)",
                    "config": {},
                    "value": {"count": 2.0, "share": 0.4},
                },
                {
                    "metric_name": "ValueDrift(column=TransactionAmt,method=Wasserstein distance)",
                    "config": {"column": "TransactionAmt"},
                    "value": 0.31,
                },
                {
                    "metric_name": "ValueDrift(column=score,method=K-S p_value,threshold=0.05)",
                    "config": {},
                    "value": 0.01,
                },
            ],
            "tests": [],
        }


def test_evidently_summary_parsing():
    from scripts.drift_report import evidently_summary

    out = evidently_summary(_Snap())
    assert out["drifted_count"] == 2.0 and out["drift_share"] == 0.4
    assert out["columns"]["TransactionAmt"]["score"] == 0.31
    assert out["columns"]["score"] == {"score": 0.01, "method": "K-S p_value"}


def test_evidently_report_runs():
    pytest.importorskip("evidently")
    from scripts.drift_report import evidently_summary, run_evidently

    ref, cur = _split(2000, 0, 0), _split(2000, 21, 1, scale=3.0)
    ref["score"], cur["score"] = np.linspace(0, 1, 2000), np.linspace(0, 1, 2000) ** 2
    snap = run_evidently(
        ref, cur, ["TransactionAmt", "C1"], ["ProductCD"], ["score"], ["isFraud"], 0.3
    )
    out = evidently_summary(snap)
    assert out["drifted_count"] >= 1
    assert {"TransactionAmt", "score", "isFraud"} <= set(out["columns"])
