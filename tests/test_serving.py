import json

import numpy as np
import pandas as pd
import pytest

from fraud.cases import CaseStore
from fraud.model import FeatureSpec
from fraud.store import MemoryStore


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amt": rng.lognormal(3, 1, n).astype("float32"),
            "cnt": rng.choice([np.nan, 0.0, 1.0, 5.0], n),
            "flag": rng.choice([True, False], n),
            "prod": rng.choice(np.array(["W", "C", "H", None], dtype=object), n),
            "y": (rng.random(n) < 0.2).astype(int),
        }
    )


@pytest.mark.parametrize("medians", [None, [10.0, 1.0, 0.0, -1.0]])
def test_transform_row_matches_batch(medians):
    df = _frame()
    spec = FeatureSpec.fit(df, ["amt", "cnt", "flag"], ["prod"], min_count=1)
    spec.medians = medians
    batch = spec.transform(df)
    rows = np.vstack([spec.transform_row(r) for r in df.to_dict("records")])
    assert np.array_equal(batch, rows, equal_nan=True)


def test_scorer_matches_batch_xgboost(tmp_path):
    xgb = pytest.importorskip("xgboost")
    from fraud.scoring import Scorer

    df = _frame(600)
    spec = FeatureSpec.fit(df, ["amt", "cnt", "flag"], ["prod"], min_count=1)
    x = spec.transform(df)
    params = {"objective": "binary:logistic", "max_depth": 3}
    booster = xgb.train(params, xgb.DMatrix(x, label=df["y"]), num_boost_round=20)
    booster.save_model(tmp_path / "model.json")
    spec.save(tmp_path / "spec.json")
    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.3, "experiment": "t"}))
    scorer = Scorer(tmp_path)
    expected = booster.predict(xgb.DMatrix(x))
    rows = df.to_dict("records")
    got = [scorer.score(r, {"client_key": "c"})["score"] for r in rows[:50]]
    assert np.allclose(got, expected[:50], atol=1e-6)
    out = scorer.score(rows[0], {"client_key": "c"})
    assert out["alert"] == (out["score"] > 0.3) and out["gnn_risk"] is None


def test_scorer_blends_known_client_risk(tmp_path):
    xgb = pytest.importorskip("xgboost")
    from fraud.scoring import Scorer

    df = _frame(300)
    spec = FeatureSpec.fit(df, ["amt", "cnt", "flag"], ["prod"], min_count=1)
    dtrain = xgb.DMatrix(spec.transform(df), label=df["y"])
    xgb.train({"objective": "binary:logistic"}, dtrain, 5).save_model(tmp_path / "model.json")
    spec.save(tmp_path / "spec.json")
    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.5}))
    (tmp_path / "blend.json").write_text(json.dumps({"weight_xgb": 0.5, "threshold": 0.4}))
    pd.DataFrame({"client_key": ["risky"], "gnn_risk": [0.99]}).to_parquet(
        tmp_path / "client_risk.parquet"
    )
    scorer = Scorer(tmp_path)
    row = df.to_dict("records")[0]
    plain = scorer.score(row, {"client_key": "new"})
    risky = scorer.score(row, {"client_key": "risky"})
    assert plain["score"] == plain["p_xgb"] and risky["score"] > plain["score"]
    assert scorer.info()["threshold"] == 0.4


def test_memory_store_roundtrip():
    store = MemoryStore()
    assert store.get("a") is None
    store.push([{"client_key": "a", "TransactionID": 1, "cnt_1h": 2.0}])
    store.push([{"client_key": "a", "TransactionID": 2, "cnt_1h": 3.0}])
    assert store.get("a")["TransactionID"] == 2


def test_case_store_flow(tmp_path):
    db = CaseStore(tmp_path / "cases.db")
    base = {"event_time": 1.0, "client_key": "c", "amount": 10.0, "p_xgb": 0.1}
    db.insert(
        [
            {**base, "transaction_id": 1, "score": 0.9, "alert": True, "true_label": 1},
            {**base, "transaction_id": 2, "score": 0.1, "alert": False, "true_label": 0},
            {**base, "transaction_id": 3, "score": 0.7, "alert": True, "true_label": 0},
        ]
    )
    assert [a["transaction_id"] for a in db.alerts("open")] == [1, 3]
    db.set_status(1, "confirmed")
    db.set_status(3, "dismissed")
    with pytest.raises(KeyError):
        db.set_status(2, "confirmed")  # not an alert
    with pytest.raises(ValueError):
        db.set_status(1, "maybe")
    s = db.stats()
    assert s["n"] == 3 and s["alerts"] == 2 and s["caught"] == 1
    assert s["confirmed"] == 1 and s["dismissed"] == 1 and s["open"] == 0
    assert db.alerts("confirmed")[0]["feedback_label"] == 1
