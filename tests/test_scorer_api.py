import json

import numpy as np
import pandas as pd
import pytest

from fraud.config import load_config
from fraud.features import feature_names
from fraud.model import FeatureSpec

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
xgb = pytest.importorskip("xgboost")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from services.scorer import app as app_module

    names = feature_names(load_config("configs/features.yaml")["windows"])
    rng = np.random.default_rng(0)
    df = pd.DataFrame({n: rng.random(200) for n in names})
    df["TransactionAmt"] = rng.lognormal(3, 1, 200)
    df["ProductCD"] = rng.choice(["W", "C"], 200)
    y = (rng.random(200) < 0.2).astype(int)
    spec = FeatureSpec.fit(df, ["TransactionAmt", *names], ["ProductCD"], min_count=1)
    dtrain = xgb.DMatrix(spec.transform(df), label=y)
    xgb.train({"objective": "binary:logistic"}, dtrain, 10).save_model(tmp_path / "model.json")
    spec.save(tmp_path / "spec.json")
    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.5}))
    cfg = load_config("configs/serving.yaml")
    cfg["model_dir"] = str(tmp_path)
    (tmp_path / "serving.yaml").write_text(json.dumps(cfg))  # JSON is valid YAML
    monkeypatch.setenv("SERVING_CONFIG", str(tmp_path / "serving.yaml"))
    monkeypatch.setenv("ONLINE_STORE", "memory")
    app_module.STATE.clear()
    with TestClient(app_module.app) as c:
        yield c, app_module.STATE, names


def _tx(tid=7):
    return {
        "TransactionID": tid,
        "TransactionDT": 86400 * 3,
        "TransactionAmt": 50.0,
        "card1": 1234,
        "addr1": 299.0,
        "D1": 1.0,
        "ProductCD": "W",
    }


def test_health_and_model(client):
    c, _, _ = client
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/model").json()["n_features"] > 0


def test_score_needs_matching_features(client):
    c, state, names = client
    assert c.post("/score", json=_tx()).status_code == 409  # nothing pushed yet
    client_key = "1234|299|2"  # card1 | addr1 | day(3) - D1(1)
    zeros = dict.fromkeys(names, 0.0)
    state["store"].push([{"client_key": client_key, "TransactionID": 6, **zeros}])
    assert c.post("/score", json=_tx()).status_code == 409  # another transaction's features
    state["store"].push([{"client_key": client_key, "TransactionID": 7, **zeros}])
    r = c.post("/score", json=_tx())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["client_key"] == client_key and 0 <= body["score"] <= 1
    assert body["alert"] == (body["score"] > body["threshold"])
