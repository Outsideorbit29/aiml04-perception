"""Smoke test of the Streamlit app with Streamlit's AppTest (skipped where streamlit is absent)."""

import pandas as pd
import pytest

from fraud.cases import CaseStore

pytest.importorskip("streamlit")
pytest.importorskip("pyarrow")


def _rows():
    rows = []
    for i in range(30):
        alert = i % 5 == 0
        rows.append(
            {
                "transaction_id": i,
                "event_time": 1000 + i,
                "client_key": f"c{i % 4}",
                "card_key": f"k{i % 3}",
                "device_key": "dev" if i % 2 else None,
                "amount": 12.5 + i,
                "product": "W",
                "score": 0.9 - i / 100 if alert else 0.01,
                "p_xgb": 0.8 if alert else 0.01,
                "gnn_risk": 0.95 if i % 4 == 0 else None,
                "alert": alert,
                "latency_ms": 5.0 + i,
                "true_label": int(alert and i % 10 == 0),
            }
        )
    return rows


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db = tmp_path / "cases.db"
    store = CaseStore(db)
    store.insert(_rows())
    store.close()
    rings = tmp_path / "rings.parquet"
    pd.DataFrame(
        {"client_key": ["c0", "c1"], "ring_id": [3, 3], "ring_size": [2, 2], "ring_devices": [1, 1]}
    ).to_parquet(rings)
    monkeypatch.setenv("CASES_DB", str(db))
    monkeypatch.setenv("RINGS_PATH", str(rings))
    monkeypatch.setenv("STATS_PATH", str(tmp_path / "stats.json"))
    monkeypatch.setenv("SCORER_URL", "http://127.0.0.1:9")  # unreachable: model tab shows an error
    monkeypatch.setenv("HTTP_TIMEOUT_S", "1")
    return db


def test_app_renders_and_confirms_an_alert(env):
    from streamlit.testing.v1 import AppTest

    from fraud import dashboard_data as dd

    at = AppTest.from_file("services/dashboard/app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "Transactions scored" in labels and "Open" in labels
    assert at.metric[labels.index("Transactions scored")].value == "30"
    confirm = [b for b in at.button if b.label == "Confirm fraud"]
    assert confirm, "alert queue shows action buttons"
    confirm[0].click().run()
    assert not at.exception, at.exception
    sc = dd.status_counts(dd.connect(env))
    assert sc["confirmed"] == 1 and sc["open"] == 5
