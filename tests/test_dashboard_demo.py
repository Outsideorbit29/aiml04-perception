"""F7.3 demo mode: synthetic case store / rings / stats for the public Streamlit Cloud app."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from fraud import dashboard_data as dd
from fraud.cases import SCHEMA
from fraud.demo_data import DEMO_ID_START, make_demo, write_case_db

NOW = 1_800_000_000.0


def _table_columns(conn: sqlite3.Connection) -> list[str]:
    return [r[1] for r in conn.execute("PRAGMA table_info(scored)")]


def test_make_demo_is_deterministic():
    a, b = make_demo(now=NOW), make_demo(now=NOW)
    assert a["scored"].equals(b["scored"])
    assert a["rings"].equals(b["rings"])
    assert a["model"] == b["model"] and a["stats"] == b["stats"]
    assert not make_demo(now=NOW, seed=1)["scored"].equals(a["scored"])


def test_make_demo_looks_like_the_live_stream():
    d = make_demo(n=4000, now=NOW)
    s = d["scored"]
    assert len(s) == 4000 and s["transaction_id"].is_unique
    assert s["transaction_id"].min() == DEMO_ID_START  # outside the IEEE-CIS id range
    assert s["client_key"].str.startswith("demo-").all()
    assert 0.025 <= s["alert"].mean() <= 0.035
    assert 0.02 <= s["true_label"].mean() <= 0.06
    alerts = s[s["alert"] == 1]
    assert (alerts["score"] >= d["model"]["threshold"]).all()
    assert (s.loc[s["alert"] == 0, "score"] < d["model"]["threshold"]).all()
    assert set(alerts["status"]) == {"open", "confirmed", "dismissed"}
    assert s.loc[s["alert"] == 0, "status"].isna().all()
    assert s["scored_at"].max() == NOW and s["latency_ms"].gt(0).all()
    rings = d["rings"]
    assert rings["ring_id"].nunique() >= 3
    assert rings["client_key"].isin(s["client_key"]).any()


def test_write_case_db_matches_case_store_schema(tmp_path):
    d = make_demo(n=1500, now=NOW)
    path = tmp_path / "cases.db"
    write_case_db(d["scored"], path)
    write_case_db(d["scored"], path)  # rebuild overwrites instead of failing
    conn = dd.connect(path)
    ref = sqlite3.connect(":memory:")
    ref.executescript(SCHEMA)
    assert _table_columns(conn) == _table_columns(ref)
    ov = dd.overview(conn, 0.01, 60, 5000, 0.027)
    assert ov["scored"] == 1500 and ov["alerts"] == int(d["scored"]["alert"].sum())
    assert ov["last_scored_at"] == NOW and ov["latency_p95_ms"] > ov["latency_p50_ms"]
    sc = dd.status_counts(conn)
    assert sc["reviewed"] > 0 and sc["open"] > 0
    assert not dd.alert_queue(conn, "open").empty
    conn.close()
    dd.set_case_status(path, int(dd.alert_queue(dd.connect(path), "open").iloc[0, 0]), "confirmed")
    conn = dd.connect(path)
    assert dd.status_counts(conn)["confirmed"] == sc["confirmed"] + 1
    conn.close()


def test_write_demo_files(tmp_path):
    pytest.importorskip("pyarrow")
    from fraud.demo_data import write_demo

    paths = write_demo(tmp_path, n=1000, now=NOW)
    rings = dd.load_rings(paths["rings_path"])
    assert not rings.empty
    conn = dd.connect(paths["cases_db"])
    assert not dd.ring_summary(conn, rings).empty
    conn.close()
    stats = dd.load_consumer_stats(paths["stats_path"])
    assert stats["demo"] is True and stats["processed"] == 1000
    assert json.loads(paths["model_path"].read_text())["threshold"] > 0


def test_importing_fraud_demo_stays_light():
    code = (
        "import sys, fraud.dashboard_data, fraud.demo_data; "
        "heavy = {'xgboost', 'feast', 'torch', 'confluent_kafka', 'redis'} & set(sys.modules); "
        "print(sorted(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert out.stdout.strip() == "[]"


def test_streamlit_app_demo_mode(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("pyarrow")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DASHBOARD_DEMO", "1")  # restored afterwards (the entrypoint sets it)
    monkeypatch.setenv("DASHBOARD_DEMO_DIR", str(tmp_path))
    at = AppTest.from_file("services/dashboard/streamlit_app.py", default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("Demo mode" in w.value for w in at.warning)
    labels = [m.label for m in at.metric]
    assert at.metric[labels.index("Transactions scored")].value == "4,000"
    confirm = [b for b in at.button if b.label == "Confirm fraud"]
    open_before = dd.status_counts(dd.connect(tmp_path / "cases.db"))["open"]
    confirm[0].click().run()
    assert not at.exception, at.exception
    assert dd.status_counts(dd.connect(tmp_path / "cases.db"))["open"] == open_before - 1
