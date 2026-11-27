import json

import pandas as pd
import pytest

from fraud import dashboard_data as dd
from fraud.cases import CaseStore


def _row(tid, client, score, alert, label, device="dev1", card="card1", p_xgb=None, risk=None):
    return {
        "transaction_id": tid,
        "event_time": 1000 + tid,
        "client_key": client,
        "card_key": card,
        "device_key": device,
        "amount": 10.0 * tid,
        "product": "W",
        "score": score,
        "p_xgb": score if p_xgb is None else p_xgb,
        "gnn_risk": risk,
        "alert": alert,
        "latency_ms": float(tid),
        "true_label": label,
    }


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "cases.db"
    store = CaseStore(path)
    store.insert(
        [
            _row(1, "a", 0.95, True, 1, p_xgb=0.80, risk=0.99),
            _row(2, "a", 0.90, True, 1, card="card2"),
            _row(3, "b", 0.70, True, 0, device="dev2"),
            _row(4, "c", 0.10, False, 0),
            _row(5, "d", 0.05, False, 1),
            *[_row(10 + i, "e", 0.01, False, 0, device=None) for i in range(5)],
        ]
    )
    store.close()
    return path


@pytest.fixture()
def rings():
    return pd.DataFrame(
        {
            "client_key": ["a", "b", "c", "z"],
            "ring_id": [7, 7, 7, 9],
            "ring_size": [3, 3, 3, 1],
            "ring_devices": [2, 2, 2, 1],
        }
    )


def test_missing_db_gives_empty_views(tmp_path, rings):
    conn = dd.connect(tmp_path / "nope.db")
    assert conn is None
    assert dd.overview(conn)["scored"] == 0
    assert dd.alert_queue(conn).empty
    assert dd.status_counts(conn)["reviewed_precision"] is None
    assert dd.evaluation(conn)["precision"] is None
    assert dd.ring_summary(conn, rings).empty
    assert dd.timeline(conn).empty


def test_overview(db):
    ov = dd.overview(dd.connect(db), alert_budget=0.01, throughput_window_s=60)
    assert ov["scored"] == 10 and ov["alerts"] == 3
    assert ov["alert_rate"] == pytest.approx(0.3) and ov["over_budget"]
    assert ov["latency_p50_ms"] == pytest.approx(7.5)  # median of 1..5, 10..14
    assert ov["throughput_tps"] == pytest.approx(10 / 60)  # all inserted just now


def test_alert_queue_sorted_with_breakdown_and_rings(db, rings):
    q = dd.alert_queue(dd.connect(db), "open", rings=rings)
    assert q["transaction_id"].tolist() == [1, 2, 3]
    assert q.loc[0, "gnn_shift"] == pytest.approx(0.15)
    assert q["ring_id"].tolist() == [7, 7, 7]
    assert dd.alert_queue(dd.connect(db), "open", min_score=0.8)["transaction_id"].tolist() == [
        1,
        2,
    ]
    with pytest.raises(ValueError):
        dd.alert_queue(dd.connect(db), "bogus")


def test_feedback_updates_status_and_precision(db):
    dd.set_case_status(db, 1, "confirmed")
    dd.set_case_status(db, 3, "dismissed")
    conn = dd.connect(db)
    sc = dd.status_counts(conn)
    assert (sc["open"], sc["confirmed"], sc["dismissed"]) == (1, 1, 1)
    assert sc["reviewed_precision"] == pytest.approx(0.5)
    assert dd.alert_queue(conn, "open")["transaction_id"].tolist() == [2]
    labels = dict(conn.execute("SELECT transaction_id, feedback_label FROM scored").fetchall())
    assert labels[1] == 1 and labels[3] == 0 and labels[2] is None
    with pytest.raises(KeyError):  # not an alert
        dd.set_case_status(db, 4, "confirmed")


def test_evaluation_uses_true_labels(db):
    dd.set_case_status(db, 1, "confirmed")  # agrees with label 1
    dd.set_case_status(db, 3, "confirmed")  # label 0: disagreement
    ev = dd.evaluation(dd.connect(db))
    assert ev["labelled"] == 10 and ev["frauds"] == 3
    assert ev["precision"] == pytest.approx(2 / 3)
    assert ev["recall"] == pytest.approx(2 / 3)
    assert ev["fpr"] == pytest.approx(1 / 7)
    assert ev["analyst_agreement"] == pytest.approx(0.5)


def test_ring_summary(db, rings):
    dd.set_case_status(db, 1, "confirmed")
    s = dd.ring_summary(dd.connect(db), rings)
    assert s["ring_id"].tolist() == [7]  # ring 9 has no live transactions
    r = s.iloc[0]
    assert (r["active_clients"], r["transactions"], r["alerts"], r["confirmed"]) == (3, 4, 3, 1)
    assert r["alert_share"] == pytest.approx(0.75)
    assert r["fraud_share_eval"] == pytest.approx(0.5)


def test_ring_graph_dot(db, rings):
    dot = dd.ring_graph_dot(dd.connect(db), rings, 7)
    assert dot.startswith("graph ring {") and dot.rstrip().endswith("}")
    assert '"c:a" -- "d:dev1"' in dot and '"c:a" -- "k:card2"' in dot
    assert '"c:b" -- "d:dev2"' in dot
    small = dd.ring_graph_dot(dd.connect(db), rings, 7, max_clients=1)
    assert "showing top 1 of 3 clients" in small and '"c:b"' not in small
    assert "no live transactions" in dd.ring_graph_dot(dd.connect(db), rings, 9)


def test_timeline(db):
    tl = dd.timeline(dd.connect(db), bucket_s=3600, last_s=600)
    assert tl["transactions"].sum() == 10 and tl["alerts"].sum() == 3


def test_dashboard_config_env_override(tmp_path):
    serving = tmp_path / "s.yaml"
    serving.write_text("cases_db: /app/data/cases.db\nscorer_url: http://scorer:8000\n")
    dash = tmp_path / "d.yaml"
    dash.write_text("rings_path: /x.parquet\nalert_budget: 0.01\nrefresh_s: 5\n")
    cfg = dd.dashboard_config(serving, dash, env={"CASES_DB": "c.db", "REFRESH_S": "2"})
    assert cfg["cases_db"] == "c.db" and cfg["refresh_s"] == 2
    assert cfg["alert_budget"] == 0.01 and cfg["scorer_url"] == "http://scorer:8000"


def test_repo_dashboard_config_has_required_keys():
    cfg = dd.dashboard_config(env={})
    for key in ("cases_db", "scorer_url", "stats_path", "rings_path", "alert_budget", "refresh_s"):
        assert key in cfg


def test_consumer_stats(tmp_path):
    p = tmp_path / "stats.json"
    assert dd.load_consumer_stats(p) is None
    p.write_text("{half")
    assert dd.load_consumer_stats(p) is None
    p.write_text(json.dumps({"processed": 5}))
    assert dd.load_consumer_stats(p) == {"processed": 5}


def test_load_rings_roundtrip(tmp_path, rings):
    pytest.importorskip("pyarrow")
    p = tmp_path / "rings.parquet"
    rings.assign(client_key=rings["client_key"]).to_parquet(p)
    assert dd.load_rings(p).equals(rings)
    assert dd.load_rings(tmp_path / "missing.parquet").empty


def test_overview_compares_total_alert_rate_with_expected(tmp_path):
    from fraud.cases import CaseStore

    store = CaseStore(tmp_path / "c.db")
    rows = [
        {
            "transaction_id": i,
            "event_time": float(i),
            "score": 0.9 if i < 3 else 0.1,
            "alert": i < 3,
            "true_label": int(i < 3),
        }
        for i in range(100)
    ]
    store.insert(rows)
    conn = dd.connect(tmp_path / "c.db")
    plain = dd.overview(conn, alert_budget=0.01)
    assert plain["alert_rate"] == 0.03 and plain["over_budget"]  # 3% > 1% budget
    ov = dd.overview(conn, alert_budget=0.01, expected_alert_rate=0.027)
    assert ov["expected_alert_rate"] == 0.027 and ov["over_budget"]
    assert not dd.overview(conn, alert_budget=0.01, expected_alert_rate=0.035)["over_budget"]
