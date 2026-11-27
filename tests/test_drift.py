import json

import numpy as np
import pandas as pd
import pytest

from fraud.config import load_config
from fraud.drift import (
    MISSING,
    OTHER,
    apply_drift,
    build_reference,
    categorical_props,
    categorical_reference,
    column_psi,
    decide_retrain,
    drift_table,
    hour_of,
    live_performance,
    monitor_window,
    numeric_props,
    numeric_reference,
    parse_drift_spec,
    psi,
    time_windows,
)

CFG = load_config("configs/drift.yaml")


def _frame(n=5000, seed=0, scale=1.0, product_p=(0.75, 0.15, 0.1)):
    rng = np.random.default_rng(seed)
    amt = rng.lognormal(4, 1.2, n) * scale
    amt[rng.random(n) < 0.05] = np.nan
    return pd.DataFrame(
        {
            "TransactionAmt": amt,
            "ProductCD": rng.choice(["W", "C", "H"], n, p=product_p),
            "C1": rng.poisson(1.5, n).astype(float),
        }
    )


def test_psi_identical_is_zero_and_shift_is_large():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)
    ref = numeric_reference(_frame(seed=0)["TransactionAmt"], bins=10)
    assert column_psi(_frame(seed=1)["TransactionAmt"], ref) < 0.02
    assert column_psi(_frame(seed=1, scale=3)["TransactionAmt"], ref) > 0.25


def test_numeric_reference_bins_and_missing():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, None, 5.0, 6.0, 7.0, 8.0])
    ref = numeric_reference(x, bins=4)
    assert len(ref["props"]) == len(ref["edges"]) + 2  # value bins + missing
    assert sum(ref["props"]) == pytest.approx(1.0)
    assert ref["props"][-1] == pytest.approx(0.2)
    # Values beyond the reference range land in the outer bins.
    props = numeric_props([-100.0, 1000.0], ref)
    assert props[0] == pytest.approx(0.5) and props[-2] == pytest.approx(0.5)
    json.dumps(ref)  # plain JSON (the monitor loads it from a file)


def test_numeric_reference_discrete_and_empty():
    ref = numeric_reference([1.0] * 50 + [2.0] * 50, bins=10)
    assert ref["edges"] == [1.0, 2.0]  # edges sit on observed values
    assert column_psi([1.0] * 50 + [2.0] * 50, ref) == pytest.approx(0.0, abs=1e-9)
    empty = numeric_reference([np.nan, np.nan])
    assert empty["edges"] == [] and empty["props"] == [0.0, 1.0]


def test_categorical_reference_other_and_missing():
    ref = categorical_reference(["W"] * 6 + ["C"] * 3 + ["H"] + [None] * 2, max_categories=2)
    assert ref["categories"] == ["C", "W"]
    props = categorical_props(["W", "H", "Z", None], ref)
    labels = [*ref["categories"], OTHER, MISSING]
    assert dict(zip(labels, props, strict=True)) == {"C": 0, "W": 0.25, OTHER: 0.5, MISSING: 0.25}
    assert column_psi(["W"] * 6 + ["C"] * 3 + ["H"] + [None] * 2, ref) == pytest.approx(0.0)
    assert column_psi(["C"] * 10, ref) > 0.5


def test_drift_table_share_and_exclude():
    ref_df, cur = _frame(seed=0), _frame(seed=1, scale=3, product_p=(0.2, 0.7, 0.1))
    ref = build_reference(ref_df, ["TransactionAmt", "C1"], ["ProductCD"])
    table = drift_table(cur, ref, threshold=0.2)
    assert set(table["drifted"]) == {"TransactionAmt", "ProductCD"}
    assert table["drift_share"] == pytest.approx(2 / 3)
    assert list(table["psi"])[-1] == "C1"  # sorted by PSI, largest first
    excl = drift_table(cur, ref, threshold=0.2, exclude=["TransactionAmt"])
    assert "TransactionAmt" in excl["psi"] and excl["drifted"] == ["ProductCD"]
    assert excl["drift_share"] == pytest.approx(0.5)
    assert drift_table(_frame(seed=2), ref)["drift_share"] == 0.0


def test_time_windows_weekly_and_merge():
    day = 86_400
    t = np.r_[np.arange(0, 14 * day, 600), np.arange(14 * day, 14 * day + 3600, 600)]
    w = time_windows(t, window_days=7, min_rows=100)
    assert [x["day_from"] for x in w] == [0, 7]
    assert w[-1]["day_to"] == 14  # the 6-row third week was merged into the second
    assert sum(len(x["index"]) for x in w) == len(t)
    assert w[0]["name"].startswith("w01_d0-6")
    assert time_windows([]) == []


def test_live_performance_feedback_and_full_labels():
    alert = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    fb = [1, 0, np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    p = live_performance(alert, fb, alert_budget=0.01, expected_alert_rate=0.2)
    assert p["reviewed"] == 3 and p["confirmed"] == 2
    assert p["precision"] == pytest.approx(2 / 3)
    assert p["alert_rate"] == pytest.approx(0.4) and p["alert_rate_ratio"] == pytest.approx(2.0)
    est = 4 * (1 / 3) / (10 - 4 * 2 / 3)
    assert p["fpr"] == pytest.approx(est) and p["fpr_budget_ratio"] == pytest.approx(est / 0.01)
    assert "recall" not in p
    full = live_performance(alert, [1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    assert full["fpr"] == pytest.approx(1 / 6) and full["recall"] == pytest.approx(0.75)
    none = live_performance([0, 0], [np.nan, np.nan])
    assert none["precision"] is None and none["fpr"] is None


def test_decide_retrain_checks():
    rc = CFG["retrain"]
    small = decide_retrain({"n": 10, "drift_share": 1.0}, rc)
    assert not small["retrain"] and "too small" in small["note"]
    quiet = decide_retrain({"n": 5000, "drift_share": 0.1, "score_psi": 0.05}, rc)
    assert not quiet["retrain"] and quiet["checks"]["precision"]["skipped"]
    drift = decide_retrain({"n": 5000, "drift_share": 0.5}, rc)
    assert drift["retrain"] and drift["reasons"][0].startswith("drift share 0.50")
    # Precision only counts once enough alerts were reviewed.
    few = decide_retrain({"n": 5000, "precision": 0.1, "reviewed": 5}, rc)
    many = decide_retrain({"n": 5000, "precision": 0.1, "reviewed": 500}, rc)
    assert not few["retrain"] and many["retrain"]
    assert decide_retrain({"n": 5000, "pr_auc": 0.3}, rc)["retrain"]
    only = decide_retrain({"n": 5000, "pr_auc": 0.3}, {**rc, "checks": ["data_drift"]})
    assert not only["retrain"] and list(only["checks"]) == ["data_drift"]


def test_drift_spec_parse_and_apply():
    spec = parse_drift_spec("amount_scale=3, shift_after=2, ProductCD=C")
    assert spec == {
        "shift_after": 2,
        "amount_scale": 3.0,
        "amount_shift": 0.0,
        "set": {"ProductCD": "C"},
    }
    msg = {"TransactionAmt": 10.0, "ProductCD": "W"}
    assert apply_drift(msg, 2, spec) is msg
    out = apply_drift(msg, 3, spec)
    assert out == {"TransactionAmt": 30.0, "ProductCD": "C"} and msg["TransactionAmt"] == 10.0
    assert apply_drift({"TransactionAmt": None}, 9, spec)["TransactionAmt"] is None
    assert parse_drift_spec("") == parse_drift_spec(None)
    with pytest.raises(ValueError):
        parse_drift_spec("amount_scale")


def _cases(n=3000, seed=0, scale=1.0, fraud_alerts=0.6, reviewed=1.0):
    rng = np.random.default_rng(seed)
    f = _frame(n, seed, scale)
    alert = (rng.random(n) < 0.03).astype(int)
    true = np.where(
        alert == 1, (rng.random(n) < fraud_alerts).astype(int), (rng.random(n) < 0.02).astype(int)
    )
    fb = np.where((alert == 1) & (rng.random(n) < reviewed), true, np.nan)
    return pd.DataFrame(
        {
            "event_time": np.sort(rng.integers(0, 7 * 86_400, n)).astype(float),
            "amount": f["TransactionAmt"],
            "product": f["ProductCD"],
            "p_xgb": rng.beta(1, 20, n),
            "score": rng.beta(1, 20, n),
            "alert": alert,
            "true_label": true,
            "feedback_label": fb,
        }
    )


def _live_reference(seed=10):
    base = _cases(20_000, seed)
    frame = pd.DataFrame(
        {
            "TransactionAmt": base["amount"],
            "ProductCD": base["product"],
            "hour": hour_of(base["event_time"]),
            "p_xgb": base["p_xgb"],
        }
    )
    return build_reference(frame, ["TransactionAmt", "hour", "p_xgb"], ["ProductCD"])


def test_monitor_window_stable_vs_drifted():
    ref = _live_reference()
    stable = monitor_window(_cases(seed=1), ref, CFG, expected_alert_rate=0.03)
    assert stable["drift"]["drifted"] == [] and not stable["decision"]["retrain"]
    assert set(stable["drift"]["psi"]) == {"TransactionAmt", "ProductCD"}
    live = {**CFG["monitor"]["live_columns"], "hour": "hour"}
    with_hour = monitor_window(
        _cases(seed=1), ref, {**CFG, "monitor": {**CFG["monitor"], "live_columns": live}}
    )
    assert with_hour["drift"]["psi"]["hour"] < 0.05  # a week of traffic covers every hour
    assert stable["score_psi"] < 0.05
    assert stable["performance"]["reviewed"] > 50
    drifted = monitor_window(_cases(seed=2, scale=3), ref, CFG, expected_alert_rate=0.03)
    assert drifted["drift"]["drifted"] == ["TransactionAmt"]
    assert drifted["decision"]["retrain"]
    bad = monitor_window(_cases(seed=3, fraud_alerts=0.1), ref, CFG, expected_alert_rate=0.03)
    assert any("precision" in r for r in bad["decision"]["reasons"])
    json.dumps(bad, default=str)


def test_monitor_window_true_label_source():
    ref = _live_reference()
    cfg = {**CFG, "monitor": {**CFG["monitor"], "feedback_source": "true_label"}}
    out = monitor_window(_cases(seed=4, reviewed=0.0), ref, cfg)
    assert out["performance"]["reviewed"] == out["performance"]["alerts"] > 0
    unreviewed = monitor_window(_cases(seed=4, reviewed=0.0), ref, CFG)
    assert unreviewed["performance"]["reviewed"] == 0
    assert unreviewed["decision"]["checks"]["precision"]["skipped"]
