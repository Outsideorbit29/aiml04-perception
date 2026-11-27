import json

import numpy as np
import pandas as pd
import pytest

from fraud.cases import CaseStore
from fraud.config import load_config
from fraud.drift import build_reference, hour_of
from services.monitor.main import Monitor, monitor_config
from services.monitor.trigger import maybe_retrain, retrain_command

CFG = load_config("configs/drift.yaml")


def _trigger_cfg(tmp_path):
    return {**CFG["trigger"], "kaggle_cli": "kaggle", "state_path": str(tmp_path / "state.json")}


def test_retrain_command(monkeypatch):
    monkeypatch.delenv("KAGGLE", raising=False)
    assert retrain_command(CFG["trigger"]) == ["kaggle", "kernels", "push", "-p", "kaggle/train"]
    monkeypatch.setenv("KAGGLE", "/opt/kaggle")
    assert retrain_command(CFG["trigger"])[0] == "/opt/kaggle"


def test_trigger_dry_run_execute_and_cooldown(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE", raising=False)
    tcfg = _trigger_cfg(tmp_path)
    yes = {"retrain": True, "reasons": ["drift share 0.50 >= 0.3"]}
    assert maybe_retrain({"retrain": False}, tcfg, 24)["action"] == "none"
    calls = []
    dry = maybe_retrain(yes, tcfg, 24, execute=False, runner=calls.append)
    assert dry["action"] == "dry_run" and dry["command"] == "kaggle kernels push -p kaggle/train"
    assert calls == [] and not (tmp_path / "state.json").exists()

    def ok(cmd):
        calls.append(cmd)
        return 0

    run = maybe_retrain(yes, tcfg, 24, execute=True, now=1000.0, runner=ok)
    assert run["action"] == "launched" and len(calls) == 1
    assert json.loads((tmp_path / "state.json").read_text())["last_launch"] == 1000.0
    again = maybe_retrain(yes, tcfg, 24, execute=True, now=1000.0 + 3600, runner=ok)
    assert again["action"] == "cooldown" and len(calls) == 1
    later = maybe_retrain(yes, tcfg, 24, execute=True, now=1000.0 + 25 * 3600, runner=ok)
    assert later["action"] == "launched" and len(calls) == 2


def test_trigger_failures_are_reported(tmp_path):
    tcfg = _trigger_cfg(tmp_path)
    yes = {"retrain": True, "reasons": ["x"]}

    def missing(cmd):
        raise FileNotFoundError("kaggle")

    assert maybe_retrain(yes, tcfg, 24, execute=True, runner=missing)["action"] == "failed"
    bad = maybe_retrain(yes, tcfg, 24, execute=True, runner=lambda c: 1)
    assert bad["action"] == "failed" and bad["returncode"] == 1
    assert not (tmp_path / "state.json").exists()


def test_drift_retrain_cli_reads_summary_and_status(tmp_path, monkeypatch, capsys):
    from scripts import drift_retrain

    monkeypatch.chdir(tmp_path.parent)  # state_path is relative; keep it out of the repo
    cfg_path = tmp_path / "drift.yaml"
    cfg = {**CFG, "trigger": _trigger_cfg(tmp_path)}
    cfg_path.write_text(json.dumps(cfg))  # JSON is valid YAML
    summary = {
        "windows": [
            {"window": "all", "decision": {"retrain": True, "reasons": ["a"]}},
            {"window": "w01", "decision": {"retrain": False, "reasons": []}},
            {"window": "w02", "decision": {"retrain": True, "reasons": ["drift"]}},
        ]
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    out = drift_retrain.main(
        ["--config", str(cfg_path), "--summary", str(tmp_path / "summary.json")]
    )
    assert out["action"] == "dry_run"
    out = drift_retrain.main(
        ["--config", str(cfg_path), "--summary", str(tmp_path / "summary.json"), "--window", "w01"]
    )
    assert out["action"] == "none"
    (tmp_path / "status.json").write_text(
        json.dumps({"decision": summary["windows"][2]["decision"]})
    )
    out = drift_retrain.main(["--config", str(cfg_path), "--status", str(tmp_path / "status.json")])
    assert out["action"] == "dry_run"
    assert "[dry-run] would retrain" in capsys.readouterr().out


def _rows(n, start_id, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    alert = rng.random(n) < 0.04
    return [
        {
            "transaction_id": start_id + i,
            "event_time": float(start_id + i) * 30,
            "amount": float(rng.lognormal(4, 1.2)) * scale,
            "product": str(rng.choice(["W", "C", "H"], p=[0.75, 0.15, 0.1])),
            "score": float(rng.beta(1, 20)),
            "p_xgb": float(rng.beta(1, 20)),
            "alert": bool(alert[i]),
            "true_label": int(rng.random() < (0.6 if alert[i] else 0.02)),
        }
        for i in range(n)
    ]


def test_monitor_end_to_end_with_case_store(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    hist = pd.DataFrame(
        {
            "TransactionAmt": rng.lognormal(4, 1.2, 20_000),
            "ProductCD": rng.choice(["W", "C", "H"], 20_000, p=[0.75, 0.15, 0.1]),
            "TransactionDT": rng.integers(0, 90 * 86_400, 20_000),
        }
    )
    hist["hour"] = hour_of(hist["TransactionDT"])
    ref = build_reference(hist, ["TransactionAmt", "hour"], ["ProductCD"])
    (tmp_path / "ref.json").write_text(json.dumps({"columns": ref, "expected_alert_rate": None}))
    db = tmp_path / "cases.db"
    store = CaseStore(db)
    store.insert(_rows(2500, 0, seed=1))  # baseline + clean traffic

    monkeypatch.setenv("MONITOR_REFERENCE_PATH", str(tmp_path / "ref.json"))
    monkeypatch.setenv("MONITOR_STATUS_PATH", str(tmp_path / "status.json"))
    monkeypatch.setenv("MONITOR_WINDOW_ROWS", "1000")
    monkeypatch.setenv("MONITOR_BASELINE_ROWS", "1000")
    monkeypatch.setenv("MONITOR_FEEDBACK_SOURCE", "true_label")
    cfg = monitor_config()
    assert (
        cfg["monitor"]["window_rows"] == 1000 and cfg["monitor"]["feedback_source"] == "true_label"
    )
    cfg["trigger"] = _trigger_cfg(tmp_path)
    monitor = Monitor(cfg, {"cases_db": str(db)})

    first = monitor.check()
    assert first["score_baseline_ready"] and "p_xgb" in monitor.reference
    assert monitor.expected_alert_rate is not None
    assert not first["decision"]["retrain"], first["decision"]

    store.insert(_rows(1000, 10_000, seed=2, scale=4.0))  # drifted amounts arrive
    second = monitor.check()
    assert "TransactionAmt" in second["drift"]["drifted"]
    assert second["decision"]["retrain"] and second["trigger"]["action"] == "dry_run"
    status = json.loads((tmp_path / "status.json").read_text())
    assert [h["retrain"] for h in status["history"]] == [False, True]
    store.close()


def test_monitor_waits_for_missing_db(tmp_path, monkeypatch):
    monkeypatch.setenv("MONITOR_REFERENCE_PATH", str(tmp_path / "missing.json"))
    cfg = monitor_config()
    pytest.importorskip("pyarrow")
    hist = pd.DataFrame(
        {"TransactionAmt": [1.0, 2.0], "ProductCD": ["W", "C"], "TransactionDT": [0, 7200]}
    )
    hist.to_parquet(tmp_path / "history.parquet")
    monitor = Monitor(
        cfg,
        {"cases_db": str(tmp_path / "none.db"), "history_path": str(tmp_path / "history.parquet")},
    )
    assert set(monitor.reference) == {"TransactionAmt", "ProductCD"}
    assert monitor.check() is None
