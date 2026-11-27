"""Drift monitor: case store window -> drift + live precision -> retraining decision.

Every `interval_s` seconds it reads the most recent `window_rows` scored transactions from the
case store (SQLite, configs/serving.yaml `cases_db`), compares them with the reference
(configs/drift.yaml `monitor`), writes `monitor_status.json` and logs the decision. When a
decision says retrain and `auto_retrain` is on, it calls the trigger (dry run unless
`execute_retrain`). Run: python -m services.monitor.main [--once]

Reference: `reference_path` (drift_reference.json from the Kaggle drift job) if it exists;
otherwise the input columns come from history.parquet (train + val) and the score reference is
the first `baseline_rows` live scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd

from fraud.config import load_config
from fraud.drift import build_reference, hour_of, monitor_window
from services.common import serving_config
from services.monitor.trigger import maybe_retrain

CASE_COLUMNS = "event_time, amount, product, score, p_xgb, alert, true_label, feedback_label"


def monitor_config(path: str | None = None) -> dict[str, Any]:
    """configs/drift.yaml; keys of the `monitor` section overridable by MONITOR_<KEY> env vars."""
    cfg = load_config(path or os.environ.get("DRIFT_CONFIG", "configs/drift.yaml"))
    for key, value in list(cfg["monitor"].items()):
        env = os.environ.get(f"MONITOR_{key.upper()}")
        if env is None or isinstance(value, dict):
            continue
        if isinstance(value, bool):
            cfg["monitor"][key] = env.lower() in ("1", "true", "yes")
        elif isinstance(value, (int, float)):
            cfg["monitor"][key] = type(value)(env)
        else:
            cfg["monitor"][key] = env
    return cfg


def load_reference(cfg: dict[str, Any], serving: dict[str, Any]) -> tuple[dict, float | None]:
    """(reference columns, expected alert rate) from the Kaggle file or from history.parquet."""
    mon, psi_cfg = cfg["monitor"], cfg["psi"]
    path = Path(mon["reference_path"])
    if path.exists():
        data = json.loads(path.read_text())
        print(f"reference: {path} ({', '.join(data['columns'])})", flush=True)
        return data["columns"], data.get("expected_alert_rate")
    inputs = [c for c in mon["live_columns"] if c != mon["score_column"]]
    hist = pd.read_parquet(serving["history_path"])
    if "hour" in inputs and "hour" not in hist:
        hist["hour"] = hour_of(hist["TransactionDT"])
    numeric = [c for c in inputs if c in hist and pd.api.types.is_numeric_dtype(hist[c])]
    categorical = [c for c in inputs if c in hist and c not in numeric]
    ref = build_reference(hist, numeric, categorical, psi_cfg["bins"], psi_cfg["max_categories"])
    print(f"reference: {len(hist):,} history rows; score baseline from live data", flush=True)
    return ref, None


def read_cases(db: str, limit: int, newest: bool = True) -> pd.DataFrame:
    order = "DESC" if newest else "ASC"
    con = sqlite3.connect(db, timeout=30)
    try:
        return pd.read_sql_query(
            f"SELECT {CASE_COLUMNS} FROM scored ORDER BY scored_at {order}, transaction_id {order} "
            "LIMIT ?",
            con,
            params=(int(limit),),
        )
    finally:
        con.close()


def count_cases(db: str) -> int:
    con = sqlite3.connect(db, timeout=30)
    try:
        return int(con.execute("SELECT COUNT(*) FROM scored").fetchone()[0])
    finally:
        con.close()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)  # readers never see a half-written file


class Monitor:
    def __init__(self, cfg: dict[str, Any], serving: dict[str, Any]):
        self.cfg, self.serving = cfg, serving
        self.mon = cfg["monitor"]
        self.reference, self.expected_alert_rate = load_reference(cfg, serving)
        self.history: list[dict[str, Any]] = []

    def _ensure_score_baseline(self, db: str, total: int) -> bool:
        score_col = self.mon["score_column"]
        if score_col in self.reference:
            return True
        need = int(self.mon["baseline_rows"])
        if total < need + int(self.cfg["retrain"]["min_window"]):
            return False
        base = read_cases(db, need, newest=False)
        ref = build_reference(base, [score_col], [], self.cfg["psi"]["bins"])
        self.reference[score_col] = ref[score_col]
        self.expected_alert_rate = float(base["alert"].mean())
        print(
            f"score baseline: first {need} live scores, alert rate {self.expected_alert_rate:.4f}",
            flush=True,
        )
        return True

    def check(self) -> dict[str, Any] | None:
        db = self.serving["cases_db"]
        if not Path(db).exists():
            print(f"waiting for the case store {db}", flush=True)
            return None
        total = count_cases(db)
        baseline_ready = self._ensure_score_baseline(db, total)
        cases = read_cases(db, int(self.mon["window_rows"]))
        result = monitor_window(cases, self.reference, self.cfg, self.expected_alert_rate)
        result["checked_at"] = time.time()
        result["cases_total"] = total
        result["score_baseline_ready"] = baseline_ready
        decision = result["decision"]
        if decision["retrain"] and self.mon["auto_retrain"]:
            result["trigger"] = maybe_retrain(
                decision,
                self.cfg["trigger"],
                self.cfg["retrain"]["cooldown_hours"],
                execute=bool(self.mon["execute_retrain"]),
            )
        perf = result["performance"]
        self.history.append(
            {
                "checked_at": result["checked_at"],
                "rows": result["window"]["rows"],
                "drift_share": result["drift"]["drift_share"],
                "drifted": result["drift"]["drifted"],
                "score_psi": result["score_psi"],
                "alert_rate": perf["alert_rate"],
                "precision": perf["precision"],
                "retrain": decision["retrain"],
                "reasons": decision["reasons"],
            }
        )
        self.history = self.history[-int(self.mon["history_limit"]) :]
        write_json(self.mon["status_path"], {**result, "history": self.history})
        psi = result["drift"]["psi"]
        print(
            f"window {result['window']['rows']} of {total} cases | PSI "
            + ", ".join(f"{k} {v:.3f}" for k, v in psi.items())
            + f" | score PSI {result['score_psi']} | precision {perf['precision']} "
            f"({perf['reviewed']} reviewed) | retrain {decision['retrain']} "
            f"{decision['reasons'] or decision.get('note', '')}",
            flush=True,
        )
        return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="one check, then exit")
    args = ap.parse_args(argv)
    cfg = monitor_config()
    monitor = Monitor(cfg, serving_config())
    while True:
        try:
            monitor.check()
        except sqlite3.Error as exc:  # e.g. the consumer is creating the schema
            print(f"case store not readable yet: {exc}", flush=True)
        if args.once:
            break
        time.sleep(float(cfg["monitor"]["interval_s"]))


if __name__ == "__main__":
    main()
