"""Read/query helpers for the F7 analyst dashboard (SQLite case store + fraud rings).

Everything here is plain pandas/sqlite so it can be unit-tested without Streamlit. The schema is
the `scored` table of fraud.cases; analyst decisions are written through CaseStore.set_status.
`true_label` is only used by evaluation() and the *_eval columns, which the dashboard shows in a
separate "evaluation" section that analysts would not see in production.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud.cases import STATUSES, CaseStore
from fraud.config import load_config

QUEUE_COLUMNS = [
    "transaction_id",
    "event_time",
    "score",
    "p_xgb",
    "gnn_risk",
    "amount",
    "product",
    "client_key",
    "card_key",
    "device_key",
    "status",
    "latency_ms",
]


def dashboard_config(
    serving_path: str | Path = "configs/serving.yaml",
    dashboard_path: str | Path = "services/dashboard/dashboard.yaml",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """serving.yaml + dashboard.yaml (serving keys win), each key overridable by an upper-case
    environment variable (CASES_DB, SCORER_URL, RINGS_PATH, ...), cast to the default's type."""
    env = os.environ if env is None else env
    cfg = {**load_config(dashboard_path), **load_config(serving_path)}
    for key, value in list(cfg.items()):
        raw = env.get(key.upper())
        if raw is None:
            continue
        if isinstance(value, bool):
            cfg[key] = raw.lower() in ("1", "true", "yes")
        elif isinstance(value, (int, float)):
            cfg[key] = type(value)(raw)
        else:
            cfg[key] = raw
    return cfg


def connect(path: str | Path) -> sqlite3.Connection | None:
    """Open the case DB for reading; None if the consumer has not created it yet."""
    if not Path(path).exists():
        return None
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _has_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scored'")
    return row.fetchone() is not None


def _pct(values: pd.Series, q: float) -> float | None:
    v = values.dropna().to_numpy(dtype="float64")
    return float(np.percentile(v, q)) if len(v) else None


def overview(
    conn: sqlite3.Connection,
    alert_budget: float = 0.01,
    throughput_window_s: float = 60.0,
    latency_last_n: int = 5000,
    expected_alert_rate: float | None = None,
) -> dict[str, Any]:
    """Headline numbers: volume, alert rate vs budget, latency percentiles, recent throughput.

    `alert_budget` is a false-positive rate (share of legit transactions flagged), so the total
    alert rate is compared with `expected_alert_rate` (budget on legit traffic plus the frauds
    caught) when it is given; otherwise with the budget itself.
    """
    target = expected_alert_rate if expected_alert_rate is not None else alert_budget
    empty = {
        "scored": 0,
        "alerts": 0,
        "alert_rate": None,
        "alert_budget": alert_budget,
        "expected_alert_rate": target,
        "over_budget": False,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "throughput_tps": 0.0,
        "last_scored_at": None,
    }
    if conn is None or not _has_table(conn):
        return empty
    row = conn.execute(
        "SELECT COUNT(*) n, SUM(alert) a, MAX(scored_at) last FROM scored"
    ).fetchone()
    n, alerts, last = int(row["n"]), int(row["a"] or 0), row["last"]
    if n == 0:
        return empty
    lat = pd.read_sql_query(
        "SELECT latency_ms FROM scored ORDER BY scored_at DESC LIMIT ?",
        conn,
        params=(latency_last_n,),
    )["latency_ms"]
    recent = conn.execute(
        "SELECT COUNT(*) FROM scored WHERE scored_at > ?", (last - throughput_window_s,)
    ).fetchone()[0]
    rate = alerts / n
    return {
        "scored": n,
        "alerts": alerts,
        "alert_rate": rate,
        "alert_budget": alert_budget,
        "expected_alert_rate": target,
        "over_budget": rate > target,
        "latency_p50_ms": _pct(lat, 50),
        "latency_p95_ms": _pct(lat, 95),
        "throughput_tps": recent / throughput_window_s,
        "last_scored_at": float(last),
    }


def timeline(
    conn: sqlite3.Connection, bucket_s: float = 10.0, last_s: float = 600.0
) -> pd.DataFrame:
    """Transactions and alerts per bucket of wall-clock scoring time (for a live chart)."""
    cols = ["t", "transactions", "alerts", "tps"]
    if conn is None or not _has_table(conn):
        return pd.DataFrame(columns=cols)
    df = pd.read_sql_query(
        "SELECT scored_at, alert FROM scored WHERE scored_at > "
        "(SELECT MAX(scored_at) FROM scored) - ?",
        conn,
        params=(last_s,),
    )
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["t"] = pd.to_datetime((df["scored_at"] // bucket_s) * bucket_s, unit="s")
    out = df.groupby("t").agg(transactions=("alert", "size"), alerts=("alert", "sum"))
    out["tps"] = out["transactions"] / bucket_s
    return out.reset_index()[cols]


def alert_queue(
    conn: sqlite3.Connection,
    status: str | None = "open",
    limit: int = 200,
    min_score: float = 0.0,
    rings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Alerts sorted by score (highest first) with the score breakdown and, if given, ring ids."""
    if conn is None or not _has_table(conn):
        return pd.DataFrame(columns=QUEUE_COLUMNS)
    if status is not None and status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES} or None")
    sql = f"SELECT {', '.join(QUEUE_COLUMNS)} FROM scored WHERE alert=1 AND score >= ?"
    args: list[Any] = [min_score]
    if status:
        sql += " AND status=?"
        args.append(status)
    sql += " ORDER BY score DESC, event_time DESC LIMIT ?"
    args.append(limit)
    df = pd.read_sql_query(sql, conn, params=args)
    # How much the graph risk moved the score: blended score minus the XGBoost-only score.
    df["gnn_shift"] = df["score"] - df["p_xgb"]
    if rings is not None and not rings.empty:
        ring_cols = rings[["client_key", "ring_id", "ring_size"]].drop_duplicates("client_key")
        df = df.merge(ring_cols, on="client_key", how="left")
    return df


def status_counts(conn: sqlite3.Connection) -> dict[str, Any]:
    """Open / confirmed / dismissed counts and live precision of the reviewed alerts."""
    out: dict[str, Any] = {s: 0 for s in STATUSES}
    out.update(reviewed=0, reviewed_precision=None)
    if conn is None or not _has_table(conn):
        return out
    for row in conn.execute(
        "SELECT status, COUNT(*) n FROM scored WHERE alert=1 GROUP BY status"
    ).fetchall():
        if row["status"] in out:
            out[row["status"]] = int(row["n"])
    reviewed = out["confirmed"] + out["dismissed"]
    out["reviewed"] = reviewed
    out["reviewed_precision"] = out["confirmed"] / reviewed if reviewed else None
    return out


def evaluation(conn: sqlite3.Connection) -> dict[str, Any]:
    """Metrics against the replay's true labels (hidden from analysts; for evaluation only)."""
    keys = ("labelled", "frauds", "precision", "recall", "fpr", "analyst_agreement")
    if conn is None or not _has_table(conn):
        return dict.fromkeys(keys)
    r = conn.execute(
        """SELECT COUNT(*) n,
                  SUM(true_label) frauds,
                  SUM(alert) alerts,
                  SUM(alert * true_label) tp,
                  SUM(alert * (1 - true_label)) fp,
                  SUM(1 - true_label) legit,
                  SUM(feedback_label IS NOT NULL) reviewed,
                  SUM(feedback_label = true_label) agree
           FROM scored WHERE true_label IS NOT NULL"""
    ).fetchone()
    n = int(r["n"])
    if n == 0:
        return dict.fromkeys(keys)
    frauds, alerts, tp = int(r["frauds"] or 0), int(r["alerts"] or 0), int(r["tp"] or 0)
    legit, reviewed = int(r["legit"] or 0), int(r["reviewed"] or 0)
    return {
        "labelled": n,
        "frauds": frauds,
        "precision": tp / alerts if alerts else None,
        "recall": tp / frauds if frauds else None,
        "fpr": int(r["fp"] or 0) / legit if legit else None,
        "analyst_agreement": int(r["agree"] or 0) / reviewed if reviewed else None,
    }


def load_rings(path: str | Path) -> pd.DataFrame:
    """rings.parquet (client_key, ring_id, ring_size, ring_devices); empty frame if missing."""
    cols = ["client_key", "ring_id", "ring_size", "ring_devices"]
    if not Path(path).exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_parquet(path, columns=cols)
    df["client_key"] = df["client_key"].astype(str)
    return df


def _scored_in_rings(conn: sqlite3.Connection, rings: pd.DataFrame) -> pd.DataFrame:
    """Scored transactions of clients that belong to a ring, with the ring columns attached."""
    tx = pd.read_sql_query(
        "SELECT transaction_id, client_key, card_key, device_key, score, alert, status, "
        "true_label FROM scored",
        conn,
    )
    return tx.merge(rings, on="client_key", how="inner")


def ring_summary(
    conn: sqlite3.Connection, rings: pd.DataFrame, only_with_alerts: bool = True
) -> pd.DataFrame:
    """One row per ring seen in the live stream: size, activity, alerts and decisions.

    `fraud_share_eval` uses true labels and belongs in the evaluation view only.
    """
    cols = [
        "ring_id",
        "ring_size",
        "ring_devices",
        "active_clients",
        "transactions",
        "alerts",
        "alert_share",
        "confirmed",
        "dismissed",
        "max_score",
        "fraud_share_eval",
    ]
    if conn is None or not _has_table(conn) or rings.empty:
        return pd.DataFrame(columns=cols)
    tx = _scored_in_rings(conn, rings)
    if tx.empty:
        return pd.DataFrame(columns=cols)
    tx["confirmed"] = (tx["status"] == "confirmed").astype(int)
    tx["dismissed"] = (tx["status"] == "dismissed").astype(int)
    g = tx.groupby("ring_id").agg(
        ring_size=("ring_size", "first"),
        ring_devices=("ring_devices", "first"),
        active_clients=("client_key", "nunique"),
        transactions=("transaction_id", "size"),
        alerts=("alert", "sum"),
        confirmed=("confirmed", "sum"),
        dismissed=("dismissed", "sum"),
        max_score=("score", "max"),
        fraud_share_eval=("true_label", "mean"),
    )
    g["alert_share"] = g["alerts"] / g["transactions"]
    g = g.reset_index()[cols]
    if only_with_alerts:
        g = g[g["alerts"] > 0]
    return g.sort_values(["alerts", "max_score"], ascending=False).reset_index(drop=True)


def ring_graph_dot(
    conn: sqlite3.Connection, rings: pd.DataFrame, ring_id: int, max_clients: int = 40
) -> str:
    """Graphviz DOT for one ring: clients linked to the devices and cards they used live.

    Alerting clients are drawn red and listed first; large rings are cut to max_clients so the
    chart stays readable (one coarse-fingerprint component has thousands of clients).
    """
    members = rings[rings["ring_id"] == ring_id]
    tx = _scored_in_rings(conn, members) if conn is not None and _has_table(conn) else None
    lines = [
        "graph ring {",
        '  graph [overlap=false, splines=true, bgcolor="transparent"];',
        '  node [style=filled, fontsize=9, fontname="Helvetica"];',
    ]
    if tx is None or tx.empty:
        lines.append(f'  "no live transactions for ring {ring_id}" [shape=plaintext];')
        return "\n".join(lines + ["}"])
    n_active = tx["client_key"].nunique()
    per_client = (
        tx.groupby("client_key")
        .agg(alerts=("alert", "sum"), max_score=("score", "max"))
        .sort_values(["alerts", "max_score"], ascending=False)
        .head(max_clients)
    )
    tx = tx[tx["client_key"].isin(per_client.index)]

    def q(s: Any) -> str:
        return '"' + str(s).replace('"', "'") + '"'

    for client, r in per_client.iterrows():
        color = "#e4572e" if r["alerts"] > 0 else "#9db4c0"
        label = f"{str(client)[:14]}\\n{int(r['alerts'])} alerts"
        lines.append(
            f"  {q('c:' + str(client))} [label={q(label)}, shape=ellipse, fillcolor={q(color)}];"
        )
    edges: set[tuple[str, str]] = set()
    for kind, col, shape, color in (
        ("d", "device_key", "box", "#f3d34a"),
        ("k", "card_key", "diamond", "#cfd8dc"),
    ):
        for key in tx[col].dropna().unique():
            node = f"{kind}:{key}"
            lines.append(
                f"  {q(node)} [label={q(str(key)[:18])}, shape={shape}, fillcolor={q(color)}];"
            )
        for client, key in (
            tx[["client_key", col]].dropna().drop_duplicates().itertuples(index=False)
        ):
            edges.add((f"c:{client}", f"{kind}:{key}"))
    lines += [f"  {q(a)} -- {q(b)};" for a, b in sorted(edges)]
    if len(per_client) < n_active:
        lines.append(f'  "showing top {len(per_client)} of {n_active} clients" [shape=plaintext];')
    return "\n".join(lines + ["}"])


def load_consumer_stats(path: str | Path) -> dict[str, Any] | None:
    """The consumer's latency/throughput snapshot (stats_path), or None if absent/half-written."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def set_case_status(db_path: str | Path, transaction_id: int, status: str) -> None:
    """Record an analyst decision (confirmed = fraud, dismissed = legit) via CaseStore."""
    store = CaseStore(db_path)
    try:
        store.set_status(int(transaction_id), status)
    finally:
        store.close()
