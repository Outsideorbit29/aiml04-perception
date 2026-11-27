"""Case store for scored transactions and analyst decisions (SQLite; one writer, many readers).

Every scored transaction is stored; alerts start as status 'open'. Analysts set the status to
'confirmed' (fraud) or 'dismissed' (legit) in the dashboard, which also records the feedback
label used by F6 to measure live precision and trigger retraining.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS scored (
    transaction_id INTEGER PRIMARY KEY,
    event_time     REAL NOT NULL,      -- TransactionDT seconds
    scored_at      REAL NOT NULL,      -- unix time
    client_key     TEXT,
    card_key       TEXT,
    device_key     TEXT,
    amount         REAL,
    product        TEXT,
    score          REAL NOT NULL,
    p_xgb          REAL,
    gnn_risk       REAL,
    alert          INTEGER NOT NULL,
    latency_ms     REAL,
    true_label     INTEGER,           -- from the replay; hidden from analysts, used for eval
    status         TEXT,              -- NULL (no alert) | open | confirmed | dismissed
    feedback_label INTEGER,           -- analyst decision: 1 fraud, 0 legit
    reviewed_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_scored_alert ON scored(alert, status);
CREATE INDEX IF NOT EXISTS idx_scored_client ON scored(client_key);
"""

STATUSES = ("open", "confirmed", "dismissed")


class CaseStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")  # dashboard reads while the consumer writes
        self.conn.executescript(SCHEMA)
        self.conn.row_factory = sqlite3.Row

    def insert(self, rows: Sequence[Mapping[str, Any]]) -> None:
        now = time.time()
        self.conn.executemany(
            """INSERT OR REPLACE INTO scored (transaction_id, event_time, scored_at, client_key,
               card_key, device_key, amount, product, score, p_xgb, gnn_risk, alert, latency_ms,
               true_label, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    int(r["transaction_id"]),
                    float(r["event_time"]),
                    now,
                    r.get("client_key"),
                    r.get("card_key"),
                    r.get("device_key"),
                    r.get("amount"),
                    r.get("product"),
                    float(r["score"]),
                    r.get("p_xgb"),
                    r.get("gnn_risk"),
                    int(bool(r["alert"])),
                    r.get("latency_ms"),
                    r.get("true_label"),
                    "open" if r["alert"] else None,
                )
                for r in rows
            ],
        )
        self.conn.commit()

    def set_status(self, transaction_id: int, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        label = {"confirmed": 1, "dismissed": 0}.get(status)
        cur = self.conn.execute(
            "UPDATE scored SET status=?, feedback_label=?, reviewed_at=? "
            "WHERE transaction_id=? AND alert=1",
            (status, label, time.time() if label is not None else None, int(transaction_id)),
        )
        if cur.rowcount == 0:
            raise KeyError(f"no alert with transaction_id {transaction_id}")
        self.conn.commit()

    def alerts(self, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT * FROM scored WHERE alert=1"
        args: list[Any] = []
        if status:
            sql += " AND status=?"
            args.append(status)
        sql += " ORDER BY score DESC, event_time DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def stats(self) -> dict[str, Any]:
        q = self.conn.execute(
            """SELECT COUNT(*) n, SUM(alert) alerts, SUM(true_label) frauds,
                      SUM(alert * true_label) caught, AVG(latency_ms) mean_latency,
                      SUM(status='confirmed') confirmed, SUM(status='dismissed') dismissed,
                      SUM(status='open') open
               FROM scored"""
        ).fetchone()
        return {k: q[k] for k in q.keys()}

    def close(self) -> None:
        self.conn.close()
