"""Synthetic demo data for the public analyst dashboard (F7.3, Streamlit Community Cloud).

The public app cannot ship IEEE-CIS data (competition rules forbid redistribution) and has no
Kafka, Redis or scorer. This module builds, from a fixed seed, everything the dashboard reads:
a case store with the `fraud.cases` schema, a rings table and a consumer stats snapshot. All
values are synthetic: keys start with `demo-`, transaction ids start at 9,000,000 (outside the
IEEE-CIS id range) and the scores come from simple distributions, not from the trained model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud.cases import CaseStore
from fraud.synthetic import PRODUCTS

DEMO_ID_START = 9_000_000
WEIGHT_XGB = 0.75  # blend weight of p_xgb when a client has a graph risk (as in the live scorer)
RING_SIZES = (12, 8, 6, 5, 4)
FILES = {"cases_db": "cases.db", "rings_path": "rings.parquet", "stats_path": "consumer_stats.json"}
MODEL_FILE = "model.json"


def _pcts(a: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def make_demo(
    n: int = 4000,
    seed: int = 2026,
    now: float | None = None,
    window_s: float = 600.0,
    alert_rate: float = 0.03,
    review_share: float = 0.35,
) -> dict[str, Any]:
    """Return {"scored", "rings", "stats", "model"}: deterministic for a given seed and `now`.

    `scored` has the columns of the case store (plus status/feedback_label/reviewed_at),
    `rings` the columns of rings.parquet, `stats` a consumer_stats.json-like snapshot and
    `model` a /model-like summary of the synthetic scoring rule.
    """
    now = time.time() if now is None else float(now)
    rng = np.random.default_rng(seed)

    # Clients, their cards and devices; a few rings of clients sharing devices.
    n_clients = max(n // 4, sum(RING_SIZES) + 10)
    clients = np.array([f"demo-client-{i:04d}" for i in range(n_clients)])
    cards = np.array([f"demo-card-{i:04d}" for i in range(n_clients)])
    has_dev = rng.random(n_clients) < 0.6
    devices = np.array(
        [f"demo-dev-{i:04d}" if d else None for i, d in enumerate(has_dev)], dtype=object
    )
    ring_of = np.full(n_clients, -1)
    ring_rows = []
    members = rng.choice(n_clients, size=sum(RING_SIZES), replace=False)
    start = 0
    for r, size in enumerate(RING_SIZES):
        idx = members[start : start + size]
        start += size
        n_dev = int(rng.integers(1, 4))
        ring_devs = [f"demo-ring{r + 1}-dev{k + 1}" for k in range(n_dev)]
        ring_of[idx] = r + 1
        for i in idx:
            devices[i] = ring_devs[int(rng.integers(n_dev))]
            ring_rows.append((clients[i], r + 1, size, n_dev))
    rings = pd.DataFrame(ring_rows, columns=["client_key", "ring_id", "ring_size", "ring_devices"])
    in_ring = ring_of > 0

    # Graph risk: high for ring members, low for half of the others, unknown for the rest.
    risk = np.where(rng.random(n_clients) < 0.5, rng.beta(1.0, 8.0, n_clients), np.nan)
    risk[in_ring] = rng.uniform(0.6, 0.98, int(in_ring.sum()))

    # Transactions: ring clients are more active and far more often fraudulent.
    weight = np.where(in_ring, 2.0, 1.0) * rng.gamma(2.0, 1.0, n_clients)
    c = rng.choice(n_clients, size=n, p=weight / weight.sum())
    fraud = (rng.random(n) < np.where(in_ring[c], 0.2, 0.02)).astype(int)
    p_xgb = np.where(fraud == 1, rng.beta(2.0, 2.0, n), rng.beta(1.4, 7.0, n))
    g = risk[c]
    score = np.where(np.isnan(g), p_xgb, WEIGHT_XGB * p_xgb + (1 - WEIGHT_XGB) * g)
    threshold = round(float(np.quantile(score, 1 - alert_rate)), 4)
    alert = score >= threshold
    amount = np.round(np.exp(rng.normal(4.0, 1.0, n)) * (1 + fraud), 2)
    latency = np.exp(rng.normal(np.log(18.0), 0.35, n))
    spikes = rng.random(n) < 0.01
    latency[spikes] *= rng.uniform(3, 6, int(spikes.sum()))
    scored_at = now - window_s + np.sort(rng.uniform(0, window_s, n))
    scored_at[-1] = now
    event_time = 15_811_200 + np.cumsum(rng.integers(1, 90, n))  # synthetic seconds

    # Analysts have already reviewed the oldest share of alerts, agreeing with the truth ~92%.
    status = np.where(alert, "open", None).astype(object)
    feedback = np.full(n, np.nan)
    reviewed_at = np.full(n, np.nan)
    alert_idx = np.flatnonzero(alert)
    reviewed = alert_idx[: int(len(alert_idx) * review_share)]
    correct = rng.random(len(reviewed)) < 0.92
    feedback[reviewed] = np.where(correct, fraud[reviewed], 1 - fraud[reviewed])
    status[reviewed] = np.where(feedback[reviewed] == 1, "confirmed", "dismissed")
    reviewed_at[reviewed] = np.minimum(
        scored_at[reviewed] + rng.uniform(20, 240, len(reviewed)), now
    )

    scored = pd.DataFrame(
        {
            "transaction_id": DEMO_ID_START + np.arange(n),
            "event_time": event_time.astype(float),
            "scored_at": scored_at,
            "client_key": clients[c],
            "card_key": cards[c],
            "device_key": devices[c],
            "amount": amount,
            "product": rng.choice(PRODUCTS, size=n),
            "score": np.round(score, 6),
            "p_xgb": np.round(p_xgb, 6),
            "gnn_risk": np.round(g, 6),
            "alert": alert.astype(int),
            "latency_ms": np.round(latency, 3),
            "true_label": fraud,
            "status": status,
            "feedback_label": feedback,
            "reviewed_at": reviewed_at,
        }
    )

    http = latency * rng.uniform(0.45, 0.6, n)
    feats = latency * rng.uniform(0.15, 0.25, n)
    push = latency * rng.uniform(0.05, 0.1, n)
    stats = {
        "demo": True,
        "note": "Synthetic snapshot for the public demo; not measured on the live stack.",
        "processed": n,
        "alerts": int(alert.sum()),
        "wall_tps": n / window_s,
        "capacity_tps": round(1000.0 / float(latency.mean()) * 8, 1),
        "batch_size": 200,
        "features_ms": _pcts(feats),
        "push_ms": _pcts(push),
        "http_ms": _pcts(http),
        "e2e_ms": _pcts(latency + rng.uniform(1, 4, n)),
    }
    model = {
        "model": "demo: synthetic scores (no trained model)",
        "threshold": threshold,
        "weight_xgb": WEIGHT_XGB,
        "clients_with_risk": int((~np.isnan(risk)).sum()),
        "demo": True,
    }
    return {"scored": scored, "rings": rings, "stats": stats, "model": model}


def _none(v: Any) -> Any:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def write_case_db(scored: pd.DataFrame, path: str | Path) -> None:
    """Write the demo rows into a fresh CaseStore DB (same schema as the consumer's)."""
    path = Path(path)
    for p in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        p.unlink(missing_ok=True)
    records = [{k: _none(v) for k, v in r.items()} for r in scored.to_dict("records")]
    store = CaseStore(path)
    try:
        store.insert(records)
        store.conn.executemany(
            "UPDATE scored SET scored_at=?, status=?, feedback_label=?, reviewed_at=? "
            "WHERE transaction_id=?",
            [
                (
                    r["scored_at"],
                    r["status"],
                    None if r["feedback_label"] is None else int(r["feedback_label"]),
                    r["reviewed_at"],
                    r["transaction_id"],
                )
                for r in records
            ],
        )
        store.conn.commit()
    finally:
        store.close()


def write_demo(
    out_dir: str | Path, n: int = 4000, seed: int = 2026, now: float | None = None
) -> dict[str, Path]:
    """Build the demo into out_dir; returns the paths keyed like the dashboard config
    (cases_db, rings_path, stats_path) plus model_path. Overwrites an earlier demo."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    demo = make_demo(n=n, seed=seed, now=now)
    paths = {k: out / v for k, v in FILES.items()}
    paths["model_path"] = out / MODEL_FILE
    write_case_db(demo["scored"], paths["cases_db"])
    demo["rings"].to_parquet(paths["rings_path"], index=False)
    paths["stats_path"].write_text(json.dumps(demo["stats"], indent=2), encoding="utf-8")
    paths["model_path"].write_text(json.dumps(demo["model"], indent=2), encoding="utf-8")
    return paths
