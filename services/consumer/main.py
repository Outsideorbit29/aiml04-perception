"""Stream consumer: Kafka transactions -> features -> online store -> scorer -> case store.

For each transaction, in order:
1. OnlineFeatureState.update(tx)  (same code as training; warm-started from history.parquet)
2. push the features to the online store (Feast / Redis), tagged with the TransactionID
3. POST the raw transaction to the scorer, which fetches those features and returns the score
4. write the result to the case store and publish it to the `scores` topic
Latency and throughput stats are written to stats_path every consumer_stats_every messages.
Run: python -m services.consumer.main
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pandas as pd

from fraud.cases import CaseStore
from fraud.config import load_config
from fraud.features import OnlineFeatureState, feature_names
from services.common import json_safe, percentiles, serving_config
from services.scorer.app import build_store


def warm_state(cfg: dict, feat_cfg: dict) -> OnlineFeatureState:
    t0 = time.time()
    history = pd.read_parquet(cfg["history_path"])
    state = OnlineFeatureState.from_history(history, feat_cfg)
    print(
        f"feature state warmed from {len(history):,} transactions, {len(state.clients):,} clients "
        f"in {time.time() - t0:.1f} s",
        flush=True,
    )
    return state


def main() -> None:
    import httpx
    from confluent_kafka import Consumer, Producer

    cfg = serving_config()
    feat_cfg = load_config(os.environ.get("FEATURES_CONFIG", "configs/features.yaml"))
    names = feature_names(feat_cfg["windows"])
    state = warm_state(cfg, feat_cfg)
    store = build_store(cfg, feat_cfg)
    cases = CaseStore(cfg["cases_db"])
    start = pd.Timestamp(cfg["start_date"], tz="UTC")
    http = httpx.Client(base_url=cfg["scorer_url"], timeout=cfg["http_timeout_s"])
    consumer = Consumer(
        {
            "bootstrap.servers": cfg["kafka_bootstrap"],
            "group.id": cfg["consumer_group"],
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([cfg["topic_transactions"]])
    out = Producer({"bootstrap.servers": cfg["kafka_bootstrap"]})

    timings: dict[str, list[float]] = {
        "features_ms": [],
        "push_ms": [],
        "http_ms": [],
        "e2e_ms": [],
    }
    n, t_start, alerts = 0, time.time(), 0
    print("consumer ready", flush=True)
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print("kafka error:", msg.error(), flush=True)
                continue
            tx: dict[str, Any] = json.loads(msg.value())
            produced_at = tx.pop("_produced_at", None)
            t0 = time.perf_counter()
            f = state.update(tx)
            t1 = time.perf_counter()
            event_ts = start + pd.Timedelta(seconds=int(tx["TransactionDT"]))
            store.push([{**f, "TransactionID": tx["TransactionID"], "event_timestamp": event_ts}])
            t2 = time.perf_counter()
            resp = http.post("/score", json=json_safe(tx))
            t3 = time.perf_counter()
            if resp.status_code != 200:
                print(
                    f"scorer {resp.status_code} for {tx['TransactionID']}: {resp.text}", flush=True
                )
                continue
            r = resp.json()
            alerts += int(r["alert"])
            cases.insert(
                [
                    {
                        "transaction_id": tx["TransactionID"],
                        "event_time": tx["TransactionDT"],
                        "client_key": f["client_key"],
                        "card_key": f["card_key"],
                        "device_key": f["device_key"] if isinstance(f["device_key"], str) else None,
                        "amount": tx["TransactionAmt"],
                        "product": tx.get("ProductCD"),
                        "score": r["score"],
                        "p_xgb": r["p_xgb"],
                        "gnn_risk": r["gnn_risk"],
                        "alert": r["alert"],
                        "latency_ms": (t3 - t0) * 1000,
                        "true_label": tx.get("isFraud"),
                    }
                ]
            )
            out.produce(cfg["topic_scores"], key=str(tx["TransactionID"]), value=json.dumps(r))
            out.poll(0)
            timings["features_ms"].append((t1 - t0) * 1000)
            timings["push_ms"].append((t2 - t1) * 1000)
            timings["http_ms"].append((t3 - t2) * 1000)
            if produced_at:
                timings["e2e_ms"].append((time.time() - produced_at) * 1000)
            n += 1
            if n % cfg["consumer_stats_every"] == 0:
                stats = {
                    "processed": n,
                    "alerts": alerts,
                    "throughput_tps": n / (time.time() - t_start),
                    "features_names": len(names),
                    **{k: percentiles(v[-5000:]) for k, v in timings.items()},
                    "cases": cases.stats(),
                }
                with open(cfg["stats_path"], "w") as fh:
                    json.dump(stats, fh, indent=2)
                print(
                    f"{n} tx, {stats['throughput_tps']:.1f} tx/s, http p95 "
                    f"{stats['http_ms'].get('p95', 0):.1f} ms, e2e p95 "
                    f"{stats['e2e_ms'].get('p95', 0):.1f} ms, alerts {alerts}",
                    flush=True,
                )
    finally:
        consumer.close()
        out.flush(5)
        cases.close()


if __name__ == "__main__":
    main()
