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


def ensure_topics(cfg: dict) -> None:
    """Create the topics up front so the consumer doesn't wait for a metadata refresh."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": cfg["kafka_bootstrap"]})
    existing = set(admin.list_topics(timeout=10).topics)
    new = [
        NewTopic(t, num_partitions=1, replication_factor=1)
        for t in (cfg["topic_transactions"], cfg["topic_scores"])
        if t not in existing
    ]
    for topic, fut in admin.create_topics(new).items() if new else []:
        try:
            fut.result()
            print(f"created topic {topic}", flush=True)
        except Exception as exc:  # already created by another process
            print(f"topic {topic}: {exc}", flush=True)


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
    ensure_topics(cfg)
    consumer.subscribe([cfg["topic_transactions"]])
    out = Producer({"bootstrap.servers": cfg["kafka_bootstrap"]})

    timings: dict[str, list[float]] = {
        "features_ms": [],
        "push_ms": [],
        "http_ms": [],
        "e2e_ms": [],
    }
    n, alerts, busy_s, t_first = 0, 0, 0.0, None
    print("consumer ready", flush=True)

    def flush(pending: list[tuple[dict, dict, float]]) -> list[dict]:
        """Push a group's features in one call, then score each transaction."""
        if not pending:
            return []
        t0 = time.perf_counter()
        store.push(
            [
                {
                    **f,
                    "TransactionID": tx["TransactionID"],
                    "event_timestamp": start + pd.Timedelta(seconds=int(tx["TransactionDT"])),
                }
                for tx, f, _ in pending
            ]
        )
        push_ms = (time.perf_counter() - t0) * 1000 / len(pending)
        rows = []
        for tx, f, feat_ms in pending:
            t1 = time.perf_counter()
            resp = http.post("/score", json=json_safe(tx))
            http_ms = (time.perf_counter() - t1) * 1000
            if resp.status_code != 200:
                print(
                    f"scorer {resp.status_code} for {tx['TransactionID']}: {resp.text}", flush=True
                )
                continue
            r = resp.json()
            timings["features_ms"].append(feat_ms)
            timings["push_ms"].append(push_ms)
            timings["http_ms"].append(http_ms)
            rows.append({"tx": tx, "f": f, "r": r, "latency_ms": feat_ms + push_ms + http_ms})
        return rows

    try:
        while True:
            msgs = consumer.consume(
                num_messages=cfg["consumer_batch"], timeout=cfg["consumer_poll_s"]
            )
            if not msgs:
                continue
            t_batch = time.perf_counter()
            t_first = t_first or time.time()
            scored, pending, seen = [], [], set()
            for msg in msgs:
                if msg.error():
                    print("kafka error:", msg.error(), flush=True)
                    continue
                tx: dict[str, Any] = json.loads(msg.value())
                t0 = time.perf_counter()
                f = state.update(tx)
                feat_ms = (time.perf_counter() - t0) * 1000
                # The online store keeps one row per client: flush before a client repeats so
                # the scorer always finds this transaction's own features.
                if f["client_key"] in seen:
                    scored += flush(pending)
                    pending, seen = [], set()
                pending.append((tx, f, feat_ms))
                seen.add(f["client_key"])
            scored += flush(pending)
            cases.insert(
                [
                    {
                        "transaction_id": s["tx"]["TransactionID"],
                        "event_time": s["tx"]["TransactionDT"],
                        "client_key": s["f"]["client_key"],
                        "card_key": s["f"]["card_key"],
                        "device_key": s["f"]["device_key"]
                        if isinstance(s["f"]["device_key"], str)
                        else None,
                        "amount": s["tx"]["TransactionAmt"],
                        "product": s["tx"].get("ProductCD"),
                        "score": s["r"]["score"],
                        "p_xgb": s["r"]["p_xgb"],
                        "gnn_risk": s["r"]["gnn_risk"],
                        "alert": s["r"]["alert"],
                        "latency_ms": s["latency_ms"],
                        "true_label": s["tx"].get("isFraud"),
                    }
                    for s in scored
                ]
            )
            now = time.time()
            for s in scored:
                out.produce(
                    cfg["topic_scores"], key=str(s["tx"]["TransactionID"]), value=json.dumps(s["r"])
                )
                if s["tx"].get("_produced_at"):
                    timings["e2e_ms"].append((now - s["tx"]["_produced_at"]) * 1000)
            out.poll(0)
            busy_s += time.perf_counter() - t_batch
            before, n = n, n + len(scored)
            alerts += sum(int(s["r"]["alert"]) for s in scored)
            if n // cfg["consumer_stats_every"] > before // cfg["consumer_stats_every"]:
                stats = {
                    "processed": n,
                    "alerts": alerts,
                    "wall_tps": n / max(time.time() - t_first, 1e-9),
                    "capacity_tps": n / max(busy_s, 1e-9),
                    "batch_size": cfg["consumer_batch"],
                    "features_names": len(names),
                    **{k: percentiles(v[-5000:]) for k, v in timings.items()},
                    "cases": cases.stats(),
                }
                with open(cfg["stats_path"], "w") as fh:
                    json.dump(stats, fh, indent=2)
                print(
                    f"{n} tx, wall {stats['wall_tps']:.1f} tx/s, capacity "
                    f"{stats['capacity_tps']:.1f} tx/s, http p95 "
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
