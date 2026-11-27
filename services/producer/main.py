"""Replay producer: test-period transactions -> Kafka topic, in time order, at a fixed rate.

Run: python -m services.producer.main [--rate 50] [--limit 2000] [--start 0]
Each message is the raw transaction as JSON (NaN -> null) plus `_produced_at` (unix time) so the
consumer can measure end-to-end latency. The label isFraud is included for offline evaluation
in the case store; the scorer never uses it.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from services.common import json_safe, serving_config


def main(argv: list[str] | None = None) -> None:
    from confluent_kafka import Producer

    cfg = serving_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate", type=float, default=float(cfg["producer_rate"]))
    ap.add_argument("--limit", type=int, default=int(cfg["producer_limit"]))
    ap.add_argument("--start", type=int, default=0, help="skip this many replay rows")
    args = ap.parse_args(argv)

    df = pd.read_parquet(cfg["replay_path"]).sort_values("TransactionDT", kind="stable")
    df = df.iloc[args.start :]
    if args.limit:
        df = df.iloc[: args.limit]
    producer = Producer({"bootstrap.servers": cfg["kafka_bootstrap"], "linger.ms": 5})
    interval = 1.0 / args.rate if args.rate > 0 else 0.0
    t_next = time.time()
    print(f"replaying {len(df):,} transactions at {args.rate} tx/s", flush=True)
    for i, row in enumerate(df.to_dict("records"), 1):
        msg = json_safe(row)
        msg["_produced_at"] = time.time()
        producer.produce(
            cfg["topic_transactions"], key=str(msg["TransactionID"]), value=json.dumps(msg)
        )
        producer.poll(0)
        if interval:
            t_next += interval
            delay = t_next - time.time()
            if delay > 0:
                time.sleep(delay)
        if i % 1000 == 0:
            print(f"produced {i:,}", flush=True)
    producer.flush(30)
    print("replay done", flush=True)


if __name__ == "__main__":
    main()
