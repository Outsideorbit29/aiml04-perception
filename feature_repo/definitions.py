"""Feast definitions: per-client behavioural features (see fraud/features.py for definitions).

The entity is the client key (card1 | addr1 | day - D1). Training reads the features with
`get_historical_features` (point-in-time join on client_key and the transaction time). Serving:
the stream consumer computes them for each incoming transaction with
`fraud.features.OnlineFeatureState` and pushes them to `client_features_push`; the scorer reads
them back with `get_online_features`. One feature view serves both paths.
"""

import os
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, PushSource, ValueType
from feast.types import Float32, Int64

# Must equal fraud.features.feature_names(configs/features.yaml windows); a test checks this.
FEATURES = [
    "client_prior_count",
    "client_amt_mean",
    "client_amt_std",
    "amt_to_client_mean",
    "amt_zscore",
    "secs_since_prev",
    "cnt_1h",
    "sum_1h",
    "cnt_24h",
    "sum_24h",
    "cnt_7d",
    "sum_7d",
    "new_device",
    "new_addr",
    "new_email",
    "card_n_devices",
    "device_prior_count",
    "device_n_clients",
]

client = Entity(name="client", join_keys=["client_key"], value_type=ValueType.STRING)

client_source = FileSource(
    name="client_features_file",
    path=os.environ.get("FRAUD_CLIENT_FEATURES", "data/client_features.parquet"),
    timestamp_field="event_timestamp",
)

client_push = PushSource(name="client_features_push", batch_source=client_source)

client_features = FeatureView(
    name="client_features",
    entities=[client],
    ttl=timedelta(days=400),
    # TransactionID lets the scorer check the features belong to the transaction it is scoring.
    schema=[Field(name=name, dtype=Float32) for name in FEATURES]
    + [Field(name="TransactionID", dtype=Int64)],
    source=client_push,
    online=True,
)
