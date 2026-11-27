"""Feast definitions: per-card behavioural features (see fraud/features.py for definitions).

Training reads them with `get_historical_features` (point-in-time join on card_key and the
transaction time). Serving: the stream consumer computes the features for each incoming
transaction with `fraud.features.OnlineFeatureState` and pushes them to `card_features_push`;
the scorer reads them back with `get_online_features`. One feature view serves both paths.
"""

import os
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, PushSource, ValueType
from feast.types import Float32

# Must equal fraud.features.feature_names(configs/features.yaml windows); a test checks this.
FEATURES = [
    "card_prior_count",
    "card_amt_mean",
    "card_amt_std",
    "amt_to_card_mean",
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
    "device_n_cards",
]

card = Entity(name="card", join_keys=["card_key"], value_type=ValueType.STRING)

card_source = FileSource(
    name="card_features_file",
    path=os.environ.get("FRAUD_CARD_FEATURES", "data/card_features.parquet"),
    timestamp_field="event_timestamp",
)

card_push = PushSource(name="card_features_push", batch_source=card_source)

card_features = FeatureView(
    name="card_features",
    entities=[card],
    ttl=timedelta(days=400),
    schema=[Field(name=name, dtype=Float32) for name in FEATURES],
    source=card_push,
    online=True,
)
