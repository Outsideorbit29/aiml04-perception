"""Online feature store clients: Feast (serving) and an in-memory store (tests, local runs).

Contract: the consumer `push`es the F2 features of each transaction (keyed by client_key, with
its TransactionID) before asking the scorer; the scorer `get`s them by client_key and checks the
TransactionID matches, so it never scores with another transaction's features.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class OnlineStore(Protocol):
    def push(self, rows: Sequence[Mapping[str, Any]]) -> None: ...

    def get(self, client_key: str) -> dict[str, Any] | None: ...


class MemoryStore:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def push(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for r in rows:
            self._rows[str(r["client_key"])] = dict(r)

    def get(self, client_key: str) -> dict[str, Any] | None:
        row = self._rows.get(client_key)
        return dict(row) if row is not None else None


class FeastStore:
    """Feast online store (Redis in the stack); `repo_path` holds feature_store.yaml."""

    def __init__(self, repo_path: str, features: Sequence[str], push_source: str, view: str):
        from feast import FeatureStore

        self.store = FeatureStore(repo_path=repo_path)
        self.features = list(features)
        self.push_source = push_source
        self.refs = [f"{view}:{n}" for n in [*self.features, "TransactionID"]]

    def push(self, rows: Sequence[Mapping[str, Any]]) -> None:
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "client_key": r["client_key"],
                    "event_timestamp": r["event_timestamp"],
                    "TransactionID": int(r["TransactionID"]),
                    **{n: _f32(r.get(n)) for n in self.features},
                }
                for r in rows
            ]
        )
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
        self.store.push(self.push_source, df)

    def get(self, client_key: str) -> dict[str, Any] | None:
        out = self.store.get_online_features(
            features=self.refs, entity_rows=[{"client_key": client_key}]
        ).to_dict()
        tx_id = out.get("TransactionID", [None])[0]
        if tx_id is None:
            return None
        row = {n: out[n][0] for n in self.features}
        row = {k: (math.nan if v is None else float(v)) for k, v in row.items()}
        return {"client_key": client_key, "TransactionID": int(tx_id), **row}


def _f32(v: Any) -> float:
    return math.nan if v is None else float(v)
