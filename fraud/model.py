"""Model input encoding shared by training and serving.

`FeatureSpec` fixes the column order and the category vocabularies learned on the training split,
so the scorer turns an incoming transaction into exactly the matrix the model was trained on.
Categoricals become integer codes (unknown or missing -> NaN), which tree models handle directly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]
    vocab: dict[str, list[str]] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        numeric: Sequence[str],
        categorical: Sequence[str],
        min_count: int = 20,
    ) -> FeatureSpec:
        """Vocabulary = categories seen at least `min_count` times in training (rest -> NaN)."""
        vocab = {}
        for c in categorical:
            counts = train[c].astype("object").dropna().astype(str).value_counts()
            vocab[c] = sorted(counts[counts >= min_count].index.tolist())
        return cls(list(numeric), list(categorical), vocab)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.numeric:
            cols.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float32"))
        for c in self.categorical:
            index = {v: i for i, v in enumerate(self.vocab[c])}
            s = df[c].astype("object")
            codes = s.map(lambda v, idx=index: idx.get(str(v), np.nan) if pd.notna(v) else np.nan)
            cols.append(codes.to_numpy(dtype="float32"))
        return np.column_stack(cols) if cols else np.empty((len(df), 0), dtype="float32")

    def transform_row(self, row: Mapping[str, Any]) -> np.ndarray:
        """One transaction (dict) -> 1 x n_features matrix; same encoding as `transform`."""
        return self.transform(pd.DataFrame([dict(row)]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> FeatureSpec:
        return cls(**json.loads(Path(path).read_text()))
