"""Model input encoding shared by training and serving.

`FeatureSpec` fixes the column order and the category vocabularies learned on the training split,
so the scorer turns an incoming transaction into exactly the matrix the model was trained on.
Categoricals become integer codes (unknown or missing -> NaN), which tree models handle directly.
If `medians` is set (models trained on SMOTE-resampled data), NaNs are replaced by the training
medians, exactly as during training.
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
    medians: list[float] | None = None

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
        x = np.column_stack(cols) if cols else np.empty((len(df), 0), dtype="float32")
        if self.medians is not None:
            med = np.asarray(self.medians, dtype="float32")
            rows, idx = np.where(np.isnan(x))
            x[rows, idx] = med[idx]
        return x

    def transform_row(self, row: Mapping[str, Any]) -> np.ndarray:
        """One transaction (dict) -> 1 x n_features matrix; same encoding as `transform`.

        Pure Python (no DataFrame), so it costs microseconds on the serving path.
        """
        if not hasattr(self, "_index"):
            self._index = {c: {v: i for i, v in enumerate(self.vocab[c])} for c in self.categorical}
        vals: list[float] = []
        for c in self.numeric:
            v = row.get(c)
            try:
                f = float(v) if v is not None else np.nan
            except (TypeError, ValueError):
                f = np.nan
            vals.append(f)
        for c in self.categorical:
            v = row.get(c)
            missing = v is None or (isinstance(v, float) and v != v)
            vals.append(np.nan if missing else float(self._index[c].get(str(v), np.nan)))
        x = np.asarray([vals], dtype="float32")
        if self.medians is not None:
            nan = np.isnan(x[0])
            x[0, nan] = np.asarray(self.medians, dtype="float32")[nan]
        return x

    def save(self, path: str | Path) -> None:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> FeatureSpec:
        return cls(**json.loads(Path(path).read_text()))
