"""IEEE-CIS Fraud Detection: loading, joining and time-ordered splitting.

The labelled data is `train_transaction.csv` (one row per transaction, target `isFraud`) left-joined
with `train_identity.csv` (device and network fields for about a quarter of transactions) on
`TransactionID`. The competition test set has no labels, so train/val/test all come from the
labelled file, split by time so validation never sees the future.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"
SPLIT_NAMES = ("train", "val", "test")


def find_data_dir(data_dir: str | Path, search_root: str | Path = "/kaggle/input") -> Path:
    """Resolve `data_dir`; "auto" searches `search_root` (a few levels deep) for the CSVs."""
    if str(data_dir) != "auto":
        path = Path(data_dir)
        if not (path / TRANSACTION_FILE).exists():
            raise FileNotFoundError(f"{path / TRANSACTION_FILE} not found")
        return path
    for depth in range(1, 5):
        pattern = os.path.join(str(search_root), *["*"] * depth, TRANSACTION_FILE)
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError(f"{TRANSACTION_FILE} not found under {search_root}")


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric columns (float64 -> float32, int64 -> smallest int) to save memory."""
    out = df.copy()
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = out[col].astype(np.float32)
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


def load_raw(
    data_dir: str | Path, nrows: int | None = None, id_column: str = "TransactionID"
) -> pd.DataFrame:
    """Read the transaction and identity tables and left-join them on `id_column`."""
    data_dir = Path(data_dir)
    tx = pd.read_csv(data_dir / TRANSACTION_FILE, nrows=nrows)
    ident_path = data_dir / IDENTITY_FILE
    if ident_path.exists():
        ident = pd.read_csv(ident_path)
        df = tx.merge(ident, on=id_column, how="left", validate="one_to_one", indicator=True)
        df["has_identity"] = df.pop("_merge").eq("both")
    else:
        df = tx.assign(has_identity=False)
    return downcast(df)


def add_time_columns(
    df: pd.DataFrame, start_date: str, time_column: str = "TransactionDT"
) -> pd.DataFrame:
    """Add `event_time` (timestamp), `day` (whole days from start), `hour` and `weekday`."""
    out = df.copy()
    seconds = out[time_column].astype("int64")
    out["event_time"] = pd.Timestamp(start_date) + pd.to_timedelta(seconds, unit="s")
    out["day"] = (seconds // 86_400).astype("int32")
    out["hour"] = out["event_time"].dt.hour.astype("int8")
    out["weekday"] = out["event_time"].dt.weekday.astype("int8")
    return out


def time_split(
    df: pd.DataFrame, fractions: Mapping[str, float], time_column: str = "TransactionDT"
) -> dict[str, pd.DataFrame]:
    """Split rows in time order into train/val/test by the given fractions (must sum to 1)."""
    missing = set(SPLIT_NAMES) - set(fractions)
    if missing:
        raise ValueError(f"split fractions missing {sorted(missing)}")
    total = sum(fractions[name] for name in SPLIT_NAMES)
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions sum to {total}, expected 1.0")
    ordered = df.sort_values(time_column, kind="stable").reset_index(drop=True)
    n = len(ordered)
    cut_train = int(round(n * fractions["train"]))
    cut_val = cut_train + int(round(n * fractions["val"]))
    return {
        "train": ordered.iloc[:cut_train].reset_index(drop=True),
        "val": ordered.iloc[cut_train:cut_val].reset_index(drop=True),
        "test": ordered.iloc[cut_val:].reset_index(drop=True),
    }


def split_summary(
    splits: Mapping[str, pd.DataFrame],
    target: str = "isFraud",
    time_column: str = "TransactionDT",
) -> dict[str, dict[str, float]]:
    """Rows, fraud count/rate and time range per split."""
    summary: dict[str, dict[str, float]] = {}
    for name, part in splits.items():
        n = len(part)
        frauds = int(part[target].sum()) if n else 0
        summary[name] = {
            "rows": n,
            "frauds": frauds,
            "fraud_rate": frauds / n if n else 0.0,
            "time_min": int(part[time_column].min()) if n else 0,
            "time_max": int(part[time_column].max()) if n else 0,
        }
    return summary
