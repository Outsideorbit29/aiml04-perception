from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud.data import (
    add_time_columns,
    downcast,
    find_data_dir,
    load_raw,
    split_summary,
    time_split,
)
from fraud.synthetic import make_synthetic

FIXTURES = Path(__file__).parent / "fixtures"
SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_load_raw_joins_identity():
    df = load_raw(FIXTURES)
    tx = pd.read_csv(FIXTURES / "train_transaction.csv")
    ident = pd.read_csv(FIXTURES / "train_identity.csv")
    assert len(df) == len(tx)  # left join keeps every transaction
    assert df["has_identity"].sum() == len(ident)
    assert df.loc[~df["has_identity"], "DeviceType"].isna().all()
    assert df["TransactionAmt"].dtype == np.float32


def test_find_data_dir(tmp_path):
    assert find_data_dir(FIXTURES) == FIXTURES
    nested = tmp_path / "comp" / "ieee"
    nested.mkdir(parents=True)
    (nested / "train_transaction.csv").write_text("TransactionID\n1\n")
    assert find_data_dir("auto", search_root=tmp_path) == nested
    with pytest.raises(FileNotFoundError):
        find_data_dir("auto", search_root=tmp_path / "nothing")


def test_time_split_is_ordered_and_complete():
    tx, _ = make_synthetic(1000, seed=1)
    shuffled = tx.sample(frac=1.0, random_state=0)
    splits = time_split(shuffled, SPLIT)
    assert sum(len(p) for p in splits.values()) == len(tx)
    assert len(splits["train"]) == 700 and len(splits["val"]) == 150
    assert splits["train"]["TransactionDT"].max() <= splits["val"]["TransactionDT"].min()
    assert splits["val"]["TransactionDT"].max() <= splits["test"]["TransactionDT"].min()
    assert not set(splits["train"]["TransactionID"]) & set(splits["test"]["TransactionID"])


def test_time_split_rejects_bad_fractions():
    tx, _ = make_synthetic(50)
    with pytest.raises(ValueError):
        time_split(tx, {"train": 0.8, "val": 0.1, "test": 0.2})
    with pytest.raises(ValueError):
        time_split(tx, {"train": 0.9, "val": 0.1})


def test_time_columns():
    df = pd.DataFrame({"TransactionDT": [86_400, 86_400 + 3_600 * 5 + 30]})
    out = add_time_columns(df, "2017-12-01")
    assert out["day"].tolist() == [1, 1]
    assert out["hour"].tolist() == [0, 5]
    assert out["event_time"].iloc[0] == pd.Timestamp("2017-12-02")


def test_split_summary_counts():
    tx, _ = make_synthetic(400, fraud_rate=0.1, seed=2)
    summary = split_summary(time_split(tx, SPLIT))
    assert sum(s["frauds"] for s in summary.values()) == tx["isFraud"].sum()
    assert all(0 <= s["fraud_rate"] <= 1 for s in summary.values())


def test_downcast_keeps_values():
    df = pd.DataFrame({"a": [1.5, np.nan], "b": [1, 2]})
    out = downcast(df)
    assert out["a"].dtype == np.float32 and out["b"].dtype == np.int8
    assert np.isnan(out["a"].iloc[1]) and out["b"].tolist() == [1, 2]
