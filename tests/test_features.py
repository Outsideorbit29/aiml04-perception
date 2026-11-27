import math

import numpy as np
import pandas as pd
import pytest

from fraud.config import load_config
from fraud.features import (
    OnlineFeatureState,
    build_features,
    card_key,
    feature_names,
    online_features,
    parity,
    row_key,
)

CFG = load_config("configs/features.yaml")
NAMES = feature_names(CFG["windows"])


def _tx(t, amt, card1=1, dev=None, addr=299.0, email="gmail.com", card4="visa"):
    return {
        "TransactionID": t,
        "TransactionDT": t,
        "TransactionAmt": amt,
        "card1": card1,
        "card2": np.nan,
        "card4": card4,
        "addr1": addr,
        "P_emaildomain": email,
        "DeviceInfo": dev,
    }


def test_hand_checked_example():
    df = pd.DataFrame(
        [
            _tx(1000, 10.0, dev="A"),
            _tx(2000, 30.0, dev="A"),
            _tx(2000 + 3600, 50.0, dev="B", addr=100.0),  # exactly 1h after the 2nd
            _tx(5000, 99.0, card1=2, dev="A"),  # other card, same device
        ]
    )
    f = build_features(df, CFG)
    third = f.iloc[2]
    assert third["card_prior_count"] == 2
    assert third["card_amt_mean"] == pytest.approx(20.0)
    assert third["card_amt_std"] == pytest.approx(math.sqrt(200.0))
    assert third["amt_to_card_mean"] == pytest.approx(2.5)
    assert third["secs_since_prev"] == 3600
    assert third["cnt_1h"] == 1 and third["sum_1h"] == pytest.approx(30.0)  # [t-1h, t)
    assert third["cnt_24h"] == 2
    assert third["new_device"] == 1 and third["new_addr"] == 1 and third["new_email"] == 0
    assert third["card_n_devices"] == 1
    other = f.iloc[3]
    assert other["card_prior_count"] == 0 and math.isnan(other["card_amt_mean"])
    assert other["device_prior_count"] == 2 and other["device_n_cards"] == 1
    assert f.iloc[0]["cnt_7d"] == 0 and math.isnan(f.iloc[0]["secs_since_prev"])


def test_missing_device_gives_nan():
    df = pd.DataFrame([_tx(1, 5.0, dev=None), _tx(2, 6.0, dev=None)])
    f = build_features(df, CFG)
    assert f["new_device"].isna().all() and f["device_n_cards"].isna().all()
    assert f["card_n_devices"].tolist() == [0, 0]


def test_card_key_matches_row_key():
    df = pd.DataFrame([_tx(1, 1.0, card1=1234), _tx(2, 1.0, card4=None)])
    keys = card_key(df, CFG["card_key_columns"])
    for i, row in enumerate(df.to_dict("records")):
        assert keys.iloc[i] == row_key(row, CFG["card_key_columns"])
    assert keys.iloc[0] == "1234|NA|visa"


def _random_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 20 * 86400, size=n))
    t[5:15] = t[5]  # same-second ties
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": t,
            "TransactionAmt": np.round(rng.lognormal(3, 1, n), 2).astype(np.float32),
            "card1": rng.choice([1, 2, 3, 4, 5], n),
            "card2": rng.choice([np.nan, 100.0], n),
            "card4": rng.choice(["visa", "mastercard"], n),
            "addr1": rng.choice([np.nan, 10.0, 20.0, 30.0], n),
            "P_emaildomain": rng.choice(np.array(["a.com", "b.com", None], dtype=object), n),
            "DeviceInfo": rng.choice(np.array(["d1", "d2", "d3", None], dtype=object), n),
        }
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_offline_online_parity(seed):
    df = _random_frame(seed=seed).sample(frac=1.0, random_state=seed)  # order must not matter
    off = build_features(df, CFG)
    on = online_features(df, CFG)
    assert (off["card_key"] == on["card_key"]).all()
    report = parity(off, on, NAMES)
    assert max(report.values()) < 1e-6, report


def test_constant_amounts_std_is_zero_not_nan():
    df = pd.DataFrame([_tx(i, 9.99) for i in range(1, 6)])
    off = build_features(df, CFG)
    on = online_features(df, CFG)
    assert off["card_amt_std"].iloc[2:].eq(0).all()
    assert off["amt_zscore"].isna().all() and on["amt_zscore"].isna().all()


def test_online_state_evicts_old_transactions():
    state = OnlineFeatureState(CFG)
    state.update(_tx(0, 1.0))
    f = state.update(_tx(10 * 86400, 1.0))
    assert f["cnt_7d"] == 0 and f["card_prior_count"] == 1
    assert len(state.cards[f["card_key"]].recent) == 1
