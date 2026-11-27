import math

import numpy as np
import pandas as pd
import pytest

from fraud.config import load_config
from fraud.features import (
    KEY_COLUMNS,
    OnlineFeatureState,
    add_keys,
    build_features,
    feature_names,
    online_features,
    parity,
    row_keys,
)

CFG = load_config("configs/features.yaml")
NAMES = feature_names(CFG["windows"])
DAY = 86_400


def _tx(t, amt, card1=1, d1=None, dev=None, os_=None, addr=299.0, email="gmail.com"):
    """One transaction; d1 defaults to `t`'s day (so d1n = 0 -> one client per card1/addr1)."""
    return {
        "TransactionID": t,
        "TransactionDT": t,
        "TransactionAmt": amt,
        "card1": card1,
        "card2": np.nan,
        "card4": "visa",
        "addr1": addr,
        "D1": float(t // DAY) if d1 is None else d1,
        "P_emaildomain": email,
        "DeviceInfo": dev,
        "id_30": os_,
    }


def test_keys():
    df = pd.DataFrame(
        [
            _tx(10 * DAY, 1.0, d1=3.0, dev="SM-G950F", os_="Android 7.0"),
            _tx(10 * DAY, 1.0, d1=np.nan, addr=np.nan),
        ]
    )
    keys = add_keys(df, CFG)
    assert keys["client_key"].tolist() == ["1|299|7", "1|NA|NA"]
    assert keys["card_key"].iloc[0] == "1|NA|visa"
    assert keys["device_key"].iloc[0] == "SM-G950F|Android 7.0"
    assert pd.isna(keys["device_key"].iloc[1])
    for i, row in enumerate(df.to_dict("records")):
        client, card, device = row_keys(row, CFG)
        assert client == keys["client_key"].iloc[i] and card == keys["card_key"].iloc[i]
        assert (device is None) == pd.isna(keys["device_key"].iloc[i])


def test_d1n_groups_one_client_across_days():
    # Same cardholder: D1 grows with the day, so day - D1 stays 5.
    df = pd.DataFrame([_tx(10 * DAY, 5.0, d1=5.0), _tx(12 * DAY, 7.0, d1=7.0)])
    f = build_features(df, CFG)
    assert f["client_key"].nunique() == 1
    assert f["client_prior_count"].tolist() == [0, 1]


def test_hand_checked_example():
    t0 = 20 * DAY
    df = pd.DataFrame(
        [
            _tx(t0 + 1000, 10.0, dev="A"),
            _tx(t0 + 2000, 30.0, dev="A"),
            _tx(t0 + 2000 + 3600, 50.0, dev="B", email="yahoo.com"),  # 1h after the 2nd
            _tx(t0 + 5000, 99.0, card1=2, dev="A"),  # other card and client, same device
            _tx(t0 + 9000, 5.0, addr=100.0),  # same card, new address -> new client
        ]
    )
    f = build_features(df, CFG)
    third = f.iloc[2]
    assert third["client_prior_count"] == 2
    assert third["client_amt_mean"] == pytest.approx(20.0)
    assert third["client_amt_std"] == pytest.approx(math.sqrt(200.0))
    assert third["amt_to_client_mean"] == pytest.approx(2.5)
    assert third["secs_since_prev"] == 3600
    assert third["cnt_1h"] == 1 and third["sum_1h"] == pytest.approx(30.0)  # [t-1h, t)
    assert third["cnt_24h"] == 2
    assert third["new_device"] == 1 and third["new_addr"] == 0 and third["new_email"] == 1
    assert third["card_n_devices"] == 1
    other = f.iloc[3]
    assert other["client_prior_count"] == 0 and math.isnan(other["client_amt_mean"])
    assert other["device_prior_count"] == 2 and other["device_n_clients"] == 1
    assert f.iloc[0]["cnt_7d"] == 0 and math.isnan(f.iloc[0]["secs_since_prev"])
    moved = f.iloc[4]
    assert moved["new_addr"] == 1 and moved["client_prior_count"] == 0
    assert moved["card_key"] == f.iloc[0]["card_key"]


def test_missing_device_gives_nan():
    df = pd.DataFrame([_tx(DAY + 1, 5.0), _tx(DAY + 2, 6.0)])
    f = build_features(df, CFG)
    assert f["new_device"].isna().all() and f["device_n_clients"].isna().all()
    assert f["card_n_devices"].tolist() == [0, 0]


def _random_frame(n=800, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 20 * DAY, size=n))
    t[5:15] = t[5]  # same-second ties
    day = t // DAY
    start = rng.choice([0, 1, 2, -1], n)  # 4 "cardholders" per card1/addr1 via d1n
    d1 = np.where(start >= 0, day - start, np.nan).astype(float)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": t,
            "TransactionAmt": np.round(rng.lognormal(3, 1, n), 2).astype(np.float32),
            "card1": rng.choice([1, 2, 3], n),
            "card2": rng.choice([np.nan, 100.0], n),
            "card4": rng.choice(["visa", "mastercard"], n),
            "addr1": rng.choice([np.nan, 10.0, 20.0], n),
            "D1": d1,
            "P_emaildomain": rng.choice(np.array(["a.com", "b.com", None], dtype=object), n),
            "DeviceInfo": rng.choice(np.array(["d1", "d2", None], dtype=object), n),
            "id_30": rng.choice(np.array(["iOS", None], dtype=object), n),
        }
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_offline_online_parity(seed):
    df = _random_frame(seed=seed).sample(frac=1.0, random_state=seed)  # order must not matter
    off = build_features(df, CFG)
    on = online_features(df, CFG)
    for k in KEY_COLUMNS:
        assert (off[k].fillna("<none>") == on[k].fillna("<none>")).all(), k
    report = parity(off, on, NAMES)
    assert max(report.values()) < 1e-6, report


def test_constant_amounts_std_is_zero_not_nan():
    df = pd.DataFrame([_tx(DAY + i, 9.99) for i in range(5)])
    off = build_features(df, CFG)
    on = online_features(df, CFG)
    assert off["client_amt_std"].iloc[2:].eq(0).all()
    assert off["amt_zscore"].isna().all() and on["amt_zscore"].isna().all()


def test_online_state_evicts_old_transactions():
    state = OnlineFeatureState(CFG)
    state.update(_tx(DAY, 1.0, d1=1.0))
    f = state.update(_tx(11 * DAY, 1.0, d1=11.0))
    assert f["cnt_7d"] == 0 and f["client_prior_count"] == 1
    assert len(state.clients[f["client_key"]].recent) == 1
