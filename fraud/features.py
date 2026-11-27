"""Per-card and per-device behavioural features, offline (vectorised) and online (incremental).

Definitions (for a transaction at time t, using the card's earlier transactions in processing
order, i.e. sorted by time with a stable tie-break):
- card_prior_count, card_amt_mean, card_amt_std: count, mean and std of earlier amounts
- amt_to_card_mean: amount / card_amt_mean; amt_zscore: (amount - mean) / std
- secs_since_prev: seconds since the card's previous transaction
- cnt_<w>, sum_<w>: count and amount sum of earlier transactions with time in [t - w, t)
- new_device / new_addr / new_email: 1 if the value was never seen on this card before
  (NaN when the value is missing); card_n_devices: distinct devices seen before
- device_prior_count, device_n_cards: earlier transactions and distinct cards on the device

`build_features` (pandas, training) and `OnlineFeatureState` (one transaction at a time, serving)
implement the same definitions; tests check they agree.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

NA = "NA"


def card_key(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Concatenate the available card columns into one string key (missing -> "NA")."""
    cols = [c for c in columns if c in df]
    if not cols:
        raise ValueError(f"none of {list(columns)} in data")
    parts = [_as_key(df[c]) for c in cols]
    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    return key


def _as_key(s: pd.Series) -> pd.Series:
    """String form of a key column; floats that are whole numbers lose the '.0'."""
    if pd.api.types.is_float_dtype(s):
        out = s.map(_fmt)
    else:
        out = s.astype("object").where(s.notna(), NA).astype(str)
    return out.astype(str)


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
        return NA
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def row_key(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    """Card key for a single transaction dict; matches `card_key` on a frame."""
    return "|".join(_fmt(row[c]) for c in columns if c in row)


def feature_names(windows: Mapping[str, int]) -> list[str]:
    names = [
        "card_prior_count",
        "card_amt_mean",
        "card_amt_std",
        "amt_to_card_mean",
        "amt_zscore",
        "secs_since_prev",
    ]
    for w in windows:
        names += [f"cnt_{w}", f"sum_{w}"]
    names += [
        "new_device",
        "new_addr",
        "new_email",
        "card_n_devices",
        "device_prior_count",
        "device_n_cards",
    ]
    return names


# Variances below this (relative to mean^2 + 1) count as zero in both implementations, so
# float rounding on constant-amount cards can't make one side NaN and the other finite.
VAR_EPS = 1e-9


def _std(var: float, mean: float) -> float:
    return 0.0 if var <= VAR_EPS * (mean * mean + 1.0) else math.sqrt(var)


def _ratio_and_z(amount, mean, std):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mean > 0, amount / mean, np.nan)
        z = np.where(std > 0, (amount - mean) / std, np.nan)
    return ratio, z


def _first_seen(key: pd.Series, value: pd.Series) -> pd.Series:
    """1.0 if `value` was not seen earlier for this key, 0.0 if seen, NaN if value missing."""
    first = ~pd.DataFrame({"k": key, "v": value}).duplicated(["k", "v"])
    return first.astype(float).where(value.notna(), np.nan)


def build_features(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Offline features for every row of `df` (any order); returned in `df`'s index order."""
    t_col, a_col = cfg["time_column"], cfg["amount_column"]
    order = df.sort_values(t_col, kind="stable").index
    d = df.loc[order]
    key = card_key(d, cfg["card_key_columns"])
    t = d[t_col].astype("int64")
    amt = d[a_col].astype("float64")
    g = amt.groupby(key, sort=False)

    out = pd.DataFrame(index=d.index)
    count = g.cumcount().astype("float64")
    csum = g.cumsum() - amt
    csq = (amt**2).groupby(key, sort=False).cumsum() - amt**2
    mean = np.where(count > 0, csum / count.replace(0, np.nan), np.nan)
    var = np.where(count > 1, (csq - count * mean**2) / (count - 1).replace(0, np.nan), np.nan)
    std = np.where(
        np.isnan(var),
        np.nan,
        np.where(var <= VAR_EPS * (mean**2 + 1.0), 0.0, np.sqrt(np.clip(var, 0, None))),
    )
    out["card_prior_count"] = count
    out["card_amt_mean"] = mean
    out["card_amt_std"] = std
    out["amt_to_card_mean"], out["amt_zscore"] = _ratio_and_z(amt.values, mean, std)
    out["secs_since_prev"] = t.groupby(key, sort=False).diff().astype("float64")

    # Time windows [t - w, t): rolling over a time index per card, closed="left".
    ts = pd.to_datetime(t, unit="s")
    frame = pd.DataFrame({"key": key.values, "amt": amt.values, "pos": np.arange(len(d))}, index=ts)
    for name, seconds in cfg["windows"].items():
        roll = frame.groupby("key", sort=False)["amt"].rolling(f"{int(seconds)}s", closed="left")
        cnt = roll.count()
        tot = roll.sum()
        pos = frame.groupby("key", sort=False)["pos"].rolling(1).max()  # aligns rows to positions
        cnt_arr = np.empty(len(d))
        sum_arr = np.empty(len(d))
        cnt_arr[pos.values.astype(int)] = cnt.values
        sum_arr[pos.values.astype(int)] = tot.values
        out[f"cnt_{name}"] = np.nan_to_num(cnt_arr, nan=0.0)
        out[f"sum_{name}"] = np.nan_to_num(sum_arr, nan=0.0)

    dev = d[cfg["device_column"]] if cfg["device_column"] in d else pd.Series(np.nan, index=d.index)
    addr = d[cfg["addr_column"]] if cfg["addr_column"] in d else pd.Series(np.nan, index=d.index)
    email = d[cfg["email_column"]] if cfg["email_column"] in d else pd.Series(np.nan, index=d.index)
    new_dev = _first_seen(key, dev)
    out["new_device"] = new_dev
    out["new_addr"] = _first_seen(key, addr)
    out["new_email"] = _first_seen(key, email)
    # distinct devices seen before this row = cumulative first-sightings, excluding this row
    first_dev = new_dev.fillna(0.0)
    out["card_n_devices"] = first_dev.groupby(key, sort=False).cumsum() - first_dev

    dev_key = dev.astype("object")
    has_dev = dev.notna()
    dev_count = dev_key.groupby(dev_key, sort=False).cumcount().astype("float64")
    new_card_on_dev = (~pd.DataFrame({"d": dev_key, "k": key}).duplicated(["d", "k"])).astype(float)
    dev_cards = new_card_on_dev.groupby(dev_key, sort=False).cumsum() - new_card_on_dev
    out["device_prior_count"] = dev_count.where(has_dev, np.nan)
    out["device_n_cards"] = dev_cards.where(has_dev, np.nan)

    out.insert(0, "card_key", key)
    return out.loc[df.index]


@dataclass
class _CardState:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    last_t: int | None = None
    recent: deque = field(default_factory=deque)  # (t, amount) within the longest window
    devices: set = field(default_factory=set)
    addrs: set = field(default_factory=set)
    emails: set = field(default_factory=set)


@dataclass
class _DeviceState:
    count: int = 0
    cards: set = field(default_factory=set)


class OnlineFeatureState:
    """Incremental version of `build_features` for the streaming path.

    Call `update(tx)` once per transaction in time order; it returns the features for that
    transaction (computed before the transaction is added to the state).
    """

    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg
        self.windows = dict(cfg["windows"])
        self.max_window = max(self.windows.values())
        self.cards: dict[str, _CardState] = {}
        self.devices: dict[str, _DeviceState] = {}

    @staticmethod
    def _missing(v: Any) -> bool:
        return v is None or (isinstance(v, float) and math.isnan(v))

    def features(self, tx: Mapping[str, Any]) -> dict[str, Any]:
        """Features for `tx` given the current state, without updating it."""
        cfg = self.cfg
        key = row_key(tx, cfg["card_key_columns"])
        t = int(tx[cfg["time_column"]])
        amt = float(tx[cfg["amount_column"]])
        st = self.cards.get(key, _CardState())
        n = st.count
        mean = st.total / n if n > 0 else math.nan
        if n > 1:
            var = (st.total_sq - n * mean * mean) / (n - 1)
            std = _std(var, mean)
        else:
            std = math.nan
        f: dict[str, Any] = {
            "card_key": key,
            "card_prior_count": float(n),
            "card_amt_mean": mean,
            "card_amt_std": std,
            "amt_to_card_mean": amt / mean if n > 0 and mean > 0 else math.nan,
            "amt_zscore": (amt - mean) / std if n > 1 and std > 0 else math.nan,
            "secs_since_prev": float(t - st.last_t) if st.last_t is not None else math.nan,
        }
        for name, w in self.windows.items():
            in_win = [a for (tt, a) in st.recent if t - w <= tt < t]
            f[f"cnt_{name}"] = float(len(in_win))
            f[f"sum_{name}"] = float(sum(in_win))
        dev = tx.get(cfg["device_column"])
        addr = tx.get(cfg["addr_column"])
        email = tx.get(cfg["email_column"])
        f["new_device"] = math.nan if self._missing(dev) else float(dev not in st.devices)
        f["new_addr"] = math.nan if self._missing(addr) else float(addr not in st.addrs)
        f["new_email"] = math.nan if self._missing(email) else float(email not in st.emails)
        f["card_n_devices"] = float(len(st.devices))
        if self._missing(dev):
            f["device_prior_count"] = math.nan
            f["device_n_cards"] = math.nan
        else:
            ds = self.devices.get(dev, _DeviceState())
            f["device_prior_count"] = float(ds.count)
            f["device_n_cards"] = float(len(ds.cards))
        return f

    def update(self, tx: Mapping[str, Any]) -> dict[str, Any]:
        """Return features for `tx`, then add it to the state."""
        f = self.features(tx)
        cfg = self.cfg
        key = f["card_key"]
        t = int(tx[cfg["time_column"]])
        amt = float(tx[cfg["amount_column"]])
        st = self.cards.setdefault(key, _CardState())
        st.count += 1
        st.total += amt
        st.total_sq += amt * amt
        st.last_t = t
        st.recent.append((t, amt))
        while st.recent and st.recent[0][0] < t - self.max_window:
            st.recent.popleft()
        for col, bucket in (
            (cfg["device_column"], st.devices),
            (cfg["addr_column"], st.addrs),
            (cfg["email_column"], st.emails),
        ):
            v = tx.get(col)
            if not self._missing(v):
                bucket.add(v)
        dev = tx.get(cfg["device_column"])
        if not self._missing(dev):
            ds = self.devices.setdefault(dev, _DeviceState())
            ds.count += 1
            ds.cards.add(key)
        return f


def online_features(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Run `OnlineFeatureState` over `df` in time order; returned in `df`'s index order."""
    state = OnlineFeatureState(cfg)
    d = df.sort_values(cfg["time_column"], kind="stable")
    rows = [state.update(_clean(r)) for r in d.to_dict("records")]
    return pd.DataFrame(rows, index=d.index).loc[df.index]


def _clean(row: dict) -> dict:
    """NaN-like values (pd.NA, numpy NaN) -> float NaN so `_missing` sees them."""
    return {k: (math.nan if _is_nan(v) else v) for k, v in row.items()}


def _is_nan(v: Any) -> bool:
    return v is pd.NA or (isinstance(v, (float, np.floating)) and v != v)


def parity(offline: pd.DataFrame, online: pd.DataFrame, names: Sequence[str]) -> dict[str, float]:
    """Max absolute difference per feature (NaNs must match position-wise; else inf)."""
    report = {}
    for n in names:
        a = offline[n].to_numpy(dtype="float64")
        b = online[n].to_numpy(dtype="float64")
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            report[n] = math.inf
            continue
        m = ~np.isnan(a)
        diff = np.abs(a[m] - b[m])
        scale = np.maximum(1.0, np.abs(a[m]))
        report[n] = float((diff / scale).max()) if m.any() else 0.0
    return report
