"""Behavioural features per client, card and device: offline (vectorised) and online (incremental).

Three keys (strings; missing parts -> "NA"):
- client_key = card1 | addr1 | d1n, where d1n = day - D1 (D1 = days since the card was first
  used, so d1n is constant for one cardholder). This is the usual IEEE-CIS client proxy.
- card_key   = card1 | ... | card6 (card product; shared by many clients)
- device_key = DeviceInfo | id_30 | id_31 | id_33 (OS, browser, screen); NaN if all missing

Definitions, for a transaction at time t using EARLIER transactions in processing order
(sorted by time, stable tie-break):
- client_prior_count, client_amt_mean, client_amt_std; amt_to_client_mean, amt_zscore
- secs_since_prev: seconds since the client's previous transaction
- cnt_<w>, sum_<w>: count and amount sum of the client's transactions with time in [t - w, t)
- new_device / new_addr / new_email: 1 if never seen on this card_key before, 0 if seen,
  NaN if the value is missing; card_n_devices: distinct devices seen before on the card_key
- device_prior_count, device_n_clients: earlier transactions and distinct clients on the device

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
SECONDS_PER_DAY = 86_400


def _fmt(v: Any) -> str:
    if v is None or v is pd.NA or (isinstance(v, (float, np.floating)) and math.isnan(v)):
        return NA
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _is_missing(v: Any) -> bool:
    return v is None or v is pd.NA or (isinstance(v, (float, np.floating)) and v != v)


def _as_key(s: pd.Series) -> pd.Series:
    """String form of a key column, identical to `_fmt` per value."""
    if pd.api.types.is_float_dtype(s):
        return s.map(_fmt).astype(str)
    return s.astype("object").where(s.notna(), NA).astype(str)


def concat_key(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Concatenate the available columns into one string key (missing -> "NA")."""
    cols = [c for c in columns if c in df]
    if not cols:
        raise ValueError(f"none of {list(columns)} in data")
    key = _as_key(df[cols[0]])
    for c in cols[1:]:
        key = key + "|" + _as_key(df[c])
    return key


def row_key(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    """Key for one transaction dict; matches `concat_key` on a frame."""
    return "|".join(_fmt(row[c]) for c in columns if c in row)


def _d1n(t: Any, d1: Any) -> float:
    if _is_missing(d1):
        return math.nan
    return float(int(t) // SECONDS_PER_DAY - float(d1))


def add_keys(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Return a frame with client_key, card_key and device_key (NaN when no device fields)."""
    d = df.copy()
    t = d[cfg["time_column"]].astype("int64")
    d1 = d[cfg["d1_column"]] if cfg["d1_column"] in d else pd.Series(np.nan, index=d.index)
    d["d1n"] = (t // SECONDS_PER_DAY).astype("float64") - d1.astype("float64")
    keys = pd.DataFrame(index=d.index)
    keys["client_key"] = concat_key(d, cfg["client_key_columns"])
    keys["card_key"] = concat_key(d, cfg["card_key_columns"])
    dev_cols = [c for c in cfg["device_key_columns"] if c in d]
    if dev_cols:
        has_dev = d[dev_cols].notna().any(axis=1)
        keys["device_key"] = concat_key(d, dev_cols).where(has_dev, np.nan)
    else:
        keys["device_key"] = np.nan
    return keys


def row_keys(tx: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[str, str, str | None]:
    """(client_key, card_key, device_key) for one transaction; matches `add_keys`."""
    row = dict(tx)
    row["d1n"] = _d1n(tx[cfg["time_column"]], tx.get(cfg["d1_column"]))
    dev_cols = [c for c in cfg["device_key_columns"] if c in row]
    has_dev = any(not _is_missing(row[c]) for c in dev_cols)
    return (
        row_key(row, cfg["client_key_columns"]),
        row_key(row, cfg["card_key_columns"]),
        row_key(row, dev_cols) if has_dev else None,
    )


def feature_names(windows: Mapping[str, int]) -> list[str]:
    names = [
        "client_prior_count",
        "client_amt_mean",
        "client_amt_std",
        "amt_to_client_mean",
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
        "device_n_clients",
    ]
    return names


# Variances below this (relative to mean^2 + 1) count as zero in both implementations, so
# float rounding on constant-amount clients can't make one side NaN and the other finite.
VAR_EPS = 1e-9


def _std(var: float, mean: float) -> float:
    return 0.0 if var <= VAR_EPS * (mean * mean + 1.0) else math.sqrt(var)


def _first_seen(key: pd.Series, value: pd.Series) -> pd.Series:
    """1.0 if `value` was not seen earlier for this key, 0.0 if seen, NaN if value missing."""
    first = ~pd.DataFrame({"k": key, "v": value}).duplicated(["k", "v"])
    return first.astype(float).where(value.notna(), np.nan)


def _col(d: pd.DataFrame, name: str) -> pd.Series:
    return d[name] if name in d else pd.Series(np.nan, index=d.index)


def build_features(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Offline keys + features for every row of `df` (any order), in `df`'s index order."""
    t_col, a_col = cfg["time_column"], cfg["amount_column"]
    d = df.loc[df.sort_values(t_col, kind="stable").index]
    keys = add_keys(d, cfg)
    client, card, device = keys["client_key"], keys["card_key"], keys["device_key"]
    t = d[t_col].astype("int64")
    amt = d[a_col].astype("float64")
    out = keys.copy()

    # Client amount profile from earlier transactions.
    count = amt.groupby(client, sort=False).cumcount().astype("float64")
    csum = amt.groupby(client, sort=False).cumsum() - amt
    csq = (amt**2).groupby(client, sort=False).cumsum() - amt**2
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(count > 0, csum / count, np.nan)
        var = np.where(count > 1, (csq - count * mean**2) / (count - 1), np.nan)
        std = np.where(
            np.isnan(var),
            np.nan,
            np.where(var <= VAR_EPS * (mean**2 + 1.0), 0.0, np.sqrt(np.clip(var, 0, None))),
        )
        out["client_prior_count"] = count
        out["client_amt_mean"] = mean
        out["client_amt_std"] = std
        out["amt_to_client_mean"] = np.where(mean > 0, amt / mean, np.nan)
        out["amt_zscore"] = np.where(std > 0, (amt - mean) / std, np.nan)
    out["secs_since_prev"] = t.groupby(client, sort=False).diff().astype("float64")

    # Velocity windows [t - w, t) per client: time-based rolling with closed="left".
    frame = pd.DataFrame(
        {"key": client.values, "amt": amt.values, "pos": np.arange(len(d))},
        index=pd.to_datetime(t.values, unit="s"),
    )
    grouped = frame.groupby("key", sort=False)
    pos = grouped["pos"].rolling(1).max().to_numpy().astype(int)  # same row order as below
    for name, seconds in cfg["windows"].items():
        roll = grouped["amt"].rolling(f"{int(seconds)}s", closed="left")
        cnt = np.zeros(len(d))
        tot = np.zeros(len(d))
        cnt[pos] = np.nan_to_num(roll.count().to_numpy(), nan=0.0)
        tot[pos] = np.nan_to_num(roll.sum().to_numpy(), nan=0.0)
        out[f"cnt_{name}"] = cnt
        out[f"sum_{name}"] = tot

    # Changes on the card product: device, address, email never seen before.
    new_dev = _first_seen(card, device)
    out["new_device"] = new_dev
    out["new_addr"] = _first_seen(card, _col(d, cfg["addr_column"]))
    out["new_email"] = _first_seen(card, _col(d, cfg["email_column"]))
    first_dev = new_dev.fillna(0.0)
    out["card_n_devices"] = first_dev.groupby(card, sort=False).cumsum() - first_dev

    # Device fingerprint: earlier transactions and distinct clients.
    has_dev = device.notna()
    dev_count = device.groupby(device, sort=False).cumcount().astype("float64")
    new_client = (~pd.DataFrame({"d": device, "c": client}).duplicated(["d", "c"])).astype(float)
    dev_clients = new_client.groupby(device, sort=False).cumsum() - new_client
    out["device_prior_count"] = dev_count.where(has_dev, np.nan)
    out["device_n_clients"] = dev_clients.where(has_dev, np.nan)
    return out.loc[df.index]


@dataclass
class _ClientState:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    last_t: int | None = None
    recent: deque = field(default_factory=deque)  # (t, amount) within the longest window


@dataclass
class _CardState:
    devices: set = field(default_factory=set)
    addrs: set = field(default_factory=set)
    emails: set = field(default_factory=set)


@dataclass
class _DeviceState:
    count: int = 0
    clients: set = field(default_factory=set)


class OnlineFeatureState:
    """Incremental version of `build_features` for the streaming path.

    Call `update(tx)` once per transaction in time order; it returns the keys and features for
    that transaction (computed before the transaction is added to the state).
    """

    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg
        self.windows = dict(cfg["windows"])
        self.max_window = max(self.windows.values())
        self.clients: dict[str, _ClientState] = {}
        self.cards: dict[str, _CardState] = {}
        self.devices: dict[str, _DeviceState] = {}

    def _values(self, tx: Mapping[str, Any]) -> tuple[Any, Any]:
        return tx.get(self.cfg["addr_column"]), tx.get(self.cfg["email_column"])

    def features(self, tx: Mapping[str, Any]) -> dict[str, Any]:
        """Keys and features for `tx` given the current state, without updating it."""
        cfg = self.cfg
        client, card, device = row_keys(tx, cfg)
        t = int(tx[cfg["time_column"]])
        amt = float(tx[cfg["amount_column"]])
        cs = self.clients.get(client, _ClientState())
        n = cs.count
        mean = cs.total / n if n > 0 else math.nan
        std = _std((cs.total_sq - n * mean * mean) / (n - 1), mean) if n > 1 else math.nan
        f: dict[str, Any] = {
            "client_key": client,
            "card_key": card,
            "device_key": device if device is not None else math.nan,
            "client_prior_count": float(n),
            "client_amt_mean": mean,
            "client_amt_std": std,
            "amt_to_client_mean": amt / mean if n > 0 and mean > 0 else math.nan,
            "amt_zscore": (amt - mean) / std if n > 1 and std > 0 else math.nan,
            "secs_since_prev": float(t - cs.last_t) if cs.last_t is not None else math.nan,
        }
        for name, w in self.windows.items():
            in_win = [a for (tt, a) in cs.recent if t - w <= tt < t]
            f[f"cnt_{name}"] = float(len(in_win))
            f[f"sum_{name}"] = float(sum(in_win))
        ks = self.cards.get(card, _CardState())
        addr, email = self._values(tx)
        f["new_device"] = math.nan if device is None else float(device not in ks.devices)
        f["new_addr"] = math.nan if _is_missing(addr) else float(addr not in ks.addrs)
        f["new_email"] = math.nan if _is_missing(email) else float(email not in ks.emails)
        f["card_n_devices"] = float(len(ks.devices))
        if device is None:
            f["device_prior_count"] = math.nan
            f["device_n_clients"] = math.nan
        else:
            ds = self.devices.get(device, _DeviceState())
            f["device_prior_count"] = float(ds.count)
            f["device_n_clients"] = float(len(ds.clients))
        return f

    def update(self, tx: Mapping[str, Any]) -> dict[str, Any]:
        """Return keys and features for `tx`, then add it to the state."""
        f = self.features(tx)
        t = int(tx[self.cfg["time_column"]])
        amt = float(tx[self.cfg["amount_column"]])
        cs = self.clients.setdefault(f["client_key"], _ClientState())
        cs.count += 1
        cs.total += amt
        cs.total_sq += amt * amt
        cs.last_t = t
        cs.recent.append((t, amt))
        while cs.recent and cs.recent[0][0] < t - self.max_window:
            cs.recent.popleft()
        ks = self.cards.setdefault(f["card_key"], _CardState())
        device = f["device_key"] if isinstance(f["device_key"], str) else None
        addr, email = self._values(tx)
        if device is not None:
            ks.devices.add(device)
            ds = self.devices.setdefault(device, _DeviceState())
            ds.count += 1
            ds.clients.add(f["client_key"])
        if not _is_missing(addr):
            ks.addrs.add(addr)
        if not _is_missing(email):
            ks.emails.add(email)
        return f


def online_features(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Run `OnlineFeatureState` over `df` in time order; returned in `df`'s index order."""
    state = OnlineFeatureState(cfg)
    d = df.sort_values(cfg["time_column"], kind="stable")
    rows = [state.update(_clean(r)) for r in d.to_dict("records")]
    return pd.DataFrame(rows, index=d.index).loc[df.index]


def _clean(row: dict) -> dict:
    """NaN-like values (pd.NA, numpy NaN) -> float NaN."""
    return {k: (math.nan if _is_missing(v) else v) for k, v in row.items()}


def parity(offline: pd.DataFrame, online: pd.DataFrame, names: Sequence[str]) -> dict[str, float]:
    """Max relative difference per feature (NaNs must match position-wise; else inf)."""
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


KEY_COLUMNS = ["client_key", "card_key", "device_key"]
