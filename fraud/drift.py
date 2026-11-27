"""F6 drift metrics and the retraining decision (numpy/pandas only; used offline and at serving).

- Population Stability Index (PSI) per feature, with bins fixed on the reference window:
  numeric columns use reference quantiles (+ a missing bin), categoricals the top reference
  categories (+ "other" and "missing"). References are plain JSON so the monitor can load them
  without the training data.
- Share of drifted features, score (prediction) drift, and live performance from analyst
  feedback labels: precision of reviewed alerts and the estimated false-alert rate vs the 1%
  budget.
- `decide_retrain` turns those numbers into a yes/no with readable reasons, using the thresholds
  in configs/drift.yaml.
- `parse_drift_spec` / `apply_drift` perturb replayed transactions for the drift demo.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

MISSING = "__missing__"
OTHER = "__other__"


# ---------------------------------------------------------------- reference bins and PSI
def numeric_reference(values: Iterable[Any], bins: int = 10) -> dict[str, Any]:
    """Quantile bin edges of the reference and its proportions (last entry = missing share)."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")
    finite = x[np.isfinite(x)]
    if len(finite):
        qs = np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1], method="lower")
        edges = np.unique(qs).tolist()
    else:
        edges = []
    ref = {"kind": "numeric", "edges": edges, "props": []}
    ref["props"] = numeric_props(x, ref).tolist()
    return ref


def numeric_props(values: Iterable[Any], ref: Mapping[str, Any]) -> np.ndarray:
    """Proportions of `values` in the reference bins: len(edges)+1 value bins, then missing."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")
    edges = np.asarray(ref["edges"], dtype="float64")
    n_bins = len(edges) + 1
    counts = np.zeros(n_bins + 1, dtype="float64")
    finite = np.isfinite(x)
    if finite.any():
        idx = np.searchsorted(edges, x[finite], side="right")
        counts[:n_bins] = np.bincount(idx, minlength=n_bins)
    counts[n_bins] = (~finite).sum()
    total = counts.sum()
    return counts / total if total else counts


def _as_category(values: Iterable[Any]) -> pd.Series:
    s = pd.Series(values, dtype="object")
    return s.map(
        lambda v: MISSING if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
    )


def categorical_reference(values: Iterable[Any], max_categories: int = 20) -> dict[str, Any]:
    """Top reference categories; everything else goes to OTHER, missing values to MISSING."""
    s = _as_category(values)
    counts = s[s != MISSING].value_counts()
    cats = sorted(counts.index[:max_categories].tolist())
    ref = {"kind": "categorical", "categories": cats, "props": []}
    ref["props"] = categorical_props(s, ref).tolist()
    return ref


def categorical_props(values: Iterable[Any], ref: Mapping[str, Any]) -> np.ndarray:
    """Proportions over [*categories, OTHER, MISSING]."""
    s = _as_category(values)
    labels = [*ref["categories"], OTHER, MISSING]
    known = set(ref["categories"])
    s = s.map(lambda v: v if v in known or v == MISSING else OTHER)
    counts = s.value_counts().reindex(labels, fill_value=0).to_numpy(dtype="float64")
    total = counts.sum()
    return counts / total if total else counts


def psi(ref_props: Sequence[float], cur_props: Sequence[float], eps: float = 1e-4) -> float:
    """Population Stability Index: sum (cur - ref) * ln(cur / ref), proportions floored at eps.

    Rule of thumb: < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 large shift.
    """
    r = np.clip(np.asarray(ref_props, dtype="float64"), eps, None)
    c = np.clip(np.asarray(cur_props, dtype="float64"), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def build_reference(
    df: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
    bins: int = 10,
    max_categories: int = 20,
) -> dict[str, dict[str, Any]]:
    """Reference bins/proportions for every monitored column present in `df`."""
    ref: dict[str, dict[str, Any]] = {}
    for c in numeric:
        if c in df:
            ref[c] = numeric_reference(df[c], bins)
    for c in categorical:
        if c in df:
            ref[c] = categorical_reference(df[c], max_categories)
    for c, r in ref.items():
        r["n"] = int(len(df))
        r["column"] = c
    return ref


def column_psi(values: Iterable[Any], ref: Mapping[str, Any], eps: float = 1e-4) -> float:
    if ref["kind"] == "numeric":
        cur = numeric_props(values, ref)
    else:
        cur = categorical_props(values, ref)
    return psi(ref["props"], cur, eps)


def drift_table(
    current: pd.DataFrame,
    reference: Mapping[str, Mapping[str, Any]],
    threshold: float = 0.2,
    eps: float = 1e-4,
    columns: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    """PSI of each reference column found in `current`, the drifted ones and their share.

    Columns in `exclude` are reported in `psi` but do not count towards the drift share.
    """
    cols = [c for c in (columns or list(reference)) if c in reference and c in current]
    values = {c: column_psi(current[c], reference[c], eps) for c in cols}
    counted = [c for c in cols if c not in set(exclude)]
    drifted = sorted([c for c in counted if values[c] >= threshold], key=lambda c: -values[c])
    return {
        "n": int(len(current)),
        "psi": {c: round(v, 5) for c, v in sorted(values.items(), key=lambda kv: -kv[1])},
        "drifted": drifted,
        "n_features": len(counted),
        "drift_share": len(drifted) / len(counted) if counted else 0.0,
    }


def time_windows(
    seconds: Iterable[Any], window_days: int = 7, min_rows: int = 1
) -> list[dict[str, Any]]:
    """Consecutive windows of `window_days` days from the first timestamp (TransactionDT).

    Returns [{"name", "day_from", "day_to", "index"}] where index is a positional row array.
    A window with fewer than `min_rows` rows is merged into the previous one.
    """
    t = pd.to_numeric(pd.Series(seconds), errors="coerce").to_numpy(dtype="float64")
    if not len(t):
        return []
    day = np.floor(t / 86_400)
    first = np.nanmin(day)
    wid = ((day - first) // window_days).astype("int64")
    windows: list[dict[str, Any]] = []
    for w in np.unique(wid):
        idx = np.flatnonzero(wid == w)
        d_from, d_to = int(np.min(day[idx])), int(np.max(day[idx]))
        if windows and len(idx) < min_rows:
            prev = windows[-1]
            prev["index"] = np.concatenate([prev["index"], idx])
            prev["day_to"] = d_to
            continue
        windows.append({"day_from": d_from, "day_to": d_to, "index": idx})
    for i, w in enumerate(windows, 1):
        w["name"] = f"w{i:02d}_d{w['day_from']}-{w['day_to']}"
    return windows


# ---------------------------------------------------------------- live performance
def live_performance(
    alert: Iterable[Any],
    label: Iterable[Any],
    alert_budget: float = 0.01,
    expected_alert_rate: float | None = None,
) -> dict[str, Any]:
    """Performance of a window from feedback labels (NaN/None = alert not reviewed yet).

    precision = confirmed / reviewed alerts. The false-alert rate (FPR) is estimated as
    alerts * (1 - precision) / (n - alerts * precision), i.e. false alerts over the estimated
    legit transactions, and compared with the alert budget (configs/train.yaml max_fpr).
    If every row is labelled (offline evaluation), recall and the exact FPR are added.
    """
    a = pd.to_numeric(pd.Series(alert), errors="coerce").fillna(0).to_numpy() > 0
    y = pd.to_numeric(pd.Series(label), errors="coerce").to_numpy(dtype="float64")
    n = int(len(a))
    alerts = int(a.sum())
    reviewed_mask = a & ~np.isnan(y)
    reviewed = int(reviewed_mask.sum())
    confirmed = int((y[reviewed_mask] == 1).sum())
    out: dict[str, Any] = {
        "n": n,
        "alerts": alerts,
        "alert_rate": alerts / n if n else None,
        "reviewed": reviewed,
        "confirmed": confirmed,
        "precision": confirmed / reviewed if reviewed else None,
        "fpr": None,
        "fpr_budget_ratio": None,
        "alert_rate_ratio": None,
    }
    if reviewed:
        p = out["precision"]
        legit_est = max(n - alerts * p, 1.0)
        out["fpr"] = alerts * (1 - p) / legit_est
    if n and not np.isnan(y).any():  # fully labelled: exact numbers
        legit = int((y == 0).sum())
        frauds = n - legit
        out["fpr"] = float((a & (y == 0)).sum() / legit) if legit else 0.0
        out["recall"] = float((a & (y == 1)).sum() / frauds) if frauds else None
        out["fraud_rate"] = frauds / n
    if out["fpr"] is not None and alert_budget:
        out["fpr_budget_ratio"] = out["fpr"] / alert_budget
    if expected_alert_rate and out["alert_rate"] is not None:
        out["alert_rate_ratio"] = out["alert_rate"] / expected_alert_rate
    return out


# ---------------------------------------------------------------- retraining decision
def _check(value: Any, threshold: float, above: bool, enough: bool = True) -> dict[str, Any]:
    if value is None or not enough:
        return {"value": value, "threshold": threshold, "fired": False, "skipped": True}
    fired = value >= threshold if above else value < threshold
    return {"value": round(float(value), 5), "threshold": threshold, "fired": bool(fired)}


def decide_retrain(metrics: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Retrain when any enabled check fires (thresholds: configs/drift.yaml `retrain`).

    `metrics` keys (missing = check skipped): n, drift_share, score_psi, precision, reviewed,
    fpr_budget_ratio, alert_rate_ratio, pr_auc.
    """
    n = int(metrics.get("n") or 0)
    if n < int(cfg["min_window"]):
        return {
            "retrain": False,
            "reasons": [],
            "checks": {},
            "note": f"window too small ({n} < {cfg['min_window']})",
        }
    enough_fb = int(metrics.get("reviewed") or 0) >= int(cfg["min_reviewed"])
    checks = {
        "data_drift": _check(metrics.get("drift_share"), cfg["drift_share"], above=True),
        "score_drift": _check(metrics.get("score_psi"), cfg["score_psi"], above=True),
        "precision": _check(metrics.get("precision"), cfg["min_precision"], False, enough_fb),
        "alert_budget": _check(
            metrics.get("fpr_budget_ratio"), cfg["max_fpr_budget_ratio"], True, enough_fb
        ),
        "alert_rate": _check(metrics.get("alert_rate_ratio"), cfg["max_alert_rate_ratio"], True),
        "pr_auc": _check(metrics.get("pr_auc"), cfg["min_pr_auc"], above=False),
    }
    enabled = set(cfg.get("checks", checks))
    checks = {k: v for k, v in checks.items() if k in enabled}
    text = {
        "data_drift": "drift share {value:.2f} >= {threshold}",
        "score_drift": "score PSI {value:.3f} >= {threshold}",
        "precision": "alert precision {value:.3f} < {threshold}",
        "alert_budget": "false-alert rate {value:.2f}x the budget >= {threshold}x",
        "alert_rate": "alert rate {value:.2f}x the expected >= {threshold}x",
        "pr_auc": "PR-AUC {value:.3f} < {threshold}",
    }
    reasons = [text[k].format(**v) for k, v in checks.items() if v["fired"]]
    return {"retrain": bool(reasons), "reasons": reasons, "checks": checks}


# ---------------------------------------------------------------- live window (monitor)
def hour_of(event_time: Iterable[Any]) -> pd.Series:
    """Hour of day from TransactionDT seconds (the anchor date starts at midnight)."""
    t = pd.to_numeric(pd.Series(event_time), errors="coerce")
    return (t // 3600) % 24


def cases_to_frame(cases: pd.DataFrame, live_columns: Mapping[str, str]) -> pd.DataFrame:
    """Case-store rows -> frame with the reference column names (`hour` derived if mapped)."""
    out = pd.DataFrame(index=cases.index)
    for ref_col, case_col in live_columns.items():
        if case_col == "hour":
            out[ref_col] = hour_of(cases["event_time"])
        elif case_col in cases:
            out[ref_col] = cases[case_col]
    return out


def monitor_window(
    cases: pd.DataFrame,
    reference: Mapping[str, Mapping[str, Any]],
    cfg: Mapping[str, Any],
    expected_alert_rate: float | None = None,
) -> dict[str, Any]:
    """Drift + live performance + decision for one window of case-store rows."""
    mon = cfg["monitor"]
    frame = cases_to_frame(cases, mon["live_columns"])
    score_col = mon["score_column"]
    feature_cols = [c for c in frame.columns if c != score_col]
    table = drift_table(
        frame, reference, cfg["psi"]["feature_threshold"], cfg["psi"]["eps"], feature_cols
    )
    score_psi = None
    if score_col in reference and score_col in frame and frame[score_col].notna().any():
        score_psi = column_psi(frame[score_col], reference[score_col], cfg["psi"]["eps"])
    label_col = "true_label" if mon["feedback_source"] == "true_label" else "feedback_label"
    labels = cases[label_col] if label_col in cases else pd.Series(np.nan, index=cases.index)
    labels = labels.where(cases["alert"].astype(bool))  # only alerts carry feedback
    perf = live_performance(
        cases["alert"], labels, cfg["retrain"]["alert_budget"], expected_alert_rate
    )
    perf.pop("recall", None)  # feedback covers alerts only; recall is unknown live
    perf.pop("fraud_rate", None)
    metrics = {**perf, "drift_share": table["drift_share"], "score_psi": score_psi}
    return {
        "window": {
            "rows": int(len(cases)),
            "event_time_min": float(cases["event_time"].min()) if len(cases) else None,
            "event_time_max": float(cases["event_time"].max()) if len(cases) else None,
        },
        "drift": table,
        "score_psi": None if score_psi is None else round(score_psi, 5),
        "performance": perf,
        "decision": decide_retrain(metrics, cfg["retrain"]),
    }


# ---------------------------------------------------------------- drift injection (demo)
def parse_drift_spec(spec: str | None) -> dict[str, Any]:
    """'amount_scale=1.5,shift_after=2000,ProductCD=C' -> dict.

    Known keys: shift_after (int, messages before the drift starts), amount_scale and
    amount_shift (TransactionAmt * scale + shift). Any other key overrides that field with the
    given string value (e.g. ProductCD=C, DeviceType=mobile).
    """
    out: dict[str, Any] = {"shift_after": 0, "amount_scale": 1.0, "amount_shift": 0.0, "set": {}}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"drift option '{part}' must be key=value")
        key, value = (s.strip() for s in part.split("=", 1))
        if key == "shift_after":
            out[key] = int(value)
        elif key in ("amount_scale", "amount_shift"):
            out[key] = float(value)
        else:
            out["set"][key] = value
    return out


def apply_drift(
    msg: dict[str, Any], index: int, drift: Mapping[str, Any], amount_column: str = "TransactionAmt"
) -> dict[str, Any]:
    """Perturb message number `index` (1-based) once index > shift_after; returns a new dict."""
    if index <= drift["shift_after"]:
        return msg
    out = dict(msg)
    amt = out.get(amount_column)
    if amt is not None and (drift["amount_scale"] != 1.0 or drift["amount_shift"] != 0.0):
        out[amount_column] = float(amt) * drift["amount_scale"] + drift["amount_shift"]
    out.update(drift["set"])
    return out
