"""EDA summaries (pure pandas) and figures (matplotlib imported lazily, Kaggle only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CATEGORICALS = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]


def fraud_rate_by(df: pd.DataFrame, column: str, target: str = "isFraud", top_n: int = 15) -> list:
    """Count and fraud rate per value of `column` (most frequent `top_n`, missing as "NA")."""
    grouped = (
        df.assign(_key=df[column].astype("object").where(df[column].notna(), "NA"))
        .groupby("_key")[target]
        .agg(count="size", fraud_rate="mean")
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    return [
        {"value": str(k), "count": int(r["count"]), "fraud_rate": float(r["fraud_rate"])}
        for k, r in grouped.iterrows()
    ]


def eda_summary(df: pd.DataFrame, target: str = "isFraud", top_n: int = 15) -> dict[str, Any]:
    """Headline numbers for the EDA write-up: imbalance, amounts, missingness, categories."""
    n = len(df)
    fraud = df[target] == 1
    amount = df["TransactionAmt"]
    missing = df.isna().mean().sort_values(ascending=False)
    summary: dict[str, Any] = {
        "rows": n,
        "columns": int(df.shape[1]),
        "frauds": int(fraud.sum()),
        "fraud_rate": float(fraud.mean()) if n else 0.0,
        "amount": {
            "median_legit": float(amount[~fraud].median()) if (~fraud).any() else None,
            "median_fraud": float(amount[fraud].median()) if fraud.any() else None,
            "p99": float(amount.quantile(0.99)) if n else None,
        },
        "identity_share": float(df["has_identity"].mean()) if "has_identity" in df else None,
        "columns_over_50pct_missing": int((missing > 0.5).sum()),
        "top_missing": {k: float(v) for k, v in missing.head(10).items()},
        "by_category": {
            col: fraud_rate_by(df, col, target, top_n) for col in CATEGORICALS if col in df
        },
    }
    if "hour" in df:
        hourly = df.groupby("hour")[target].mean()
        summary["fraud_rate_by_hour"] = {int(h): float(r) for h, r in hourly.items()}
    if "day" in df:
        daily = df.groupby("day")[target].agg(["size", "mean"])
        summary["days"] = int(len(daily))
        summary["daily_fraud_rate_std"] = float(daily["mean"].std()) if len(daily) > 1 else 0.0
    return summary


def save_figures(
    df: pd.DataFrame, out_dir: str | Path, target: str = "isFraud", bins: int = 50
) -> list[Path]:
    """Write the EDA figures as PNGs and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fraud = df[target] == 1
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = df[target].value_counts().sort_index()
    ax.bar(["legit", "fraud"], counts.values, color=["#4C78A8", "#E45756"])
    ax.set_yscale("log")
    ax.set_title(f"Class balance (fraud {fraud.mean():.2%})")
    paths.append(_save(fig, out_dir / "eda_class_balance.png"))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    edges = np.logspace(0, np.log10(max(df["TransactionAmt"].max(), 10)), bins)
    ax.hist(df.loc[~fraud, "TransactionAmt"], bins=edges, alpha=0.6, density=True, label="legit")
    ax.hist(df.loc[fraud, "TransactionAmt"], bins=edges, alpha=0.6, density=True, label="fraud")
    ax.set_xscale("log")
    ax.set_xlabel("TransactionAmt (USD, log)")
    ax.legend()
    ax.set_title("Amount distribution")
    paths.append(_save(fig, out_dir / "eda_amount.png"))

    if "day" in df:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        daily = df.groupby("day")[target].agg(["size", "mean"])
        ax.plot(daily.index, daily["mean"], lw=1)
        ax.set_xlabel("day")
        ax.set_ylabel("fraud rate")
        ax.set_title("Daily fraud rate")
        paths.append(_save(fig, out_dir / "eda_daily_fraud_rate.png"))

    for col in [c for c in ("ProductCD", "DeviceType", "card6") if c in df]:
        rows = fraud_rate_by(df, col, target)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar([r["value"] for r in rows], [r["fraud_rate"] for r in rows], color="#E45756")
        ax.set_title(f"Fraud rate by {col}")
        ax.tick_params(axis="x", rotation=45)
        paths.append(_save(fig, out_dir / f"eda_rate_{col}.png"))

    missing = df.isna().mean()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(missing.values, bins=20, color="#72B7B2")
    ax.set_xlabel("share missing")
    ax.set_ylabel("columns")
    ax.set_title("Missingness per column")
    paths.append(_save(fig, out_dir / "eda_missingness.png"))
    return paths


def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
