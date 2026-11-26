"""Synthetic transactions with the IEEE-CIS schema (core columns only).

Used for unit tests and the committed fixtures. The real competition data is not redistributed
in this repo, so fixtures are generated rather than sampled from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCTS = ["W", "C", "R", "H", "S"]
CARD_BRANDS = ["visa", "mastercard", "american express", "discover"]
EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", None]
DEVICES = ["Windows", "iOS Device", "MacOS", "SM-G950F", "Trident/7.0", None]


def make_synthetic(
    n: int = 200, fraud_rate: float = 0.035, seed: int = 0, identity_share: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, identity) frames shaped like the IEEE-CIS train files."""
    rng = np.random.default_rng(seed)
    ids = np.arange(2_987_000, 2_987_000 + n)
    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    if n and is_fraud.sum() == 0:
        is_fraud[rng.integers(n)] = 1  # keep at least one positive for tests
    # ~180 days of seconds, starting at the competition's first value (86400).
    dt = np.sort(rng.integers(86_400, 86_400 + 180 * 86_400, size=n))
    amount = np.round(np.exp(rng.normal(4.0, 1.0, size=n)) * (1 + is_fraud), 2)
    tx = pd.DataFrame(
        {
            "TransactionID": ids,
            "isFraud": is_fraud,
            "TransactionDT": dt,
            "TransactionAmt": amount,
            "ProductCD": rng.choice(PRODUCTS, size=n),
            "card1": rng.integers(1000, 18400, size=n),
            "card2": rng.choice([np.nan, 111.0, 321.0, 555.0], size=n),
            "card4": rng.choice(CARD_BRANDS, size=n),
            "card6": rng.choice(["debit", "credit"], size=n),
            "addr1": rng.choice([np.nan, 204.0, 299.0, 325.0], size=n),
            "P_emaildomain": rng.choice(np.array(EMAILS, dtype=object), size=n),
            "C1": rng.integers(0, 5, size=n).astype(float),
            "D1": rng.choice([np.nan, 0.0, 14.0, 120.0], size=n),
        }
    )
    has_id = rng.random(n) < identity_share
    ident = pd.DataFrame(
        {
            "TransactionID": ids[has_id],
            "id_01": rng.choice([0.0, -5.0, -20.0], size=int(has_id.sum())),
            "DeviceType": rng.choice(["desktop", "mobile"], size=int(has_id.sum())),
            "DeviceInfo": rng.choice(np.array(DEVICES, dtype=object), size=int(has_id.sum())),
        }
    )
    return tx, ident
