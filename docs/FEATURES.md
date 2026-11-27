# Features (F2)

Code: `fraud/features.py`. Config: `configs/features.yaml`. Feast: `feature_repo/`.
Kaggle job `anish2906/aiml01-features` v4 (full), 2026-11-27.

## Keys
| Key | Definition | Used for | Distinct (590k tx) |
|---|---|---|---:|
| `client_key` | `card1 \| addr1 \| day − D1` (D1 = days since the card was first used, so `day − D1` is constant per cardholder) | amount profile, velocity, Feast entity | 217,850 |
| `card_key` | `card1 \| … \| card6` (card product) | "new device / address / email" flags | 14,893 |
| `device_key` | `DeviceInfo \| id_30 \| id_31 \| id_33` (device, OS, browser, screen) | device reuse across clients | 9,704 |

The first version keyed everything on `card_key` and raw `DeviceInfo`. Those turned out to be
product and OS labels, not people or devices (about 800 transactions per "card" and 1,700 cards
per "device"), and single-feature separation stayed near chance. The client key fixed that.

## Feature definitions
All computed from the same client's (or card's, or device's) **earlier** transactions only.

| Feature | Meaning |
|---|---|
| `client_prior_count`, `client_amt_mean`, `client_amt_std` | history size and amount profile |
| `amt_to_client_mean`, `amt_zscore` | this amount vs the client's usual amount |
| `secs_since_prev` | seconds since the client's previous transaction |
| `cnt_1h/24h/7d`, `sum_1h/24h/7d` | velocity: count and amount in `[t − w, t)` |
| `new_device`, `new_addr`, `new_email` | value never seen on this card before (NaN if missing) |
| `card_n_devices` | distinct devices seen on the card before |
| `device_prior_count`, `device_n_clients` | how busy / shared the device fingerprint is |

## Training–serving consistency
- `build_features` (pandas, vectorised) is used for training; `OnlineFeatureState` (one
  transaction at a time) is used by the stream consumer. They must agree.
- **Parity on 30,000 real transactions: max relative difference 1.6e-11, keys 100% equal.**
  Unit tests check the same on random data with same-second ties.
- Feast (`client_features` view, entity `client_key`): point-in-time `get_historical_features`
  matched the builder on 5,000 sampled transactions (100%); push → `get_online_features`
  round-trip matched on 20 clients.
- Online cost: ~0.31 ms per transaction in pure Python.

## Single-feature signal (train split)
ROC-AUC of each feature alone (0.5 = no signal; below 0.5 = lower values mean fraud).

| Feature | AUC | Legit mean | Fraud mean |
|---|---:|---:|---:|
| `secs_since_prev` | 0.330 | 8.3 days | 2.5 days |
| `cnt_7d` | 0.644 | 1.75 | 3.80 |
| `sum_7d` | 0.634 | $176 | $406 |
| `cnt_24h` | 0.624 | 1.27 | 2.60 |
| `card_n_devices` | 0.595 | 81 | 110 |
| `cnt_1h` | 0.580 | 0.27 | 0.65 |
| `new_device` | 0.424 | 0.50 | 0.35 |

Fraudsters reuse a client identity quickly and in bursts. Amount-ratio features are weak on
their own (AUC ≈ 0.52); the model in F3 combines these with the raw C/D/M fields.

## Limitations
- The device fingerprint is still coarse (median device seen with many clients), so device
  features are weak; F4's graph uses it together with card and email links.
- 40% of transactions are a client's first, so their history features are NaN by design.
