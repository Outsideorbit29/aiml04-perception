# Graph model and fraud rings (F4)

Code: `fraud/graph.py`, `fraud/gnn.py`, `scripts/train_gnn.py`. Config: `configs/gnn.yaml`.
Kaggle job `anish2906/aiml01-gnn` v2 (full, T4 GPU), 2026-11-27, commit `77a74c6`. Logged in
MLflow (`aiml01-fraud`, run `gnn_blend`).

## Graph
| Node type | Count | Edges to transactions |
|---|---:|---:|
| transaction | 590,540 | — |
| client (`card1 \| addr1 \| day − D1`) | 217,850 | 590,540 |
| card product (`card1…card6`) | 14,893 | 590,540 |
| device fingerprint (`DeviceInfo \| OS \| browser \| screen`) | 9,704 | 140,589 |

Transactions carry the same 46 model inputs as XGBoost (signed-log, standardised with train
statistics). Entity nodes start from a learned per-type vector. A 2-layer heterogeneous
**GraphSAGE** (hidden 64, mean aggregation, `HeteroConv`) means a transaction sees its
siblings through every shared client, card and device. It was trained full-batch on the
T4 with class-weighted BCE on train labels only, with early stopping on val PR-AUC: best at
epoch 375 of 400, 104 s.

The graph is **transductive**: val and test transactions are in the graph (their features, not their
labels) while training. That is how the offline GNN and blend numbers below were measured. The
**live-servable blend** avoids that: see the next section.

## Results (test, threshold set on val for a 1% false-positive budget)

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Recall | Precision | Test FPR |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost + SMOTE (F3) | 0.545 | 0.491 | 0.890 | 45.4% | 55.8% | 1.30% |
| GraphSAGE alone | 0.555 | 0.506 | 0.897 | 46.7% | 55.7% | 1.34% |
| Blend, full graph (w = 0.5) | 0.593 | **0.545** | 0.909 | 50.4% | 58.3% | 1.30% |
| **Live blend: XGBoost + client GNN risk** (w = 0.5) | 0.615 | **0.525** | 0.905 | **49.8%** | 58.0% | 1.30% |

Blend = `w·logit(p_xgb) + (1 − w)·logit(p_gnn)`, with `w` picked on val from 0.0–1.0. Both
blends peak at w = 0.5, so the two models carry complementary signal.

**Live blend (used by the F5 scorer).** For each client, its *GNN risk* is the mean GNN
probability over that client's transactions in **earlier periods only**: train for val and
train+val for test. At serving time this is a key-value lookup (`client_risk.parquet`, 180k clients).
Clients never seen before get the XGBoost score alone. That gives +0.034 test PR-AUC and +4.4
points of recall at the same alert rate, for a lookup that costs microseconds.

## Fraud rings
Clients linked by a **specific** shared device fingerprint (one used by 2–20 clients; generic
fingerprints such as "Windows + Chrome" are ignored). Connected groups of ≥ 3 clients are
candidate rings.

- **1,001 rings with 15,059 clients**; median 5 clients, 75% have ≤ 8. One component grows to
  7,522 clients through chains of shared mid-popularity fingerprints. That's a limitation of the
  coarse fingerprint, and the dashboard lists rings by size so it can be told apart.
- **Fraud rate inside rings vs outside:** train 9.2% vs 2.8%; val 9.8% vs 2.8%; **test 9.9% vs 2.5%
  (about 3.9×)**. The ring structure is found without labels, so it carries over to the future period.
- **54 rings were more than 50% fraud in train.** The top ones are 3–6 clients on one device with
  every transaction fraudulent: the textbook pattern of one fraudster cycling through stolen cards.

Outputs (Kaggle output of `anish2906/aiml01-gnn`): `rings.parquet` (client → ring), `ring_stats.json`,
`client_risk.parquet` (serving lookup, train+val), `scores.parquet` (per-transaction scores),
`gnn_state.pt`.

## Limitations and next steps
- Val PR-AUC was still rising at epoch 400; more epochs or a learning-rate schedule may add a little.
- Device fingerprints are coarse (no true device IDs in IEEE-CIS), which limits ring precision.
- The live blend uses client risk only; adding device and card risk lookups is a small extension.
