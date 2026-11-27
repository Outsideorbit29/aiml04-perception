# PLAN – AIML-01 Real-Time Fraud Detection

Start 2026-11-26, deadline **2026-12-15**, target submission **2026-12-13** (two days of buffer).
Brief: `docs/PROJECT_BRIEF.md`. **(S)** = stretch, skip if late.

## Key decisions
- **Dataset: IEEE-CIS Fraud Detection** (590k tx, 3.5% fraud; has card, address, email and device
  fields, so it supports both the velocity features and the account/device/card graph). Kaggle
  competition data: the user must accept the competition rules once on kaggle.com.
- **Cloud first, as in AIML-04.** EDA, feature build, XGBoost and GNN training run on Kaggle.
  The laptop runs only the serving stack (Kafka, Redis, FastAPI, dashboard) in short sessions.
- **Feast** with a file offline store (parquet) and a Redis online store; the same feature
  definitions feed training (offline) and the FastAPI service (online).
- **MLflow:** hosted tracking (DagsHub free tier) so Kaggle jobs and the laptop log to one place.
  Fallback: file store in the Kaggle output, downloaded as a small `mlruns/` folder.
- **Serving:** Kafka (single KRaft broker) → FastAPI scorer → Postgres/SQLite case table →
  Streamlit dashboard. `docker compose` for development, Minikube manifests for the final demo.
- **GNN:** heterogeneous graph (transaction, card, device, email domain) with GraphSAGE in PyTorch
  Geometric, trained on Kaggle; serve the blended score from precomputed node embeddings so the
  live path stays in milliseconds.

## F0 Setup (Nov 26–27)
- [x] F0.1 Orphan branch `AIML-01` in the shared repo; brief, plan, agent rules
- [x] F0.2 Python package skeleton (`fraud/`), `pyproject.toml`, CI (ruff + pytest)
- [x] F0.3 Accept IEEE-CIS rules on Kaggle; MLflow on DagsHub (`configs/mlflow.yaml`, token in private Kaggle dataset `aiml04-secrets`)
- [ ] F0.4 Docker Desktop (or WSL Docker) and Minikube available locally (check, don't install yet)

## F1 Data + EDA (Nov 27–28)
- [x] F1.1 Kaggle job: load and join transaction + identity tables, time-based train/val/test split
      (code done: `scripts/prep_data.py`, `kaggle/data_prep/`; full run 2026-11-27, see `docs/EXPERIMENTS.md`)
- [x] F1.2 EDA: imbalance, amount distribution, missingness, fraud by time/device/card
      (`fraud/eda.py`, run inside the F1.1 job; write-up in `docs/EDA.md`)
- [x] F1.3 Small fixtures in `tests/fixtures/`: 300 **synthetic** rows with the IEEE-CIS schema
      (`scripts/make_fixtures.py`); the competition data is not redistributed in the repo

## F2 Features + Feast (Nov 29 – Dec 1)
- [x] F2.1 Features: per-client velocity (1 h / 24 h / 7 d counts and sums), amount / card mean,
      new device, new address, email-domain risk
- [x] F2.2 Feast repo: entity client, feature views, offline parquet, Redis online
- [x] F2.3 Point-in-time correct training set from Feast; parity test offline vs online (`docs/FEATURES.md`)

## F3 XGBoost + MLflow (Dec 1–3)
- [ ] F3.1 Baseline LR, then XGBoost with `scale_pos_weight` (SMOTE as a comparison)
- [ ] F3.2 Metrics: PR-AUC, ROC-AUC, recall at 1% FPR, threshold choice; all runs in MLflow
- [ ] F3.3 Register the best model in MLflow

## F4 Graph + GNN (Dec 3–5)
- [ ] F4.1 Build the heterogeneous graph; connected components as candidate fraud rings
- [ ] F4.2 GraphSAGE node classifier on Kaggle; PR-AUC vs XGBoost
- [ ] F4.3 Blend (weighted average or stacked LR) on validation; report the lift
- [ ] F4.4 **(S)** Elliptic dataset as a second GNN benchmark

## F5 Streaming + scoring service (Dec 5–7)
- [ ] F5.1 Kafka producer replaying the test split at a configurable rate
- [ ] F5.2 FastAPI `/score` plus a Kafka consumer: Feast online lookup → model → score; p95 latency
- [ ] F5.3 Write scored transactions and alerts to the case table
- [ ] F5.4 Load test: throughput and p50/p95/p99 latency recorded

## F6 Drift + retraining (Dec 7–8)
- [ ] F6.1 Evidently data-drift report: reference = train, current = sliding window of live traffic
- [ ] F6.2 Retrain trigger when drift share or PR-AUC on feedback labels crosses a threshold
      (launches the Kaggle training job); inject drift in the replay to demo it

## F7 Dashboard + Kubernetes (Dec 8–10)
- [ ] F7.1 Streamlit dashboard: live alerts, case status (open / confirmed / dismissed),
      analyst feedback labels, fraud-ring view
- [ ] F7.2 Kubernetes manifests (Minikube): Kafka, Redis, scorer, consumer, dashboard
- [ ] F7.3 **(S)** Public dashboard on Render / Streamlit Cloud as the Live Deployment URL

## F8 Deliverables (Dec 10–13)
- [ ] F8.1 README: architecture, setup, run steps, feature breakdown, results
- [ ] F8.2 Architecture diagram + API docs (FastAPI OpenAPI) + data docs
- [ ] F8.3 Test report and metrics table; screenshots in `docs/screenshots/`
- [ ] F8.4 Demo video (3–4 min)
- [ ] F8.5 10-minute presentation
- [ ] F8.6 Individual reflection (**user**, own words)

## Submission (Dec 13)
- [ ] `main` README links to the `AIML-01` branch, and this README links back to AIML-04
- [ ] Repo public, both branches pushed, screenshots present
- [ ] Submit once on the portal (**user**)
