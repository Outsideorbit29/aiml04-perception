# AGENTS.md

Rules for any AI coding agent working on this branch.

## Context
- Project: **AIML-01 Real-Time Financial Fraud Detection Platform**. Brief: `docs/PROJECT_BRIEF.md`.
  Roadmap: `PLAN.md`. Do not add features outside them; propose changes instead.
- This is the `AIML-01` branch of the shared capstone repo. The AIML-04 project is on `main`.
  Never merge between the two.
- Deadline **2026-12-15**; target submission 2026-12-13. Prefer the simplest approach that meets the brief.

## Environment: cloud first, low laptop load
| Where | What runs there |
|---|---|
| Kaggle notebook | Data prep, EDA, feature build, XGBoost and GNN training, evaluation |
| GitHub Actions | `ruff`, `pytest` on every push |
| MLflow (hosted) | Runs, metrics, model registry |
| Laptop (Windows 11 + WSL2 / Docker) | Serving stack only: Kafka, Redis, FastAPI, dashboard, Minikube; short sessions |

- No training or full-dataset jobs on the laptop. Only `tests/fixtures/` data locally.
- System Python is 3.14; use Python 3.10 (WSL or Docker images).

## Code rules
- Python 3.10, type hints on public functions, `ruff` + `black` (line length 100).
- Core logic in the `fraud/` package, free of Kafka/Kubernetes so it can be unit-tested.
  Services in `services/` are thin wrappers.
- Tunables in `configs/*.yaml`; no magic numbers in code.
- Every module gets a test in `tests/` using small fixtures.
- Never commit `data/`, `mlruns/`, model files (`*.json` boosters, `*.pt`, `*.pkl`), `.env`, `kaggle.json`.
- Git: small commits with imperative messages; feature branches `aiml01-f<id>-<slug>` merged into `AIML-01`.
