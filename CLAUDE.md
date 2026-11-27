# CLAUDE.md

Shared rules: @AGENTS.md. Roadmap: `PLAN.md`. Brief: `docs/PROJECT_BRIEF.md`.

## Project in one line
**AIML-01 – Real-Time Financial Fraud Detection Platform** (Nexvia internship, AI/ML track), on
branch `AIML-01` of the shared repo `Outsideorbit29/aiml04-perception`. Checked out as a git
worktree at `D:\Project_ai_aiml01`; AIML-04 (`main`) is at `D:\Project_ai`. Deadline **2026-12-15**.

## Claude-specific notes
- Keep the laptop load low (the user's explicit request): training and evals run on Kaggle,
  tests in GitHub Actions, metrics in MLflow.
- Every shell command must say where it runs (Windows, WSL, Kaggle, Docker).
- Kaggle CLI: `C:\Users\anish\AppData\Roaming\Python\Python314\Scripts\kaggle.exe` (not on PATH).
- Run a smoke-mode Kaggle job before any full run.
- Never submit on the portal for the user.

## Progress snapshot
_Last updated: 2026-11-27_ — F0–F4 done. F5 code done and CI-green (46 tests): warm-start state,
row scorer (Booster), Feast/memory store, case store, FastAPI scorer, consumer, producer, Dockerfile,
docker-compose.yml. Serving bundle ready in Kaggle output `anish2906/aiml01-serving-bundle` (30 MB:
model, blend w=0.5 thr 0.30995, client_risk, history, replay). Next (laptop, ask user first): start
Docker Desktop, `KAGGLE=<cli> bash scripts/fetch_serving_bundle.sh`, `docker compose up -d --build
kafka redis scorer consumer`, replay with `docker compose run --rm producer --rate 50 --limit 5000`,
record latency/throughput (serving_bundle/data/consumer_stats.json), then `docker compose down`.
MLflow: DagsHub `Outsideorbit29/aiml04-perception.mlflow`, experiment `aiml01-fraud`; jobs call
`fraud.tracking.configure()` and must attach dataset `anish2906/aiml04-secrets` (file `dagshub_token`).

Useful facts:
- Local tests: `py -3.14 -m pytest -q` works (pandas, pytest, yaml installed) except the parquet
  end-to-end test (no pyarrow on the laptop); CI runs everything.
- Kaggle runner clones `--branch AIML-01`, so the branch must be pushed before `kaggle kernels push`.
