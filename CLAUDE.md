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
_Last updated: 2026-11-27_ — F0–F5 done, F7.1/F7.2 done (merged from aiml01-f7-dashboard; Minikube
not yet run). F5 load test: 50 tx/s steady e2e p95 128 ms, burst 242 tx/s (`docs/SERVING.md`). Stack:
`docker compose up -d --build kafka redis scorer consumer dashboard` (dashboard :8501), replay with
`docker compose run --rm producer --rate 50 --limit 5000`, then `docker compose down`. Serving bundle in
gitignored `serving_bundle/` (from Kaggle `anish2906/aiml01-serving-bundle`). F6 drift being built on
branch `aiml01-f6-drift` by an agent (merge + QA when done). Known fix: dashboard "over budget" compares
total alert rate with the 1% FPR budget (expected total alert rate ~2.7%). Then F8 deliverables.
MLflow: DagsHub `Outsideorbit29/aiml04-perception.mlflow`, experiment `aiml01-fraud`; jobs call
`fraud.tracking.configure()` and must attach dataset `anish2906/aiml04-secrets` (file `dagshub_token`).

Useful facts:
- Local tests: `py -3.14 -m pytest -q` works (pandas, pytest, yaml installed) except the parquet
  end-to-end test (no pyarrow on the laptop); CI runs everything.
- Kaggle runner clones `--branch AIML-01`, so the branch must be pushed before `kaggle kernels push`.
