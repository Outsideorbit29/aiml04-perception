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
_Last updated: 2026-11-26_ — F0.1, F0.2, F1.3 done. F1.1/F1.2 code ready (`scripts/prep_data.py`,
`kaggle/data_prep/`, MODE=smoke). Blocked on: user accepting IEEE-CIS competition rules and
approving the push of branch `AIML-01`. Next: smoke job, then full, then F2 features.

Useful facts:
- Local tests: `py -3.14 -m pytest -q` works (pandas, pytest, yaml installed) except the parquet
  end-to-end test (no pyarrow on the laptop); CI runs everything.
- Kaggle runner clones `--branch AIML-01`, so the branch must be pushed before `kaggle kernels push`.
