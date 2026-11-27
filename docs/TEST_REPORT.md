# Test report – AIML-01

Date: 2026-11-27. Branch `AIML-01`, commit `f1104e7` (the docs commits on `aiml01-f8-docs` add no
code or tests). All numbers below are copied from the CI log or from the docs named in each row.

## 1. Unit tests and lint

### GitHub Actions (authoritative)
Workflow `.github/workflows/ci.yml`: Ubuntu 22.04, Python 3.10, `pip install -r requirements-dev.txt`,
then `ruff check .` and `pytest -q`.

| Run | Commit | Result |
|---|---|---|
| [45955895921](https://github.com/Outsideorbit29/aiml04-perception/actions/runs/45955895921) (2026-11-27, "Tidy line continuation in bundle fetch script") | `f1104e7` | ruff: **All checks passed!** pytest: **93 passed, 2 skipped, 3 warnings in 3.22s** |

The four runs before it on `AIML-01` (45955791920, 45955591677, 45955556316, 45955312805) also
completed with `success`.

The 2 skips are expected: CI installs only `requirements-dev.txt`, which has no `streamlit` or
`evidently`, so `tests/test_dashboard_app.py` (module-level `importorskip("streamlit")`) and
`test_evidently_report_runs` skip. The CI log does not print skip reasons (`pytest -q` without
`-rs`); this attribution comes from the skip conditions in the test files, which are the only ones
that can fire with the CI packages.

### Laptop (Windows, `py -3.14 -m pytest -q`, CPU, 2.1 s)
**83 passed, 11 skipped.** The laptop's Python 3.14 has pandas, pytest and PyYAML but no
`pyarrow`, `xgboost`, `streamlit` or `evidently`, so these skip:

| Skipped (laptop) | Missing package |
|---|---|
| `test_dashboard_app.py` (whole module) | streamlit |
| `test_scorer_api.py` (whole module, 2 tests) | xgboost |
| `test_build_features_end_to_end`, `test_eda.py::test_prep_data_end_to_end`, `test_dashboard_data.py::test_load_rings_roundtrip`, `test_drift_report.py` (2 tests), `test_drift_services.py::test_monitor_end_to_end_with_case_store` | pyarrow |
| `test_serving.py` (2 tests) | xgboost |
| `test_drift_report.py::test_evidently_report_runs` | evidently |

Laptop and CI totals differ because module-level skips count as one item locally (92 tests
collected locally vs 95 in CI). CI is the reference.

## 2. Test files

95 test cases in 14 files, all on small fixtures: `tests/fixtures/` holds 300 synthetic rows with
the IEEE-CIS schema (`scripts/make_fixtures.py`); the competition data is not in the repo.

| File | Tests | Covers |
|---|---:|---|
| `test_data.py` | 7 | F1: identity join, data-dir discovery, time split is ordered, complete and rejects bad fractions, time columns, split summary, dtype downcast keeps values |
| `test_eda.py` | 3 | F1: EDA summary, fraud rate with missing categories, `prep_data` end to end on the fixtures |
| `test_features.py` | 11 | F2: client / card / device keys, `day − D1` grouping, a hand-checked example, missing device → NaN, **offline vs online parity on random data with same-second ties** (3 seeds), zero std, eviction of old transactions, warm start equals full replay (2 seeds) |
| `test_build_features.py` | 3 | F2: Feast feature list equals `feature_names()`, AUC ranking helper, `build_features` script end to end |
| `test_metrics_model.py` | 7 | F3: average precision on known values and ties (matches scikit-learn), ROC-AUC, threshold respects the FPR budget, evaluation keys, `FeatureSpec` round trip and median imputation |
| `test_tracking.py` | 3 | MLflow: token from the attached Kaggle dataset, remote config, file-store fallback |
| `test_graph.py` | 4 | F4: entity encoding skips NaN, rings link only through specific devices, ring min size and empty input, blend endpoints and weight choice |
| `test_serving.py` | 6 | F5: `transform_row` equals batch transform (2 cases), `Scorer` equals batch XGBoost, blend with a known client risk, memory store round trip, case store flow (insert, status, feedback) |
| `test_scorer_api.py` | 2 | F5: `/health`, `/model`; `/score` returns **409** with no features and with another transaction's features, then 200 with matching features and `alert == score > threshold` |
| `test_drift.py` | 11 | F6: PSI (identical = 0, shift large), numeric and categorical reference bins with missing / other, drift share with exclusions, weekly windows and merge, live performance from feedback or full labels, retrain decision checks, drift spec parsing, monitor window stable vs drifted |
| `test_drift_report.py` | 4 | F6: weekly report without and with a model, Evidently summary parsing, Evidently report runs |
| `test_drift_services.py` | 6 | F6: retrain command, trigger dry run / execute / 24 h cooldown, trigger failures reported, `drift_retrain` CLI on summary and status files, monitor end to end on a case store, monitor waits for a missing DB |
| `test_dashboard_data.py` | 13 | F7: empty DB, overview, alert queue sorted with score breakdown and rings, feedback updates status and precision, evaluation uses true labels, ring summary and graph, timeline, config env override, required config keys, consumer stats, rings round trip, total alert rate vs expected |
| `test_dashboard_app.py` | 1 | F7: Streamlit `AppTest` renders the app and confirms an alert |
| `test_k8s_manifests.py` | 14 | F7.2: expected objects, kustomization, namespaces, selectors and ports, images and resources, fits a small Minikube, PVC volumes, mounts / commands / Kafka env / memory limits consistent with `docker-compose.yml`, service DNS matches `configs/serving.yaml`, ConfigMap keys are known config keys, dashboard port |

## 3. System-level checks

These ran on Kaggle or on the laptop's Docker stack and are recorded in the docs; they are not part
of CI.

| Check | Where | Result | Source |
|---|---|---|---|
| Offline vs online feature parity | Kaggle `aiml01-features` v4 | max relative difference **1.6e-11** on 30,000 real transactions; keys 100% equal | `docs/FEATURES.md` |
| Feast point-in-time read vs builder | Kaggle `aiml01-features` v4 | **100%** match on 5,000 sampled transactions | `docs/FEATURES.md` |
| Feast push → online read | Kaggle `aiml01-features` v4 | **100%** match on 20 clients | `docs/FEATURES.md` |
| Online feature cost | Kaggle | ~0.31 ms per transaction (pure Python) | `docs/FEATURES.md` |
| F3 determinism | Kaggle `aiml01-train` v2 and v3 | same numbers on both runs | `docs/EXPERIMENTS.md` |
| Consumer warm start | laptop Docker | 501,959 transactions in ~13 s; unit test checks it is bit-identical to replay | `docs/SERVING.md` |
| Load test, steady 50 tx/s (batched consumer) | laptop Docker, 4 CPU / 6 GB | 5,000 messages, keeps up at 50.0 tx/s; end-to-end **p50 83 / p95 128 / p99 161 ms** | `docs/SERVING.md` |
| Stage latency at 50 tx/s | laptop Docker | feature update p95 0.06 ms; Feast push p95 1.91 ms; scorer HTTP **p50 3.81 / p95 5.29 / p99 7.88 ms** | `docs/SERVING.md` |
| Burst | laptop Docker | 10,000 messages unthrottled, **242 tx/s** sustained | `docs/SERVING.md` |
| One-by-one consumer (before batching) | laptop Docker | max ~43 tx/s (~23 ms per transaction) | `docs/SERVING.md` |
| Live detection vs offline | laptop Docker, first 5,000 test-period tx | 147 alerts; 64 of 131 frauds caught (49%); 83 false alerts of 4,869 legit (1.7%). Offline test: recall 49.8% at ~1.3% FPR | `docs/SERVING.md` |
| Memory in the burst | laptop Docker | consumer 563 MB, scorer 226 MB, Kafka 349 MB, Redis 8 MB | `docs/SERVING.md` |
| Offline drift job | Kaggle `aiml01-drift` v2 (132 s, CPU) | 9 weekly windows: PSI drift share 0 (max feature PSI 0.19), Evidently drift share 0.08–0.22, score PSI ≤ 0.006, FPR 0.88% → 1.36%, weekly PR-AUC 0.42–0.69; no retrain trigger | `docs/DRIFT.md` |
| Retrain trigger dry run | laptop CLI | last window w09 → `retrain: false`, action `none` | `docs/DRIFT.md` |
| Live monitor on injected drift | laptop, no Docker | synthetic 4,000-row case store, `amount_scale=3,shift_after=2000`: PSI `TransactionAmt` 2.37, drift share 0.50 → retrain True, dry-run command printed | `docs/DRIFT.md` |

## 4. Model metrics (test split, threshold set on val for a 1% FPR budget)

| Model | Test PR-AUC | Test ROC-AUC | Recall | Precision | Test FPR | Source |
|---|---:|---:|---:|---:|---:|---|
| Logistic regression | 0.357 | 0.860 | 35.2% | 44.3% | 1.59% | `docs/MODEL.md` |
| XGBoost, raw fields | 0.463 | 0.874 | 43.5% | 53.6% | 1.36% | `docs/MODEL.md` |
| XGBoost + F2 features | 0.457 | 0.869 | 42.6% | 55.8% | 1.22% | `docs/MODEL.md` |
| XGBoost + F2 + SMOTE | 0.491 | 0.890 | 45.4% | 55.8% | 1.30% | `docs/MODEL.md` |
| GraphSAGE alone | 0.506 | 0.897 | 46.7% | 55.7% | 1.34% | `docs/GRAPH.md` |
| Blend, full graph (w = 0.5) | 0.545 | 0.909 | 50.4% | 58.3% | 1.30% | `docs/GRAPH.md` |
| **Live blend (served)** | **0.525** | 0.905 | **49.8%** | 58.0% | 1.30% | `docs/GRAPH.md` |

## 5. Not yet tested

| Item | Status |
|---|---|
| Live monitor + drift demo on the Docker stack | passed 2026-11-27: TransactionAmt PSI 1.47–1.63 after the shift, dry-run retrain printed; one false trigger on a 962-row warm-up window, fixed by `min_window` 1000 (`docs/DRIFT.md`) |
| Minikube deployment (F7.2 manifests) | passed 2026-11-27: all pods Ready, 5,000 replayed, 147 alerts (= Compose), e2e p95 99 ms; Kafka probe changed to tcpSocket (`docs/DEPLOY.md`) |
| Screenshots of the running dashboard, scorer docs and drift monitor | TBD (main session, `docs/screenshots/`) |
| SMOTE gain vs imputation gain | open question in `docs/MODEL.md` |
