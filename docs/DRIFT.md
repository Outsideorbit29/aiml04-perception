# Drift monitoring and retraining (F6)

Code: `fraud/drift.py` (metrics + decision), `scripts/drift_report.py` (offline weekly report with
Evidently), `services/monitor/` (live monitor + retrain trigger), `scripts/drift_retrain.py`
(trigger CLI). Config: `configs/drift.yaml`. Kaggle job: `kaggle/drift` (`anish2906/aiml01-drift`).

## What is monitored
| Signal | How | Where |
|---|---|---|
| Input drift | PSI per feature, bins fixed on the reference (10 quantile bins + missing; top 20 categories + other + missing). A feature has drifted at PSI ≥ 0.2. **Drift share** = drifted / monitored features. | offline: every model input vs train; live: `TransactionAmt`, `ProductCD` from the case store vs train |
| Evidently | `DataDriftPreset` on the model inputs + `ValueDrift` on the score (prediction drift) and `isFraud` (target drift), Evidently 0.7.23. HTML + JSON reports. | offline job |
| Prediction drift | PSI of the XGBoost probability vs its distribution on val (train scores are optimistic). | offline + live (`p_xgb` in the case store) |
| Live performance | Precision of analyst-reviewed alerts (`feedback_label` in the case store); estimated false-alert rate = alerts·(1−precision) / (n − alerts·precision) against the 1% budget; alert rate vs the val alert rate. Offline windows have every label, so PR-AUC, recall and the exact FPR are added. | offline + live |

Columns that grow with calendar time by construction (`TransactionDT`, the `D*` day counters,
`client_prior_count`, and the cumulative counters `card_n_devices`, `device_prior_count`, `device_n_clients`) are reported but excluded from the drift share: they drift in every window
and say nothing about customer behaviour.

## Retraining rule (`configs/drift.yaml` → `retrain`)
Retrain when any check fires on a window of at least 500 transactions:

| Check | Fires when |
|---|---|
| `data_drift` | drift share ≥ 0.3 |
| `score_drift` | score PSI ≥ 0.2 |
| `precision` | precision of reviewed alerts < 0.35 (needs ≥ 50 reviewed; val precision ≈ 0.56) |
| `alert_budget` | estimated false-alert rate ≥ 1.5 × the 1% budget (needs ≥ 50 reviewed) |
| `alert_rate` | alert rate ≥ 2 × the val alert rate |
| `pr_auc` | PR-AUC < 0.40 (offline only; val 0.545) |

The trigger (`services/monitor/trigger.py`) runs `kaggle kernels push -p kaggle/train` (the F3
training job), at most once per 24 h. It is a **dry run by default**: it prints and records the
command; `--execute` (CLI) or `execute_retrain: true` (monitor) actually launches it.

## Offline results (Kaggle `anish2906/aiml01-drift`)
RESULTS_PLACEHOLDER

## Live monitor
`python -m services.monitor.main` (in the services image; `--once` for a single check). Every
30 s it reads the latest 2,000 scored transactions from `cases_db`, computes the numbers above and
writes `/app/data/monitor_status.json` (current window, PSI per feature, performance, decision,
trigger action, last 200 decisions). Reference: `/app/data/drift_reference.json` from the Kaggle
job if present (train inputs + val scores + val alert rate); otherwise `history.parquet` for the
inputs and the first 2,000 live scores as the score baseline. `MONITOR_<KEY>` env vars override
the `monitor` section (e.g. `MONITOR_FEEDBACK_SOURCE=true_label`).

## Demo (laptop, Docker; ask before starting the stack)
```bash
# Windows, Git Bash: copy the reference from the Kaggle job into the bundle (optional)
kaggle kernels output anish2906/aiml01-drift -p serving_bundle/data --file-pattern 'drift_reference\.json$'
docker compose up -d --build kafka redis scorer consumer monitor
# 2,000 normal transactions, then amounts x3 (PSI 0.81 on synthetic lognormal; x1.5 gives only 0.10)
docker compose run --rm producer --rate 50 --limit 5000 --drift amount_scale=3,shift_after=2000
docker compose logs -f monitor        # "retrain True ['drift share 0.50 >= 0.3']" + "[dry-run] ..."
python -m scripts.drift_retrain --status serving_bundle/data/monitor_status.json   # dry run
docker compose down
```
`ProductCD=C` (e.g. `--drift shift_after=2000,ProductCD=C`) also fires and raises the scores.
Without analyst reviews the precision checks are skipped; set `MONITOR_FEEDBACK_SOURCE=true_label`
on the monitor to simulate an analyst who reviews every alert.

## Limitations
- Retraining reuses the F2 splits: the dataset has no newer data. In production the data-prep and
  feature jobs would first be re-run on the latest window.
- Live drift covers what the case store keeps (amount, product, scores); the full input set is
  covered offline.
