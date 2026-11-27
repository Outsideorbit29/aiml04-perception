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
Version 2 (full, commit `fdad117`, 132 s on CPU), 2026-11-27. Reference: all 413,378 train rows
(Evidently: a 50k sample); current: val + test (177,162 rows, days 120–182) in weekly windows.
Score = F3 XGBoost (`xgb_smote`) at its val 1%-FPR threshold 0.2149; val alert rate 2.54%.
MLflow run `f6-drift-full` (`5bb057144c4940d39dfe4d76aad62c71`) in experiment `aiml01-fraud`:
per-window metrics, `drift_summary.json`, `drift_reference.json` and the HTML reports. Smoke run:
version 1 (60k train rows vs the first 40k val rows).

| Window (days) | Rows | Fraud rate | PSI drift share | Evidently drift share | Score PSI | PR-AUC | Precision | Recall | FPR | Retrain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all (120–182) | 177,162 | 3.46% | 0.00 | 0.19 | 0.001 | 0.516 | 0.588 | 0.456 | 1.15% | no |
| w01 (120–126) | 20,798 | 3.43% | 0.00 | 0.11 | 0.002 | 0.695 | 0.709 | 0.601 | 0.88% | no |
| w02 (127–133) | 19,258 | 3.78% | 0.00 | 0.08 | 0.002 | 0.567 | 0.627 | 0.455 | 1.06% | no |
| w03 (134–140) | 18,572 | 3.29% | 0.00 | 0.14 | 0.001 | 0.518 | 0.639 | 0.440 | 0.85% | no |
| w04 (141–147) | 18,549 | 3.44% | 0.00 | 0.14 | 0.000 | 0.421 | 0.560 | 0.365 | 1.02% | no |
| w05 (148–154) | 21,443 | 3.08% | 0.00 | 0.11 | 0.000 | 0.494 | 0.537 | 0.427 | 1.17% | no |
| w06 (155–161) | 19,811 | 2.85% | 0.00 | 0.11 | 0.002 | 0.446 | 0.495 | 0.446 | 1.34% | no |
| w07 (162–168) | 21,038 | 3.47% | 0.00 | 0.19 | 0.004 | 0.454 | 0.530 | 0.412 | 1.31% | no |
| w08 (169–175) | 19,941 | 3.68% | 0.00 | 0.22 | 0.004 | 0.538 | 0.579 | 0.473 | 1.31% | no |
| w09 (176–182) | 17,752 | 4.21% | 0.00 | 0.11 | 0.006 | 0.498 | 0.601 | 0.469 | 1.36% | no |

Findings:
1. **No input feature crosses PSI 0.2.** The largest PSIs (0.10–0.19) are the history counters
   (`card_n_devices`, `client_prior_count`), `D15`, `secs_since_prev`, `C9` and `weekday` in w01.
   `TransactionDT` itself has PSI 8.3, which is why the time-trend columns are excluded.
   Evidently (Wasserstein / Jensen–Shannon, default thresholds) flags 8–22% of the inputs per
   week, always under the 30% preset share; it flags mostly the same counter features.
2. **Prediction drift is negligible** (score PSI ≤ 0.006), and the label drifts only mildly (fraud
   rate 2.9–4.2%; the 4.2% in the last week matches the upward drift in `docs/EDA.md`).
3. **Performance drifts, but stays within the trigger thresholds.** The false-alert rate climbs from
   0.88% to 1.31–1.36% over the last four weeks (1.36× the 1% budget against a trigger at 1.5×), and
   weekly PR-AUC falls from 0.69 to 0.42–0.54 (trigger < 0.40). With `max_fpr_budget_ratio: 1.25`
   the trigger would fire on w06–w09. The thresholds were kept as designed, not tuned on this
   result.
4. So, on this data, a fixed alert threshold slowly overspends the alert budget long before the
   inputs drift. Re-picking the threshold on recent feedback is the cheaper fix than retraining.
   The live monitor watches for this with the `alert_budget` and `alert_rate` checks.

Retrain trigger dry run on this summary:
`python -m scripts.drift_retrain --summary f6_drift/drift_summary.json` → last window w09,
`retrain: false`, action `none`; on a window with drift it prints
`[dry-run] would retrain: kaggle kernels push -p kaggle/train (...)`.

## Live monitor
`python -m services.monitor.main` (in the services image; `--once` for a single check). Every
30 s it reads the latest 2,000 scored transactions from `cases_db`, computes the numbers above and
writes `/app/data/monitor_status.json` (current window, PSI per feature, performance, decision,
trigger action, last 200 decisions). Reference: `/app/data/drift_reference.json` from the Kaggle
job if present (train inputs + val scores + val alert rate); otherwise `history.parquet` for the
inputs and the first 2,000 live scores as the score baseline. `MONITOR_<KEY>` env vars override
the `monitor` section (e.g. `MONITOR_FEEDBACK_SOURCE=true_label`).

Checked without Docker (2026-11-27): one `Monitor.check()` on a 4,000-row synthetic case store
drawn from the Kaggle reference, with `amount_scale=3,shift_after=2000` applied, gives
`PSI TransactionAmt 2.37, ProductCD 0.00 | score PSI 0.008 | retrain True ['drift share 0.50 >= 0.3']`
and `[dry-run] would retrain: kaggle kernels push -p kaggle/train`.

## Demo (laptop, Docker; ask before starting the stack)
```bash
# Windows, Git Bash: scripts/fetch_serving_bundle.sh also fetches the drift reference
# (Kaggle writes it as f6_drift/drift_reference.json; it must end up in serving_bundle/data/)
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

### Live run (Docker Desktop, 2026-11-27)
Full stack (kafka, redis, scorer, consumer, monitor, dashboard) with the Kaggle drift reference,
fresh case store, `--rate 50 --limit 5000 --drift amount_scale=3,shift_after=2000`. Monitor log:

| Window (last ≤2,000 cases) | PSI TransactionAmt | PSI ProductCD | Score PSI | Retrain |
|---|---|---|---|---|
| 962 cases (all before drift) | 0.054 | 0.231 | 0.054 | **True** (false trigger, see below) |
| 2,000 of 2,466 (mostly before drift) | 0.158 | 0.131 | 0.011 | False |
| 2,000 of 3,973 (after drift) | **1.472** | 0.164 | 0.039 | **True**, `[dry-run] would retrain: kaggle kernels push -p kaggle/train` |
| 2,000 of 5,000 (after drift) | **1.630** | 0.096 | 0.028 | **True** (dry run) |

- The amount drift is detected within one monitor interval once ~1,000 drifted rows are in the
  window. The model score barely moves (score PSI ≤ 0.04): amount alone does not change the
  scores much, so only the data-drift check fires.
- The first window was a **false trigger**: 962 rows covering a few hours of the replay have a
  different product mix from the whole reference. `retrain.min_window` was raised from 500 to
  1,000 after this run, so that window would now give no decision.
- Consumer for the run: 5,000 processed, 153 alerts (64 of 131 frauds caught), wall 55.9 tx/s,
  capacity 137 tx/s with the monitor and dashboard also running; scorer HTTP p50 3.5 ms / p95
  5.1 ms; end-to-end p50 85 ms, but p95 7.2 s (max 11.4 s). The tail was not traced (the
  case store does not keep per-transaction end-to-end times); a likely cause is a backlog while
  the consumer, monitor and dashboard compete for CPU. The steady-state F5 run without them gave
  p95 128 ms (`docs/SERVING.md`).

## Limitations
- Retraining reuses the F2 splits: the dataset has no newer data. In production the data-prep and
  feature jobs would first be re-run on the latest window.
- Live drift covers what the case store keeps (amount, product, scores); the full input set is
  covered offline.
