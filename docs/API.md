# API and data reference – AIML-01

Interfaces between the services, taken from the code on branch `AIML-01`:
scorer HTTP API (`services/scorer/app.py`), Kafka messages (`services/producer/main.py`,
`services/consumer/main.py`), case store (`fraud/cases.py`), Feast definitions
(`feature_repo/definitions.py`), the serving bundle (`scripts/export_serving_bundle.py`) and the
configuration files. Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 1. Scorer HTTP API (FastAPI)

Base URL: `http://scorer:8000` inside Compose / Kubernetes, `http://localhost:8000` from the host
(Compose publishes port 8000; on Minikube use `kubectl -n aiml01 port-forward svc/scorer 8000:8000`).
Interactive OpenAPI docs: `GET /docs`; schema: `GET /openapi.json` (FastAPI defaults).
Title "AIML-01 fraud scorer", version 0.1.0.

Start: `uvicorn services.scorer.app:app --host 0.0.0.0 --port 8000` (the Compose `scorer`
command). On start it loads the serving artefacts from `model_dir` and connects to the online
store: Feast by default, or an in-memory store when `ONLINE_STORE=memory` (used by the tests).

### `GET /health`
```json
{"status": "ok", "uptime_s": 812.4, "scored": 5000}
```
| Field | Type | Meaning |
|---|---|---|
| `status` | string | always `"ok"` when the app answers |
| `uptime_s` | float | seconds since start-up |
| `scored` | int | transactions scored since start-up |

### `GET /model`
Model card from `Scorer.info()`.

| Field | Type | Meaning |
|---|---|---|
| `model` | string | experiment name from `threshold.json` (`xgb` if absent) |
| `n_features` | int | number of model input columns in `spec.json` |
| `threshold` | float | alert threshold; the blend threshold when `blend.json` exists (0.310 in the F5 bundle, `docs/EXPERIMENTS.md`) |
| `weight_xgb` | float | XGBoost weight `w` in the blend; 1.0 means XGBoost alone (0.5 in the F5 bundle) |
| `clients_with_risk` | int | clients in the GNN risk lookup (0 if no blend) |

### `POST /score`
Scores one raw transaction. The transaction's F2 features must already be in the online store
(the consumer pushes them first).

**Request body**: the raw transaction as JSON. Three fields are required; any other fields are
accepted and passed to the model (`extra="allow"`).

| Field | Type | Required | Notes |
|---|---|---|---|
| `TransactionID` | int | yes | must match the `TransactionID` stored with the client's features |
| `TransactionDT` | int | yes | seconds from the dataset's reference time |
| `TransactionAmt` | float | yes | |
| `card1`, `addr1`, `D1` | number | for the client key | client key = `card1 \| addr1 \| day − D1` (`configs/features.yaml`) |
| passthrough fields | number / string | no | `configs/features.yaml` `passthrough_numeric` and `passthrough_categorical`; missing values become NaN and are imputed as in training |

```json
{"TransactionID": 3577540, "TransactionDT": 15811200, "TransactionAmt": 49.0,
 "card1": 1234, "addr1": 299.0, "D1": 1.0, "ProductCD": "W", "card6": "debit"}
```
(Illustrative values, not a real record.)

**Response 200** (`ScoreResponse`):

| Field | Type | Meaning |
|---|---|---|
| `TransactionID` | int | echoed |
| `client_key` | string | key used for the feature lookup and the GNN risk lookup |
| `score` | float | final fraud probability: `sigmoid(w·logit(p_xgb) + (1 − w)·logit(gnn_risk))`, or `p_xgb` when the client has no GNN risk |
| `p_xgb` | float | XGBoost probability |
| `gnn_risk` | float or null | client's GNN risk from earlier periods; null for clients not in the lookup |
| `alert` | bool | `score > threshold` |
| `threshold` | float | threshold in use |
| `fetch_ms` | float | time to read the features from the online store |
| `model_ms` | float | feature transform + XGBoost + blend |
| `total_ms` | float | time inside the endpoint |

**Status codes**

| Code | When |
|---|---|
| 200 | scored |
| 409 | no features for this client in the online store, or the stored features carry a different `TransactionID` (`"features for transaction <id> not in the online store yet"`). Guards against scoring with stale features. |
| 422 | request body fails validation (FastAPI default), e.g. a missing `TransactionID`, `TransactionDT` or `TransactionAmt` |

The consumer treats any non-200 response by logging it and skipping that transaction; it does not
retry. `tests/test_scorer_api.py` checks the 409 cases and a 200 response.

Measured latency of the HTTP round trip from the consumer (Feast read + XGBoost + blend), laptop
Docker, 50 tx/s: p50 3.81 ms, p95 5.29 ms, p99 7.88 ms (`docs/SERVING.md`).

## 2. Kafka

Single KRaft broker (`apache/kafka:3.8.0`), `kafka:9092`. The consumer creates both topics at
start-up if they are missing (1 partition, replication factor 1).

| Topic (config key) | Producer | Consumer | Key | Value |
|---|---|---|---|---|
| `transactions` (`topic_transactions`) | `services/producer` | `services/consumer`, group `fraud-consumer` (`consumer_group`), `auto.offset.reset=earliest` | `TransactionID` as string | transaction JSON |
| `scores` (`topic_scores`) | `services/consumer` | none in the stack (for downstream use) | `TransactionID` as string | scorer response JSON |

### `transactions` message
One row of `replay.parquet` (test period, time order) as JSON. NaN and numpy values are converted
by `json_safe`, so missing values are `null`.

| Field group | Fields |
|---|---|
| IDs and time | `TransactionID`, `TransactionDT` |
| Key inputs | `card1`–`card6`, `addr1`, `D1`, `P_emaildomain`, `DeviceInfo`, `id_30`, `id_31`, `id_33` |
| Passthrough numeric | `TransactionAmt`, `hour`, `weekday`, `has_identity`, `dist1`, `C1`, `C2`, `C4`, `C5`, `C6`, `C9`, `C11`, `C13`, `C14`, `D1`, `D2`, `D3`, `D4`, `D10`, `D15` |
| Passthrough categorical | `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `M4`, `M6` |
| Label | `isFraud`: stored in the case store as `true_label` for evaluation; the scorer never reads it |
| Added by the producer | `_produced_at`: unix time at publish, used for end-to-end latency |

Drift injection (F6 demo): `--drift amount_scale=3,shift_after=2000[,ProductCD=C]` changes messages
after the first `shift_after`: `TransactionAmt · amount_scale + amount_shift`, and any other
`key=value` overwrites that field (`fraud/drift.py` `parse_drift_spec`).

### `scores` message
The `/score` 200 response body above (`TransactionID`, `client_key`, `score`, `p_xgb`, `gnn_risk`,
`alert`, `threshold`, `fetch_ms`, `model_ms`, `total_ms`).

### Consumer stats file
Every `consumer_stats_every` (500) transactions the consumer writes `stats_path`
(`/app/data/consumer_stats.json`): `processed`, `alerts`, `wall_tps`, `capacity_tps`,
`batch_size`, `features_names` (number of feature columns), percentiles (`n`, `mean`, `p50`,
`p95`, `p99`, `max`) over the last 5,000 values of `features_ms`, `push_ms`, `http_ms`, `e2e_ms`,
and `cases` (case-store totals, see `CaseStore.stats()` below). The dashboard's Model page shows it.

## 3. Case store (SQLite)

File `cases_db` (`/app/data/cases.db`), WAL mode, 30 s busy timeout. Writers: the consumer (scores)
and the dashboard (analyst decisions). Readers: dashboard, monitor.

Table `scored`, one row per scored transaction:

| Column | Type | Meaning |
|---|---|---|
| `transaction_id` | INTEGER PK | `TransactionID` (insert-or-replace) |
| `event_time` | REAL NOT NULL | `TransactionDT` seconds |
| `scored_at` | REAL NOT NULL | unix time of the insert |
| `client_key`, `card_key`, `device_key` | TEXT | keys from `OnlineFeatureState` (`device_key` null if missing) |
| `amount` | REAL | `TransactionAmt` |
| `product` | TEXT | `ProductCD` |
| `score` | REAL NOT NULL | blended score |
| `p_xgb` | REAL | XGBoost probability |
| `gnn_risk` | REAL | client GNN risk, null if none |
| `alert` | INTEGER NOT NULL | 1 if `score > threshold` |
| `latency_ms` | REAL | consumer feature + push + HTTP time for this transaction |
| `true_label` | INTEGER | `isFraud` from the replay; for evaluation only, hidden from analysts |
| `status` | TEXT | NULL (no alert), `open`, `confirmed`, `dismissed` |
| `feedback_label` | INTEGER | analyst decision: 1 fraud (`confirmed`), 0 legit (`dismissed`) |
| `reviewed_at` | REAL | unix time of the decision |

Indexes: `idx_scored_alert (alert, status)`, `idx_scored_client (client_key)`.

`CaseStore` methods (`fraud/cases.py`):

| Method | Does |
|---|---|
| `insert(rows)` | insert-or-replace a batch in one transaction; alerts get status `open` |
| `set_status(transaction_id, status)` | `status` in `open`/`confirmed`/`dismissed`; sets `feedback_label` (1/0/NULL) and `reviewed_at`; `KeyError` if the row is not an alert, `ValueError` for an unknown status |
| `alerts(status=None, limit=200)` | alerts sorted by score, then newest |
| `stats()` | `n`, `alerts`, `frauds`, `caught`, `mean_latency`, `confirmed`, `dismissed`, `open` |

## 4. Feast feature store

Repo `feature_repo/`, project `aiml01_fraud`, provider `local`, file offline store.

| Object | Name | Definition |
|---|---|---|
| Entity | `client` | join key `client_key` (STRING) = `card1 \| addr1 \| day − D1` |
| FileSource | `client_features_file` | `data/client_features.parquet` (env `FRAUD_CLIENT_FEATURES`), timestamp `event_timestamp` |
| PushSource | `client_features_push` | batch source `client_features_file`; the consumer pushes here |
| FeatureView | `client_features` | entity `client`, TTL 400 days, online; 18 Float32 features + `TransactionID` (Int64) |

Features in the view (definitions in `docs/FEATURES.md`): `client_prior_count`, `client_amt_mean`,
`client_amt_std`, `amt_to_client_mean`, `amt_zscore`, `secs_since_prev`, `cnt_1h`, `sum_1h`,
`cnt_24h`, `sum_24h`, `cnt_7d`, `sum_7d`, `new_device`, `new_addr`, `new_email`,
`card_n_devices`, `device_prior_count`, `device_n_clients`. `tests/test_build_features.py` checks
this list equals `fraud.features.feature_names()`.

`TransactionID` is stored with the features so the scorer can check they belong to the transaction
it is scoring.

| Store config | Online store | Used by |
|---|---|---|
| `feature_store.yaml` | SQLite `data/online_store.db` | development, Kaggle `scripts/feast_check.py` |
| `feature_store.redis.yaml` | Redis `redis:6379` | Compose and Minikube (`feature_store_yaml` in `configs/serving.yaml`) |

Training reads the view with `get_historical_features` (point-in-time join on `client_key` and the
transaction time); serving pushes with the push source and reads with `get_online_features`.
`services/entrypoint.sh` runs `feast -f feature_store.redis.yaml apply` before the scorer and
consumer start (skip with `FEAST_APPLY=0`).

## 5. Serving bundle

Built on Kaggle by `scripts/export_serving_bundle.py` (`anish2906/aiml01-serving-bundle`, 30 MB)
and downloaded with `scripts/fetch_serving_bundle.sh` into `./serving_bundle/` (gitignored).
Compose mounts `models/` read-only at `/app/models` and `data/` at `/app/data`; Minikube uses the
PVC `aiml01-bundle` with the same sub-paths.

| Path | Contents | Used by |
|---|---|---|
| `models/model.json` | XGBoost booster (F3 `xgb_smote`) | scorer |
| `models/spec.json` | column order, category vocabularies, imputation medians (`FeatureSpec`) | scorer |
| `models/threshold.json` | XGBoost threshold and experiment name | scorer |
| `models/blend.json` | `weight_xgb` (0.5) and blend `threshold` (0.310) | scorer |
| `models/client_risk.parquet` | `client_key`, `gnn_risk` (191,596 clients in the bundle) | scorer |
| `data/history.parquet` | train + val transactions (501,959 rows), columns the feature state needs | consumer warm start |
| `data/replay.parquet` | test period (88,581 rows) with the scorer's raw columns and `isFraud` | producer |
| `data/rings.parquet` | client → ring id (F4) | dashboard |
| `data/drift_reference.json` | optional, from `anish2906/aiml01-drift` | monitor |
| `data/cases.db`, `consumer_stats.json`, `monitor_status.json` | written at run time | consumer, dashboard, monitor |

Counts are from `docs/EXPERIMENTS.md` (F5 bundle row).

## 6. Configuration reference

All files are YAML in `configs/` unless noted. Paths in `serving.yaml` are inside the containers.

### `configs/serving.yaml` (producer, consumer, scorer, dashboard)
Each key can be overridden by an environment variable with the same name in upper case
(`services/common.py` `serving_config`); `SERVING_CONFIG` points at another file.

| Key | Default | Meaning |
|---|---|---|
| `kafka_bootstrap` | `kafka:9092` | broker address |
| `topic_transactions` / `topic_scores` | `transactions` / `scores` | topic names |
| `consumer_group` | `fraud-consumer` | Kafka consumer group |
| `model_dir` | `/app/models` | serving artefacts |
| `feature_repo` / `feature_store_yaml` | `/app/feature_repo` / `feature_store.redis.yaml` | Feast repo and config |
| `history_path` | `/app/data/history.parquet` | consumer warm start |
| `replay_path` | `/app/data/replay.parquet` | producer input |
| `cases_db` | `/app/data/cases.db` | case store |
| `stats_path` | `/app/data/consumer_stats.json` | consumer stats |
| `scorer_url` | `http://scorer:8000` | scorer base URL |
| `start_date` | `2017-12-01` | anchor for `TransactionDT` → timestamp |
| `producer_rate` | 50 | tx/s (`--rate`) |
| `producer_limit` | 0 | 0 = whole replay file (`--limit`) |
| `consumer_stats_every` | 500 | write stats every N transactions |
| `consumer_batch` | 200 | max messages per poll |
| `consumer_poll_s` | 0.05 | max wait to fill a batch |
| `http_timeout_s` | 5 | scorer call timeout |

Other environment variables: `ONLINE_STORE` (`feast` or `memory`), `FEATURES_CONFIG`
(default `configs/features.yaml`), `FEAST_APPLY` (`1` or `0`).

### `services/dashboard/dashboard.yaml`
Merged over `serving.yaml`; upper-case env overrides (`CASES_DB`, `RINGS_PATH`, `REFRESH_S`, ...).

| Key | Default | Meaning |
|---|---|---|
| `rings_path` | `/app/data/rings.parquet` | F4 rings |
| `alert_budget` | 0.01 | false-positive budget |
| `expected_alert_rate` | 0.027 | total alert rate expected at that budget; "over budget" is flagged above this |
| `refresh_s` | 5 | live panel refresh |
| `throughput_window_s` | 60 | throughput window |
| `latency_last_n` | 5000 | latency percentiles window |
| `timeline_bucket_s` / `timeline_last_s` | 10 / 600 | chart buckets and span |
| `queue_limit` | 100 | alerts in the queue |
| `ring_graph_max_clients` | 40 | cap for the ring network chart |

### `configs/drift.yaml` (F6)
| Section | Keys |
|---|---|
| `psi` | `bins` 10, `max_categories` 20, `eps` 1e-4, `feature_threshold` 0.2 |
| `retrain` | `checks`, `min_window` 500, `drift_share` 0.3, `score_psi` 0.2, `min_reviewed` 50, `min_precision` 0.35, `alert_budget` 0.01, `max_fpr_budget_ratio` 1.5, `max_alert_rate_ratio` 2.0, `min_pr_auc` 0.40, `cooldown_hours` 24 |
| `exclude_from_share` | time-trend columns left out of the drift share |
| `offline` | Kaggle input/model/output dirs, `window_days` 7, `min_window_rows` 2000, `score_reference: val`, Evidently options, smoke sizes |
| `monitor` | `interval_s` 30, `window_rows` 2000, `reference_path`, `baseline_rows`, `status_path`, `history_limit`, `feedback_source` (`feedback_label` or `true_label`), `score_column`, `live_columns`, `auto_retrain` true, `execute_retrain` false; env override `MONITOR_<KEY>` |
| `trigger` | `kaggle_cli` (or env `KAGGLE`), `kernel_dir: kaggle/train`, `state_path` |

### Offline job configs
| File | Main keys |
|---|---|
| `configs/data.yaml` | `data_dir`, `out_dir`, `start_date`, `split` (0.70/0.15/0.15), `target`, `smoke_rows` |
| `configs/features.yaml` | `client_key_columns`, `card_key_columns`, `device_key_columns`, `windows` (1h/24h/7d), `passthrough_numeric`, `passthrough_categorical`, `parity_rows` |
| `configs/train.yaml` | `max_fpr` 0.01, `experiments`, `xgb`, `smote` (`sampling_strategy` 0.2), `logreg`, `register_name` `aiml01-xgb` |
| `configs/gnn.yaml` | `entities`, `gnn` (hidden 64, 2 layers, 400 epochs, patience 60), `rings` (`max_clients_per_device` 20, `min_size` 3), `blend_grid` |
| `configs/mlflow.yaml` | `tracking_uri` (DagsHub), `experiment` `aiml01-fraud`, `token_file`, `fallback_uri` |
