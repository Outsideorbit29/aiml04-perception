# Streaming and scoring service (F5)

Code: `services/` (producer, consumer, scorer), `fraud/features.py` (`OnlineFeatureState`),
`fraud/scoring.py`, `fraud/store.py`, `fraud/cases.py`. Config: `configs/serving.yaml`.
Stack: `docker-compose.yml` (Kafka 3.8 KRaft, Redis 7.2, one `aiml01-services` image).

## Flow
```
replay.parquet ──producer──▶ Kafka `transactions` ──consumer──┬─▶ OnlineFeatureState.update(tx)   (F2 features, same code as training)
                                                              ├─▶ Feast push → Redis             (client_features view, tagged with TransactionID)
                                                              ├─▶ POST scorer /score             (FastAPI: Feast read → XGBoost + client GNN risk)
                                                              ├─▶ SQLite case store              (every score; alerts start "open")
                                                              └─▶ Kafka `scores`
```
- **Warm start:** the consumer rebuilds the feature state from 501,959 train+val transactions in
  ~13 s (`OnlineFeatureState.from_history`; a unit test checks it is bit-identical to replaying
  them one by one).
- **Scorer contract:** `/score` reads the client's features from Feast and refuses (HTTP 409) if
  they belong to a different transaction, so it can't score with stale features.
- **Scoring rule:** the F4 live blend, `0.5·logit(p_xgb) + 0.5·logit(client GNN risk)` with
  threshold 0.310 (XGBoost alone for unseen clients). The model card is at `GET /model`.
- **Micro-batching:** each Kafka poll takes up to 200 messages (waiting at most 50 ms). Features are
  pushed to Feast in one call and cases committed in one transaction; each transaction is still
  scored by its own `/score` call. A client repeating inside a batch forces a flush first.

## Load test (laptop: Docker Desktop, 4 CPUs, 6 GB; 2026-11-27)

| Run | Messages | Result |
|---|---:|---|
| v1, one-by-one consumer, 50 tx/s | 5,000 | could not keep up: ~23 ms per tx (Feast push 6.2 ms, scorer 4.4 ms, SQLite commit + publish ~10 ms) → max ~43 tx/s |
| **v2 batched, 50 tx/s (steady state)** | 5,000 | keeps up (50.0 tx/s); **end-to-end p50 83 ms / p95 128 ms / p99 161 ms** (producer → scored and stored) |
| **v2 batched, unthrottled burst** | 10,000 | **242 tx/s sustained** |

Per-transaction stage times in the steady-state run (ms):

| Stage | p50 | p95 | p99 |
|---|---:|---:|---:|
| Feature update (`OnlineFeatureState`) | 0.02 | 0.06 | 0.10 |
| Feast push to Redis (amortised per tx) | 1.52 | 1.91 | 3.11 |
| Scorer HTTP round trip (Feast read + XGBoost + blend) | 3.81 | 5.29 | 7.88 |

Memory during the burst: consumer 563 MB, scorer 226 MB, Kafka 349 MB, Redis 8 MB.

**Live detection on the replayed test-period transactions** (case store, first 5,000): 147 alerts,
64 of 131 frauds caught (49%), 83 false alerts out of 4,869 legit (1.7%). That matches the offline
test numbers (recall 49.8% at ~1.3% FPR); the early test weeks have a slightly higher false-alert
rate, the drift F6 monitors.

## Run it
```bash
# Windows (Git Bash) or WSL, in the AIML-01 checkout; Docker Desktop running
KAGGLE=<path to kaggle CLI> bash scripts/fetch_serving_bundle.sh      # ~30 MB, gitignored
docker compose up -d --build kafka redis scorer consumer
docker compose run --rm producer --rate 50 --limit 5000               # replay test-period transactions
curl localhost:8000/model                                             # model card
cat serving_bundle/data/consumer_stats.json                           # latency / throughput
docker compose down
```
