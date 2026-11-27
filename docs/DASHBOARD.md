# Analyst dashboard (F7)

A Streamlit app for fraud analysts, `services/dashboard/app.py`. It reads the case store the
consumer writes (`cases.db`, SQLite in WAL mode), the F4 fraud rings (`rings.parquet`) and the
scorer's `/model` and `/health` endpoints. The query logic is in `fraud/dashboard_data.py`
(plain pandas + sqlite, unit-tested in `tests/test_dashboard_data.py`); the app itself is
smoke-tested with Streamlit's `AppTest` in `tests/test_dashboard_app.py`.

## Features
| Area | What it shows |
|---|---|
| **Live overview** (refreshes every `refresh_s`, 5 s) | transactions scored, alerts, alert rate vs the 1% budget, latency p50/p95 (consumer end-to-end per transaction, last 5,000), throughput over the last 60 s, live precision of reviewed alerts, and a chart of tx/s and alerts over the last 10 minutes |
| **Alert queue** | open alerts sorted by score with amount, product, client/card/device keys, ring id, and the score breakdown: `p_xgb` (XGBoost), `gnn_risk` (client graph risk) and `gnn_shift` (how much the graph moved the blended score). **Confirm fraud** / **Dismiss** buttons call `CaseStore.set_status`, which stores the feedback label (1 / 0) F6 uses. Filters for status and minimum score; any alert's status can also be changed by id |
| **Cases** | open / confirmed / dismissed counts and live precision = confirmed / reviewed. An **evaluation-only** expander shows alert precision, recall, FPR and analyst agreement against the replay's `isFraud`, clearly marked as hidden from analysts |
| **Fraud rings** | rings (clients sharing a specific device fingerprint, docs/GRAPH.md) that appear in the live stream: size, devices, active clients, transactions, alerts, alert share, confirmed/dismissed, max score; true fraud share only behind an "evaluation only" checkbox. A Graphviz network of the selected ring links clients (red = has alerts) to the devices and cards they used, capped at 40 clients for the 7,522-client chain component |
| **Model** | `GET /model` (model name, threshold, XGBoost weight in the blend, clients with GNN risk), `GET /health`, and the consumer's `consumer_stats.json` stage latencies |

## Configuration
`configs/serving.yaml` (`cases_db`, `scorer_url`, `stats_path`, `http_timeout_s`) merged with
`services/dashboard/dashboard.yaml` (`rings_path`, `alert_budget`, `refresh_s`, window sizes,
queue limit, graph cap). Every key can be overridden with an upper-case environment variable:
`CASES_DB`, `SCORER_URL`, `RINGS_PATH`, `STATS_PATH`, `REFRESH_S`, ...

The dashboard is a second SQLite writer (analyst decisions only, a few rows at a time), next to
the consumer. WAL mode and a 30 s busy timeout make that safe on one host or one Minikube node.

## Run
**Docker Compose** (image `aiml01-dashboard:dev` from `services/dashboard/Dockerfile`, which is
lighter than the services image: no xgboost, Feast or Kafka). Add this service to
`docker-compose.yml`:
```yaml
  dashboard:
    build: {context: ., dockerfile: services/dashboard/Dockerfile}
    image: aiml01-dashboard:dev
    volumes:
      - ./serving_bundle/data:/app/data
    ports: ["8501:8501"]
    mem_limit: 512m
    depends_on:
      scorer: {condition: service_started}
```
```bash
docker compose up -d --build kafka redis scorer consumer dashboard
docker compose run --rm producer --rate 50 --limit 5000
# open http://localhost:8501
```

**Locally** (Python 3.10–3.13 with `pip install -r requirements-dashboard.txt`), from the repo
root, against the compose stack's bundle folder:
```bash
CASES_DB=serving_bundle/data/cases.db RINGS_PATH=serving_bundle/data/rings.parquet \
STATS_PATH=serving_bundle/data/consumer_stats.json SCORER_URL=http://localhost:8000 \
  streamlit run services/dashboard/app.py
```

**Kubernetes (Minikube):** see [DEPLOY.md](DEPLOY.md); the dashboard is on NodePort 30851.

## Public demo (Streamlit Community Cloud)
Live URL: **`<add the Streamlit Cloud URL here>`** (F7.3, the portal's optional Live Deployment URL)

The public app runs the same `services/dashboard/app.py` in **demo mode**. The entrypoint
`services/dashboard/streamlit_app.py` sets `DASHBOARD_DEMO=1`; at startup
`fraud/demo_data.py` generates, from a fixed seed, a case store with the `fraud.cases` schema
(4,000 scored transactions over 10 minutes, ~3% alerts, ~3.4% fraud, a third of the alerts
already reviewed), five fraud rings and a consumer-stats snapshot, written to a temp folder
(`DASHBOARD_DEMO_DIR`). Nothing comes from IEEE-CIS (competition rules forbid redistribution):
keys start with `demo-`, transaction ids start at 9,000,000 and scores come from simple
distributions, not the trained model. There is no Kafka, Redis or scorer; the Model tab shows the
synthetic scoring rule instead of `GET /model`. A banner says so on every page.

Confirm / Dismiss work against the demo DB. All visitors share it until the app restarts
(Streamlit Cloud's disk is ephemeral) or someone clicks **Reset demo data** in the sidebar.
`services/dashboard/requirements.txt` (next to the entrypoint, so Streamlit Cloud uses it instead
of the root `requirements.txt`) holds only streamlit, numpy, pandas, pyarrow and pyyaml.

**Deploy (one time, about 5 minutes):**
1. Go to <https://share.streamlit.io> and sign in with GitHub (account `Outsideorbit29`); allow
   access to the public repo when asked.
2. Click **Create app** → **Deploy a public app from GitHub** (or **New app**).
3. Repository `Outsideorbit29/aiml04-perception`, branch `AIML-01`, main file path
   `services/dashboard/streamlit_app.py`.
4. Optional: set **App URL** to something readable, e.g. `aiml01-fraud-dashboard`.
5. **Advanced settings** → Python version **3.12**. No secrets are needed. Save, then **Deploy**.
6. The first build takes a few minutes. Open the URL, check the yellow demo banner, confirm one
   alert, and paste the URL above and into the portal.

Free apps sleep after a few days without visits; opening the URL wakes them in about a minute.
Every push to `AIML-01` redeploys the app.

**Run demo mode locally** (any Python 3.10–3.13 with the dashboard requirements):
```bash
streamlit run services/dashboard/streamlit_app.py          # or: DASHBOARD_DEMO=1 streamlit run services/dashboard/app.py
```
The Docker image and the Kubernetes deployment keep demo mode off (`DASHBOARD_DEMO` unset).
