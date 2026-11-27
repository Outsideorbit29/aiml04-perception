# Demo video script (F8.4)

Target length **3:35** (limit 3–4 min). Screen recording of the Docker Compose stack on the
laptop, plus short browser shots of Kaggle and MLflow (DagsHub), title cards and a closing
results card. Every number on a card comes from `docs/EXPERIMENTS.md`, `SERVING.md`,
`MODEL.md`, `GRAPH.md` or `DRIFT.md`. Numbers that appear in a live terminal or on the live
dashboard are whatever the recording shows; do not caption them with the offline numbers.

> **Status (2026-11-27).** Not recorded yet. Two parts have **not been run live** before and
> need a rehearsal first: the `monitor` service with drift injection (checked only without
> Docker, `docs/DRIFT.md`) and the `/score` helper call in shot 5. The dashboard service was
> added to compose in `e87d838`. Recording is the user's action, in one laptop session of
> 30 minutes or less.

## Shot list

| # | Time | Dur. | On screen | Command (Git Bash, AIML-01 checkout, Docker Desktop running) | Narration / caption |
|---|---|---|---|---|---|
| 1 | 0:00 | 0:12 | Title card: "AIML-01 Real-Time Financial Fraud Detection. Kafka → Feast → XGBoost + GNN → FastAPI → analyst dashboard." Small copy of the architecture diagram (deck slide 7). | – | "This is a fraud detection stack that scores card transactions as they stream in, on a laptop, with Docker Compose." |
| 2 | 0:12 | 0:23 | Terminal, full screen. Images are pre-built, so `up` takes seconds. Then `docker compose ps`, then the consumer log until `consumer ready`. Label **"4x"** on the log wait if sped up. | `docker compose up -d kafka redis scorer consumer monitor dashboard`<br>`docker compose ps`<br>`docker compose logs -f consumer` | "Six services: Kafka, Redis as the Feast online store, the FastAPI scorer, the stream consumer, a drift monitor and the dashboard. The consumer warms its feature state from about half a million past transactions, around 13 seconds, before it reads the stream." |
| 3 | 0:35 | 0:20 | Split screen: terminal left (producer output), dashboard right (Live overview counters rising). | `docker compose run --rm producer --rate 50 --limit 5000` | "The producer replays the test period, transactions the models never saw, at 50 per second. Each one gets features computed from the client's earlier transactions, a score, and a row in the case store." |
| 4 | 0:55 | 0:45 | Dashboard, browser full screen (`http://localhost:8501`). (a) Live overview: alert rate vs the 1% budget, latency p50/p95, throughput. (b) Alert queue: hover the score breakdown; click **Confirm fraud** on one alert and **Dismiss** on another. (c) Fraud rings: pick a small ring, show the network graph. | – | (a) "The overview tracks the alert rate against a budget of 1% of legit traffic, and end-to-end latency." (b) "Each alert shows why it fired: the XGBoost probability, the client's graph risk, and how much the graph moved the score. Confirm or dismiss stores a feedback label." (c) "Rings are clients that share a specific device. On test data they have about four times the normal fraud rate." |
| 5 | 1:40 | 0:30 | Browser: `http://localhost:8000/docs` (expand `POST /score` to show the request and response schema), then `/model`. Terminal: the helper below, showing the JSON response. | see "Scoring one transaction" below | "The scorer is a FastAPI service. It reads the client's features from Feast, refuses if they belong to another transaction, and returns the blended score, the XGBoost part, the graph risk, the alert flag and its own timing, a few milliseconds." |
| 6 | 2:10 | 0:45 | Split screen: producer output left; `docker compose logs -f monitor` right. Hold on the line with `retrain True` and the `[dry-run] would retrain: kaggle kernels push -p kaggle/train` line. Time-lapse label (e.g. **"8x"**) while waiting. Optional: dashboard overview showing the alert rate. | `docker compose run --rm producer --rate 50 --start 5000 --limit 5000 --drift amount_scale=3,shift_after=2000`<br>`docker compose logs -f monitor` | "Now I inject drift: after 2,000 normal transactions, every amount is multiplied by three. The monitor checks the latest 2,000 cases every 30 seconds. The amount distribution shifts, the drift share crosses its threshold, and the monitor triggers a retrain. By default it's a dry run: it records the Kaggle command instead of launching the training job." Caption: "Synthetic drift. On the real 9-week replay no trigger fired; the false-alert rate drifted 0.88% → 1.36%." |
| 7 | 2:55 | 0:20 | Browser: Kaggle notebook `anish2906/aiml01-train` (version list, logs), then DagsHub MLflow experiment `aiml01-fraud` (runs table), registered model `aiml01-xgb`, run `f6-drift-full` (drift metrics per window). | – | "Training, evaluation and drift reports run as Kaggle jobs. Every run is logged to MLflow on DagsHub, and the serving model is registered there." |
| 8 | 3:15 | 0:20 | Closing results card (static, below). | – | "Results on the held-out test period: the served blend reaches 0.525 PR-AUC and catches about half of the fraud at the alert budget, end-to-end in 128 milliseconds at p95." |
| | **3:35** | | | | |

If the cut runs long, shorten shot 4 to 0:35 (skip the Model tab and one click) and shot 6 to
0:35. If it runs short, add 10 s in shot 4 on the Cases tab (open / confirmed / dismissed).

## Scoring one transaction (shot 5)

`/score` only accepts a transaction whose features are the latest ones in Feast for its client
(otherwise HTTP 409), and a real transaction has 400+ fields, so typing one into the Swagger page
is not practical. After the first replay has finished, this re-scores the last replayed
transaction from inside the scorer container. **Rehearse it once**; if it returns 409, show that
response instead and say it is the stale-feature guard.

```bash
# Git Bash, AIML-01 checkout, after shot 3's replay has finished
docker compose exec -T scorer python - <<'EOF'
import json, sqlite3, urllib.request
import pandas as pd
from services.common import json_safe
tid = sqlite3.connect("/app/data/cases.db").execute(
    "SELECT MAX(transaction_id) FROM scored").fetchone()[0]
df = pd.read_parquet("/app/data/replay.parquet")
row = json_safe(df[df["TransactionID"] == tid].to_dict("records")[0])
req = urllib.request.Request("http://localhost:8000/score", data=json.dumps(row).encode(),
                             headers={"Content-Type": "application/json"})
print(json.dumps(json.loads(urllib.request.urlopen(req).read()), indent=2))
EOF
curl -s localhost:8000/model
```

Response fields to point at: `score`, `p_xgb`, `gnn_risk`, `alert`, `threshold`, `fetch_ms`,
`model_ms`, `total_ms`.

## Closing results card (shot 8)

| Area | Result | Source |
|---|---|---|
| Data | IEEE-CIS, 590,540 transactions, 3.5% fraud, split by time | EDA.md |
| Features | Batch vs stream parity 1.6e-11 on 30,000 tx; Feast point-in-time 100% | FEATURES.md |
| XGBoost + SMOTE (test) | PR-AUC 0.491, recall 45.4%, precision 55.8% at the val 1% FPR threshold | MODEL.md |
| Served blend, XGBoost + client GNN risk (test) | PR-AUC 0.525, recall 49.8%, precision 58.0% | GRAPH.md |
| Fraud rings (test) | 9.9% fraud in rings vs 2.5% outside | GRAPH.md |
| Serving, laptop, 50 tx/s | end-to-end p50 83 ms, p95 128 ms; burst 242 tx/s | SERVING.md |

Footer line: "Laptop: Docker Desktop, 4 CPUs / 6 GB. Offline drift over 9 weeks: no retrain
trigger; false-alert rate 0.88% → 1.36%."

## Recording checklist

### Before the session (can be done earlier, not during recording)
- [ ] Serving bundle present: `KAGGLE=<path to kaggle.exe> bash scripts/fetch_serving_bundle.sh`
- [ ] Drift reference for the monitor (optional, better baseline):
      `kaggle kernels output anish2906/aiml01-drift -p serving_bundle/data --file-pattern 'drift_reference\.json$'`
- [ ] Images built: `docker compose build` (do this before, not while recording)
- [ ] Rehearse shots 5 and 6 once, then `docker compose down`
- [ ] Fresh case store: with the stack down, move the old one aside, e.g.
      `mkdir -p serving_bundle/old && mv serving_bundle/data/cases.db* serving_bundle/old/`
      (the consumer creates a new one), and `monitor_status.json` so the monitor's history starts
      clean. A dry run does not start the 24 h retrain cool-down (only a real launch writes
      `retrain_state.json`), so rehearsals do not block the trigger.
- [ ] Log in to Kaggle and DagsHub in the browser; open the tabs for shot 7 in advance

### Low laptop load
- [ ] Close CARLA, the AIML-04 ROS2/WSL session, Minikube and other heavy apps; the compose
      stack alone uses about 2–3 GB (burst: consumer 563 MB, Kafka 349 MB, scorer 226 MB)
- [ ] Docker Desktop resources as in the load test (4 CPUs, 6 GB WSL cap)
- [ ] Keep the session under 30 minutes; `docker compose down` right after the last take
- [ ] Record each shot as a separate short take instead of one long recording

### OBS settings
- Canvas and output **1280x720**, **30 fps** (enough for terminal and dashboard text)
- Encoder: **NVENC H.264** (hardware, uses the RTX 3060 and spares the CPU); rate control CQP 23,
  or CBR 4,000 kbps
- Record to **MKV** (safe if OBS crashes), then *File → Remux Recordings* to MP4
- Audio: 48 kHz mic, noise suppression filter; desktop audio off
- Scenes: "Terminal", "Browser", "Split" (terminal left, browser right); hotkeys to switch

### Window layout
- Windows Terminal (Git Bash), font size 16 or more, dark theme, 100–110 columns
- Browser at 1280x720 or snapped half screen, zoom 110–125% so dashboard numbers are readable
- Hide the bookmarks bar, notifications (Windows Focus mode), the taskbar clock if distracting

### Do not show on screen
- `kaggle.json`, the DagsHub token, `.env`, the `aiml04-secrets` dataset contents
- E-mail, other browser tabs, personal account pages

## Editing and upload
- Light editor on the laptop (Clipchamp or Shotcut). Export 1280x720 H.264, under 100 MB.
- Label every speed change ("4x", "8x"); do not change speed silently.
- Title and results cards: plain text on a dark background, at most two lines per caption.
- Upload as unlisted YouTube or to the Drive folder and put the link in the README and the deck
  (`[[LINK: demo video]]` on slide 14). Upload, links and the portal submission are the user's
  actions.
