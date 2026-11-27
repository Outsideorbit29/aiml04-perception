# Speaker notes – AIML-01 deck (F8.5)

Deck: `presentation/deck/index.html` (open in a browser; arrow keys or swipe, `t` toggles
light/dark, Ctrl+P prints one slide per page for a PDF). 14 slides, **10:00 total**. Speak at
about 130–140 words per minute; the word counts below are sized for the target time.

Every number here is from `docs/EDA.md`, `FEATURES.md`, `MODEL.md`, `GRAPH.md`, `SERVING.md`,
`DRIFT.md`, `DEPLOY.md` or `EXPERIMENTS.md`. If a number changes in those docs, change it here
and in `index.html` in the same commit.

These notes are a starting point. Say them in your own words; do not read them out.

| # | Slide | Target | Running |
|---:|---|---:|---:|
| 1 | Title | 0:25 | 0:25 |
| 2 | Problem | 0:45 | 1:10 |
| 3 | Data | 0:45 | 1:55 |
| 4 | Features + Feast | 0:50 | 2:45 |
| 5 | Models | 0:50 | 3:35 |
| 6 | Graph + rings | 0:50 | 4:25 |
| 7 | Architecture | 0:50 | 5:15 |
| 8 | Serving results | 0:45 | 6:00 |
| 9 | Drift | 0:55 | 6:55 |
| 10 | Dashboard | 0:40 | 7:35 |
| 11 | Kubernetes | 0:35 | 8:10 |
| 12 | Challenges | 0:50 | 9:00 |
| 13 | Limitations + next steps | 0:40 | 9:40 |
| 14 | Summary + questions | 0:20 | 10:00 |

If you run long, shorten slides 11 and 13 first; they are also covered by the docs.

---

## 1. Title (0:25)

This is AIML-01, a real-time fraud detection platform. It reads transactions from a stream,
scores each one in milliseconds, groups suspicious accounts into fraud rings, and gives
analysts a dashboard to review the alerts. Four numbers to keep in mind: 590,540 transactions,
a test PR-AUC of 0.525 for the model I actually serve, 128 milliseconds end-to-end at the 95th
percentile, and 242 transactions per second in a burst, all on a laptop.

## 2. Problem (0:45)

The brief names the tools: Kafka, Feast, XGBoost, PyTorch Geometric, MLflow, FastAPI,
Evidently and Kubernetes. The hard parts are not the tools. First, imbalance: a model that
says "legit" every time is about 97% accurate, so accuracy tells you nothing. Second,
training-serving skew: features built in batch for training have to equal the features the
stream computes one transaction at a time. Third, fraud changes week to week. And analysts can
only review so many alerts, so I fixed the operating point at an alert budget of 1% of legit
transactions. Everything heavy ran in the cloud, on Kaggle; the laptop runs only the serving
stack in Docker, with 4 CPUs and 6 GB.

## 3. Data (0:45)

I used IEEE-CIS Fraud Detection: 590,540 labelled transactions, 439 columns after joining the
identity table. Fraud is 3.5%. The brief says under 1%; that fits other datasets, but the
conclusion is the same: judge by PR-AUC and recall at a fixed false-positive rate. I split by
time, 70/15/15 over 182 days, not randomly. The chart shows why: the daily fraud rate moves
between about 1 and 7 percent and steps up after day 25. A random split would leak the future
into training and hide exactly the drift I want to monitor later. Two EDA findings that shaped
the features: product C has 11.2% fraud against 2.1% for W, and only 27% of transactions have
identity data.

## 4. Features + Feast (0:50)

My first feature version keyed everything on the card fields. That was a mistake: card1 to
card6 describe a card product, about 800 transactions per "card", and single features scored
an AUC between 0.43 and 0.62, close to chance. I switched to a client key: card1, address, and
the transaction day minus D1, which is constant per cardholder. That gives 217,850 clients.
Features only look at earlier transactions: velocity over 1 hour, 24 hours and 7 days, amount
against the client's mean, time since the previous transaction. The key result is parity: the
batch builder and the streaming state agree to a relative difference of 1.6e-11 on 30,000 real
transactions, and Feast's point-in-time join matched 100% on 5,000.

## 5. Models (0:50)

All models are judged at the same operating point: the threshold is set on validation so 1% of
legit transactions are flagged, and test is reported once. Logistic regression reaches 0.357
PR-AUC; XGBoost 0.46; XGBoost with SMOTE 0.491, with 45% recall and 56% precision. That model
is registered in MLflow as aiml01-xgb. Two honest notes. My F2 features add almost nothing
offline, because the raw C and D columns are already per-card counts; their value is that they
can be computed live. And SMOTE also fills missing values with training medians, so part of its
gain may be the imputation; I have not separated the two. Finally, the same threshold gives
1.2 to 1.6% false positives on test. That is the first sign of drift.

## 6. Graph + rings (0:50)

The graph links transactions to clients, card products and device fingerprints. A two-layer
heterogeneous GraphSAGE trained in 104 seconds on a Kaggle T4. Alone it scores 0.506; blended
with XGBoost, 0.545. But that number is transductive: validation and test transactions sit in
the graph during training, so I can't serve it. What I serve is a lookup: each client's average
GNN risk from earlier periods, blended with XGBoost. That gives 0.525 test PR-AUC and 4.4 more
points of recall at the same alert rate, for microseconds of cost. On the right: rings are
clients linked by a specific shared device. Found without labels, the 1,001 rings carry 9.9%
fraud on test against 2.5% outside, about four times higher.

## 7. Architecture (0:50)

The live path from left to right. A producer replays the test period into Kafka. The consumer
updates the same feature code used in training, pushes the client's features to Feast's Redis
store, then calls the FastAPI scorer. The scorer reads the features back from Feast and refuses
with HTTP 409 if they belong to a different transaction, so it can never score with stale
features. Every score goes to a SQLite case store; the Streamlit dashboard reads it, and analyst
decisions go back as feedback labels. The drift monitor reads the same store every 30 seconds
and can trigger a retrain on Kaggle, as a dry run by default. The model, threshold and client
risk come in a 30 MB serving bundle built by a Kaggle job.

## 8. Serving results (0:45)

On my laptop, Docker Desktop with 4 CPUs and 6 GB, at a steady 50 transactions per second the
end-to-end latency, from the producer to scored and stored, is 83 milliseconds at the median and
128 at p95. The scorer call itself is 5.3 milliseconds at p95. The first consumer handled one
message at a time and topped out around 43 per second; micro-batching took it to 242 in a burst.
The detection result holds up live: on the first 5,000 replayed test transactions it caught 64
of 131 frauds, 49%, with 1.7% false alerts. The offline test number was 49.8% recall at about
1.3%. The false-alert rate is higher in the early test weeks; that is what the monitor tracks.

## 9. Drift (0:55)

I ran the drift job on nine weekly windows against the training data. The inputs did not drift:
no feature crossed a PSI of 0.2, Evidently flagged 8 to 22% of inputs, under its 30% default,
and the score distribution barely moved. But the false-alert rate at the fixed threshold climbed
from 0.88% in week 1 to 1.36% in week 9, and weekly PR-AUC fell from 0.69 to between 0.42 and 0.54. My
trigger is set at 1.5 times the budget or PR-AUC below 0.40, so nothing fired. At 1.25 times it
would have fired in weeks 6 to 9. I kept the thresholds as designed instead of tuning them to
make the demo fire. The lesson: here a fixed threshold overspends the alert budget before the
inputs drift, so re-picking the threshold is cheaper than retraining. For the live demo I inject
drift, amounts times three, to show the trigger.

## 10. Dashboard (0:40)

The dashboard is for the analyst. The overview shows volume, the alert rate against the 1%
budget, latency and throughput. The alert queue sorts open alerts by score and shows why: the
XGBoost probability, the client's graph risk, and how much the graph moved the score. Confirm or
dismiss writes the feedback label the monitor uses for live precision. Ground truth is only
behind an "evaluation only" switch, so an analyst never sees it by accident. The ring view lists
the rings active in the stream and draws the clients, devices and cards of the selected ring.

## 11. Kubernetes (0:35)

The brief asks for Kubernetes; Minikube is allowed. I wrote manifests for every component in an
aiml01 namespace, with the same service names as Docker Compose so the configs work unchanged.
One detail the manifests have to handle: Kubernetes injects a KAFKA_PORT variable that the Kafka
image reads as a broker setting, so service links are turned off. The manifests are checked in
CI against compose and the configs. To be clear: I have not run them on Minikube yet; the demo
uses Docker Compose.

> Update this slide and note if a Minikube run happens before the talk.

## 12. Challenges (0:50)

Six things that went wrong. The card-keyed features were too coarse; the client key fixed that.
The consumer was too slow: about 23 milliseconds per message, mostly the Feast push and the
SQLite commit, so at most 43 per second; batching fixed it. It also polled Kafka with a one
second timeout; the batched poll waits at most 50 milliseconds. SMOTE needs complete data, so
training imputed with medians, and the scorer has to impute the same way; the medians now ship
in the feature spec. Loading XGBoost through its scikit-learn wrapper pulled in scikit-learn,
which the lean serving image does not have; it now loads a plain Booster. And the data is
competition data, so Kaggle needed the rules accepted once, and the repo only ships synthetic
fixtures.

## 13. Limitations + next steps (0:40)

The best offline number, 0.545, can't be served; 0.525 can. Device fingerprints are coarse,
which is why one ring component grows to 7,522 clients. The SMOTE ablation is missing. The
dataset has no newer data, so a retrain can only reuse the same splits. The live monitor sees
amount, product and scores, not every input. SQLite limits me to one consumer. Next steps, in
order: re-pick the threshold on recent feedback, add device and card risk to the live blend, run
the SMOTE ablation, and move the case store to Postgres.

## 14. Summary + questions (0:20)

To sum up: features that match between training and serving, a served model at 0.525 PR-AUC
with half the fraud caught at the alert budget, rings with four times the base fraud rate, and
a stack that scores in about 128 milliseconds at p95 on a laptop. Thank you; happy to take
questions.

---

## Likely questions (answers from the docs)

- **Why not accuracy or ROC-AUC?** At 3.5% fraud, "always legit" is about 97% accurate. ROC-AUC
  is reported (0.890 for XGBoost + SMOTE) but PR-AUC reflects the alert queue.
- **Why does the full-graph blend not go live?** Val/test transactions were nodes during
  training (transductive). The live blend uses client risk from earlier periods only.
- **Why didn't the retrain trigger fire?** Offline, no window crossed the configured thresholds
  (1.36× budget vs 1.5×; PR-AUC ≥ 0.421 vs 0.40). The live demo uses injected drift.
- **Is the latency on a server?** No: laptop Docker Desktop, 4 CPUs / 6 GB, one consumer, SQLite.
- **Why SQLite and not Postgres?** Simplest store that works with one consumer; WAL mode lets the
  dashboard write analyst decisions at the same time. Postgres is the scaling step.
- **What did the GNN add?** +0.034 test PR-AUC and +4.4 points of recall at the same alert rate
  (live blend vs XGBoost alone).
