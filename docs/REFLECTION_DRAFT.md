# Individual reflection — DRAFT (AIML-01)

> **DRAFT. Do not submit this text as it is.** It is a scaffold written from the repo history,
> `PLAN.md` (key decisions), `docs/EXPERIMENTS.md` and the write-ups in `docs/`. The portal asks
> for an *individual* reflection on what **you** learned and what **you** owned, so you must
> **rewrite it in your own words**, keep only what is true for you, answer the guiding questions,
> and replace every `[fill in: ...]`. Delete this box when done. Target: one page (about 500–700
> words). Save the final version as `docs/REFLECTION.md` (PLAN.md F8.6, owner: the user).
>
> Facts below are from the docs. Feelings, anecdotes and opinions are left to you on purpose.

## What I built

Guiding questions: How would you explain the project to someone outside ML in three sentences?
What does it do end to end, and what is the one result you would point to?

Facts you can use:
- A streaming fraud scoring stack for the Nexvia AIML-01 brief, on IEEE-CIS Fraud Detection
  (590,540 transactions, 3.5% fraud, split by time 70/15/15).
- Point-in-time client features shared by training and the stream consumer (parity 1.6e-11 on
  30,000 transactions), stored in Feast with a Redis online store.
- XGBoost with SMOTE (test PR-AUC 0.491, recall 45.4% at the val 1% FPR threshold), plus a
  GraphSAGE model whose per-client risk is blended in live (test PR-AUC 0.525, recall 49.8%).
- Fraud rings from shared device fingerprints: 1,001 rings, 9.9% fraud inside vs 2.5% outside
  on test.
- Kafka → consumer → Feast/Redis → FastAPI scorer → SQLite case store → Streamlit dashboard, in
  Docker Compose: end-to-end p50 83 ms / p95 128 ms at 50 tx/s, burst 242 tx/s on the laptop.
- Weekly drift report (PSI + Evidently) and a live monitor with a dry-run retrain trigger;
  Kubernetes manifests for Minikube (written and tested in CI, not yet run on a cluster).

[fill in: one or two sentences in your own words on what you are most satisfied with, and why.]

## What I owned

Guiding questions: Which decisions did you make yourself? What did you write, run, debug or
check with your own hands? Where did you rely on someone or something else?

Facts you can use:
- The branch was built on 2026-11-26 and 2026-11-27 (42 commits before the F8 deliverables),
  in parallel with AIML-04, with the same cloud-first setup.
- Claude Code with project subagents wrote much of the code and documentation; the commit
  trailers show this.

[fill in: be specific and honest. Examples to check against your memory: choosing IEEE-CIS over
PaySim, the alert-budget operating point, accepting the Kaggle rules and setting up DagsHub,
running every Kaggle job and reading its smoke output, the laptop load test, deciding when a
milestone was done, judging whether numbers were plausible.]

[fill in: what you did not own or did not fully understand, if anything.]

## Challenges I actually met

Guiding questions: Pick two or three. What was the symptom, how did you find the cause, what
fixed it, and what would you check earlier next time?

- **Kaggle rules acceptance.** IEEE-CIS is competition data, so the Kaggle jobs could only read
  it after the competition rules were accepted once on kaggle.com (PLAN.md F0.3). The repo ships
  300 synthetic rows with the same schema instead of the real data (F1.3).
  [fill in: how did you notice, and how long did it cost?]
- **Weak card-keyed features, fixed by a client key.** The first feature version keyed on
  card1–card6 and raw DeviceInfo. Those turned out to be product and OS labels (about 800
  transactions per "card", 1,700 cards per "device"), and single-feature AUC stayed at
  0.43–0.62. Re-keying on `card1 | addr1 | day − D1` gave 217,850 clients and real velocity
  signal (`cnt_7d` AUC 0.644, `secs_since_prev` 0.330). [fill in: what made you suspect the key?]
- **Consumer throughput 43 → 242 tx/s with micro-batching.** The first consumer handled one
  message at a time: about 23 ms per transaction (Feast push 6.2 ms, scorer 4.4 ms, SQLite
  commit + publish about 10 ms), so at most about 43 tx/s and it could not keep up with the
  50 tx/s replay. Taking up to 200 messages per poll, one Feast push and one commit per batch,
  reached 242 tx/s in a burst. [fill in]
- **Poll timeout 1 s → end-to-end p50 83 ms.** The first consumer called `consumer.poll(1.0)`.
  The batched poll waits at most 50 ms (`consumer_poll_s: 0.05`), which bounds the delay that
  batching adds; at 50 tx/s the end-to-end latency is p50 83 ms, p95 128 ms, p99 161 ms.
  [fill in: how did you trade batch size against latency?]
- **SMOTE needs imputation, and serving has to match it.** SMOTE cannot work with missing
  values, so training filled them with training medians. To serve that model, the medians are
  stored in the feature spec (`spec.json`) and the scorer imputes the same way. Open question:
  part of SMOTE's +0.04 PR-AUC may come from the imputation, not the synthetic samples.
  [fill in]
- **scikit-learn dependency in CI / serving.** The scorer first loaded the model through
  XGBoost's scikit-learn wrapper, which needs scikit-learn; the lean serving image does not
  install it. The scorer and its test now load a plain XGBoost `Booster` (commit `7d4e80c`).
  [fill in: what did the failure look like?]

## Decisions and trade-offs

Guiding questions: Which of these did you actually weigh? What was the alternative, and would
you make the same call again?

- **Cloud first:** data prep, training, GNN and drift jobs on Kaggle; MLflow on DagsHub; tests
  in GitHub Actions; the laptop only runs the serving stack. [fill in]
- **Time-based split and a fixed alert budget** (1% of legit val transactions) instead of a
  random split and a default 0.5 threshold. [fill in]
- **Serve the client-risk blend (0.525), not the full-graph blend (0.545),** because the full
  graph is transductive and cannot be served. [fill in]
- **SQLite case store** with one consumer, instead of Postgres. [fill in]
- **Keep the drift thresholds as designed** even though no trigger fired on the real data (FPR
  0.88% → 1.36%, trigger at 1.5× the budget); the demo injects synthetic drift instead.
  [fill in]
- **Dry-run retrain trigger by default.** [fill in]

[fill in: one decision you would defend in an interview, and one you are less sure about.]

## What I learned

Guiding questions: What do you understand now that you did not before? Pick two or three and
write what *you* took from them.

- Entity keys decide whether behavioural features work at all. [fill in]
- Training-serving parity has to be tested, not assumed (batch builder vs `OnlineFeatureState`,
  Feast point-in-time join). [fill in]
- A fixed threshold can overspend the alert budget before any input drifts. [fill in]
- Offline gains do not always carry over: the F2 features added almost nothing offline but are
  what make live scoring possible. [fill in]
- Throughput problems were in I/O and commits, not in the model (scorer p95 5.3 ms). [fill in]
- [fill in: something about planning two projects with one deadline, or working with AI coding
  agents.]

## What I would do differently

[fill in in your own words. Possible points from the record; keep only those you agree with:]

- Check the entity key's cardinality before building features on it.
- Run the weights-plus-imputation ablation to explain the SMOTE gain.
- Run the monitor and the Minikube manifests live earlier, not only in unit tests.
- [fill in]

## Numbers I may quote (all from the docs)

| Area | Number | Source |
|---|---|---|
| Data | 590,540 tx; train/val/test 413,378 / 88,581 / 88,581; fraud 3.52 / 3.43 / 3.48% | EDA.md |
| Feature parity | max rel. diff 1.6e-11 on 30,000 tx; Feast PIT 100% on 5,000 | FEATURES.md |
| XGBoost + SMOTE (test) | PR-AUC 0.491, ROC-AUC 0.890, recall 45.4%, precision 55.8%, FPR 1.30% | MODEL.md |
| Live blend (test) | PR-AUC 0.525, recall 49.8%, precision 58.0% | GRAPH.md |
| Rings (test) | 1,001 rings; 9.9% fraud inside vs 2.5% outside | GRAPH.md |
| Serving (laptop) | e2e p50 83 / p95 128 / p99 161 ms at 50 tx/s; scorer p95 5.3 ms; burst 242 tx/s | SERVING.md |
| Live detection (first 5,000 test tx) | 64 of 131 frauds (49%); 83 false alerts of 4,869 legit (1.7%) | SERVING.md |
| Drift (9 weeks) | PSI drift share 0; score PSI ≤ 0.006; FPR 0.88% → 1.36%; no trigger | DRIFT.md |
