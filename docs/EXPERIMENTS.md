# Experiment log – AIML-01

Every Kaggle job, training run and benchmark, newest last.

| Date | Milestone | Job / run | Commit | Result |
|---|---|---|---|---|
| 2026-11-27 | F1.1 smoke | Kaggle `anish2906/aiml01-data-prep` v1 (50k rows) | `8384c07` | 35k/7.5k/7.5k split, fraud 2.9% train; schema and joins OK |
| 2026-11-27 | F1.1/F1.2 full | Kaggle `anish2906/aiml01-data-prep` v2 | `5123b89` | 590,540 rows × 439 cols; train/val/test 413,378 / 88,581 / 88,581, fraud 3.52 / 3.43 / 3.48%. EDA in `docs/EDA.md` |
| 2026-11-27 | F2 v1 (card key) | Kaggle `anish2906/aiml01-features` v2 | `3f14f17` | parity OK, but card1–6 / DeviceInfo keys too coarse: single-feature AUC 0.43–0.62. Replaced |
| 2026-11-27 | F2 smoke (client key) | Kaggle `anish2906/aiml01-features` v3 (20k rows) | `b6c2bf6` | parity 1.6e-11; Feast offline/online 100% |
| 2026-11-27 | F2 full | Kaggle `anish2906/aiml01-features` v4 | `93782dc` | 590,540 rows in 45 s; 217,850 clients; parity 1.6e-11 on 30k; Feast PIT 100% (5k) + online 100%; best single features secs_since_prev (AUC 0.33), cnt_7d (0.64). `docs/FEATURES.md` |
| 2026-11-27 | F3 smoke | Kaggle `anish2906/aiml01-train` v1 (60k train rows) | `e3e051f` | pipeline + DagsHub MLflow logging OK; numbers not meaningful (3 weeks of data) |
| 2026-11-27 | F3 full | Kaggle `anish2906/aiml01-train` v2 | `be362b3` | test PR-AUC: logreg 0.357, xgb_raw 0.463, xgb 0.457, xgb_smote 0.491; class-weighted xgb registered (superseded) |
| 2026-11-27 | F3 full | Kaggle `anish2906/aiml01-train` v3, MLflow `aiml01-fraud`, model `aiml01-xgb` | `ab63519` | same numbers (deterministic); serving = xgb_smote with medians in spec: test PR-AUC 0.491, ROC-AUC 0.890, recall 45.4% / precision 55.8% at val 1%-FPR threshold (test FPR 1.30%). `docs/MODEL.md` |
| 2026-11-27 | F4 smoke | Kaggle `anish2906/aiml01-gnn` v1 (70k rows, 40 epochs, T4) | `16e1f6f` | pipeline OK; rings 4–5× fraud rate; used the older F3 model |
| 2026-11-27 | F4 full | Kaggle `anish2906/aiml01-gnn` v2 (T4, 104 s), MLflow run `gnn_blend` | `77a74c6` | test PR-AUC: xgb 0.491, GNN 0.506, blend 0.545, live blend (client risk, w=0.5, thr 0.310) 0.525 with recall 49.8% / precision 58.0%; 1,001 rings, test fraud 9.9% in rings vs 2.5% outside. `docs/GRAPH.md` |
| 2026-11-27 | F5 bundle | Kaggle `anish2906/aiml01-serving-bundle` v1 | `11d1d45` | 30 MB: model, blend (w=0.5, thr 0.310), client risk (191,596 clients), history 501,959 rows, replay 88,581 rows |
| 2026-11-27 | F5 load test | laptop docker compose (4 CPU, 6 GB) | this commit | one-by-one consumer max ~43 tx/s; batched: 50 tx/s steady with e2e p50 83 / p95 128 ms, scorer HTTP p95 5.3 ms; burst 242 tx/s; live recall 49% at 1.7% false-alert rate on first 5k test tx. `docs/SERVING.md` |
| 2026-11-27 | F6 smoke | Kaggle `anish2906/aiml01-drift` v1 (60k train vs first 40k val) | `a53c323` | PSI + Evidently 0.7.23 + MLflow OK |
| 2026-11-27 | F6 full | Kaggle `anish2906/aiml01-drift` v2, MLflow `f6-drift-full` (5bb05714) | `fdad117` | 9 weekly windows val+test vs train: PSI drift share 0 (max feature PSI 0.19), Evidently drifted share 0.08–0.22, score PSI ≤ 0.006; FPR 0.88% → 1.36%, weekly PR-AUC 0.42–0.69; no retrain trigger at the configured thresholds (1.5× budget, PR-AUC < 0.40). `docs/DRIFT.md` |
