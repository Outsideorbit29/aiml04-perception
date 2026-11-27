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
