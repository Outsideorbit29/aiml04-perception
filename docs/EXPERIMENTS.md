# EXPERIMENTS.md – Experiment log

Add one row per run as soon as it finishes. W&B project: `aiml04-perception`.

## Detection (M2)
| Date | Run | Model | Data | Epochs | mAP@0.5 | mAP@0.5:0.95 | W&B | Notes |
|---|---|---|---|---|---|---|---|---|
| | baseline-zeroshot | YOLO11s COCO | BDD val | 0 | | | | |

## Tracking (M2)
| Date | Tracker | Detector | Clip set | MOTA | IDF1 | ID sw | FPS | W&B |
|---|---|---|---|---|---|---|---|---|

## Lanes (M3)
| Date | Method | Eval set | IoU | F1 | Night F1 | FPS | W&B |
|---|---|---|---|---|---|---|---|

## Depth / distance (M4)
| Date | Depth model | Scale method | GT | AbsRel 0–10m | AbsRel 10–30m | AbsRel 30–60m | RMSE | W&B |
|---|---|---|---|---|---|---|---|---|

## Closed loop in CARLA (M5)
| Date | Scenario | Weather | Stopped safely? | Min gap (m) | Pipeline FPS | Video |
|---|---|---|---|---|---|---|

## Decisions log
| Date | Decision | Why |
|---|---|---|
