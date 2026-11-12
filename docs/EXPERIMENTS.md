# EXPERIMENTS.md – Experiment log

Add one row per run as soon as it finishes. W&B project: `aiml04-perception`.

## Detection (M2)
| Date | Run | Model | Data | Epochs | mAP@0.5 | mAP@0.5:0.95 | W&B | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | data-check (sanity only) | YOLO11n COCO | 1% of BDD 20k subset, imgsz 320 | 1 | 0.085 | 0.049 | – | Pipeline check only, not a result. Proves `configs/data_bdd.yaml` trains and validates on all 10k val images. |
| 2026-11-09 | train-yolo smoke (sanity only) | YOLO11s COCO → BDD | 640 train imgs (668 with RFS), 500 val, imgsz 320 | 1 | 0.016 | 0.006 | [njln4om8](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/njln4om8) | Job check only (3.6 min). Zero-shot on the same 500 val images at 640: mAP@0.5 0.356 (7 shared classes). Staging 60k files to /kaggle/temp took ~140 s. |
| 2026-11-09 | baseline-zeroshot (M2.1) | YOLO11s COCO | BDD val 10k, imgsz 640 | 0 | 0.361 | 0.210 | [6euheeu2](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/6euheeu2) | 7 shared classes only (rider scored as person, traffic sign excluded). Per class AP50: car 0.578, person 0.492, bus 0.375, bicycle 0.317, truck 0.315, motorcycle 0.272, traffic light 0.179. |
| 2026-11-09 | yolo11s-bdd (M2.2) | YOLO11s COCO → BDD | 20k subset + RFS (20,853 entries), BDD val 10k, imgsz 640 | 50 | **0.553** | **0.309** | [6euheeu2](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/6euheeu2) | One Kaggle T4 session, 4.6 h, no resume needed. On the 7 shared classes: 0.560 / 0.321 (+0.20 mAP@0.5 over zero-shot). Per class AP50: car 0.775, sign 0.630, light 0.596, person 0.587, truck 0.582, bus 0.572, rider 0.426, bicycle 0.425, motorcycle 0.383. Weights: W&B artifact `yolo11s-bdd-best:candidate`. |
| 2026-11-12 | yolo11s-bdd-eval (M2.3) | YOLO11s BDD (`yolo11s-bdd-best:candidate`) | BDD val 10k, imgsz 640 | – | 0.553 | 0.309 | [8tzih6my](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/8tzih6my) | Re-evaluation matches training exactly. Day 0.559 / night 0.533 / dawn-dusk 0.568 mAP@0.5, so night costs only ~0.03. Recall 0.67, precision 0.71 overall; lowest recall: motorcycle 0.38, bicycle 0.39, rider 0.39. Main confusions: truck↔car, traffic light↔sign. Detector-only latency on T4 (torch, batch 1): 11.1 ms mean, 90 FPS. Galleries (night, small, occlusion) in W&B. |

## Tracking (M2)
| Date | Tracker | Detector | Clip set | MOTA | IDF1 | ID sw | FPS | W&B |
|---|---|---|---|---|---|---|---|---|
| 2026-11-13 | ByteTrack default (buffer 30, consecutive 1) | YOLO11s BDD, agnostic NMS | KITTI tracking 0000/0001/0004/0011/0013/0015/0016/0017 (2,358 frames), IoU 0.5 | 52.3 | 65.7 | 310 | 52 (T4) | [qvrkp7t1](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/qvrkp7t1) |
| 2026-11-13 | **ByteTrack tuned (buffer 60, consecutive 2) — new default** | same | same | **52.5** | **67.4** | **176** | 52 (T4) | [qvrkp7t1](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/qvrkp7t1) |
| 2026-11-12 | ByteTrack (supervision) | YOLO11s BDD, agnostic NMS | KITTI tracking 0000, 0001, 0011 (754 frames) | – (M2.6) | – (M2.6) | – | 52 (det 15.9 ms + track 3.2 ms, T4) | [f4sdcuqk](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/f4sdcuqk) |

KITTI videos are qualitative only (no MOT metrics yet). Fragmentation is visible: seq 0000 has 75 unique ids over 154 frames with ~7 objects active, and 23 tracks are shorter than 3 frames. M2.6 measures this properly (MOTA/IDF1/ID switches); if ID switches are high, try a longer `lost_track_buffer` and a lower activation threshold before trying DeepSORT.

**M2.6 tracker sweep (6 configs, same cached detections).** Requiring 2 matched frames before a track gets an id removes most one-frame ghost tracks: ID switches 310 → 176, unique ids 972 → 591, tracks shorter than 3 frames 220 → 67, IDF1 65.7 → 67.4, MOTA unchanged (52.3 → 52.5). Cost: recall 67.7 → 64.6 and a new object appears one frame later (50 ms at 20 Hz in CARLA), acceptable for AEB at urban speeds. Longer buffers alone and a lower activation threshold made things slightly worse. By group: vehicles MOTA 63.7 / IDF1 75.7, pedestrians 38.5 / 54.9 (recall 50%; KITTI Cyclist boxes include the bike while BDD rider boxes don't, so part of the pedestrian gap is label mismatch). Ground truth is KITTI tracking, not BDD MOT (not available on Kaggle); classes are mapped in `configs/tracker_mot.yaml`, matching ignores class.

## Lanes (M3)
| Date | Method | Eval set | IoU | F1 | Night F1 | FPS | W&B |
|---|---|---|---|---|---|---|---|

## Depth / distance (M4)
| Date | Depth model | Scale method | GT | AbsRel 0–10m | AbsRel 10–30m | AbsRel 30–60m | RMSE | W&B |
|---|---|---|---|---|---|---|---|---|

## Closed loop in CARLA (M5)
| Date | Scenario | Weather | Stopped safely? | Min gap (m) | Pipeline FPS | Video |
|---|---|---|---|---|---|---|

## Data (M1)
Full numbers: `docs/figures/eda_summary.md`; plots: `docs/figures/eda_*.png`.

| Split | Images | Boxes | Boxes/img (mean) | Night share | Small boxes |
|---|---|---|---|---|---|
| BDD train (all) | 69,863 | 1,273,579 | 18.2 | 40% | 55% |
| BDD train subset (seed 42) | 20,000 | 364,698 | 18.2 | 40% | 55% |
| BDD val | 10,000 | 185,930 | 18.6 | 39% | 55% |

## Decisions log
| Date | Decision | Why |
|---|---|---|
| 2026-11-09 | **Datasets:** BDD100K from Kaggle `awsaf49/bdd100k-dataset` (100k images + det_20 labels); weather/timeofday from `solesensei/solesensei_bdd100k`; KITTI from `garymk/kitti-3d-object-detection-dataset` (image_2, calib, label_2, velodyne; 7,481 frames); fixture clip from `leducnhuan/kitti-tracking`. | All on Kaggle, so nothing is downloaded to the laptop. KITTI upload includes velodyne, needed for metric distance GT in M4. Licences: BDD100K research/non-commercial (Kaggle's CC0 label is wrong), KITTI CC BY-NC-SA 3.0. Fine for an internship project; cite both. |
| 2026-11-09 | **Data hand-off:** later jobs attach the data-prep kernel output (`kernel_sources: anish2906/aiml04-data-prep`, mounted at `/kaggle/input/notebooks/anish2906/aiml04-data-prep/bdd_yolo`) instead of a separate `aiml04-bdd-yolo` dataset. | Kernels have no Kaggle API credentials to create datasets; kernel chaining needs no extra upload. Path confirmed by the data-check job. |
| 2026-11-09 | **Class handling:** keep all 9 classes (no merging). No per-class loss weights. For M2, use repeat-factor sampling (LVIS style, threshold t = 0.1 on image frequency): images containing motorcycle repeat ≈1.75×, rider ≈1.36×, bicycle ≈1.24×; others 1×. Implemented as a train image list file with duplicated entries. | Car is 55% of boxes; motorcycle is 239× rarer than car, rider 148×, bicycle 96× (subset numbers). Rider vs person matters for driving behaviour, so merging loses information. Ultralytics has no native per-class loss weights; repeat sampling is the standard fix and only adds ~6% more images per epoch (~1.2k of 20k). Re-check per-class AP after M2.2; if rider/motorcycle AP stays far below others, raise t to 0.2. |
| 2026-11-12 | **Keep the M2.2 weights; no RFS retrain.** Rider (0.43) and motorcycle (0.38) AP50 trail car (0.78) but are close to bicycle and person-level classes, not "far below". | A retrain at t = 0.2 costs ~4.6 GPU-hours for an uncertain gain, and the deadline is shared with AIML-01. Revisit only if M5 sim eval shows rider/motorcycle misses. |
| 2026-11-09 | **Small objects:** 89% of traffic lights and 76% of traffic signs are small (<32² px at 1280×720). Train at imgsz 640 first; try 960 only if time allows. | At 640 those objects shrink to a few pixels, so their AP will be low. 960 costs ~2.2× GPU time per epoch, which the Kaggle budget can't afford by default. |
| 2026-11-09 | **M2 must copy data to local disk first:** the training job copies `bdd_yolo` from `/kaggle/input/...` to `/kaggle/temp` before training. | The data-check job measured 0.6–0.7 MB/s reads from the mounted kernel output and could not write the label cache there. Training 50 epochs from the mount would waste most of the GPU budget on I/O. |
| 2026-11-09 | **KITTI depth-eval pairs (old M1 scope) moved to M4.4** (lane-depth agent). | Only M4 uses them; `configs/data_kitti.yaml` already records the paths. |
