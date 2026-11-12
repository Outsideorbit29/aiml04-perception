# EXPERIMENTS.md – Experiment log

Add one row per run as soon as it finishes. W&B project: `aiml04-perception`.

## Detection (M2)
| Date | Run | Model | Data | Epochs | mAP@0.5 | mAP@0.5:0.95 | W&B | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | data-check (sanity only) | YOLO11n COCO | 1% of BDD 20k subset, imgsz 320 | 1 | 0.085 | 0.049 | – | Pipeline check only, not a result. Proves `configs/data_bdd.yaml` trains and validates on all 10k val images. |
| 2026-11-09 | train-yolo smoke (sanity only) | YOLO11s COCO → BDD | 640 train imgs (668 with RFS), 500 val, imgsz 320 | 1 | 0.016 | 0.006 | [njln4om8](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/njln4om8) | Job check only (3.6 min). Zero-shot on the same 500 val images at 640: mAP@0.5 0.356 (7 shared classes). Staging 60k files to /kaggle/temp took ~140 s. |
| 2026-11-09 | baseline-zeroshot (M2.1) | YOLO11s COCO | BDD val 10k, imgsz 640 | 0 | 0.361 | 0.210 | [6euheeu2](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/6euheeu2) | 7 shared classes only (rider scored as person, traffic sign excluded). Per class AP50: car 0.578, person 0.492, bus 0.375, bicycle 0.317, truck 0.315, motorcycle 0.272, traffic light 0.179. |
| 2026-11-09 | yolo11s-bdd (M2.2) | YOLO11s COCO → BDD | 20k subset + RFS (20,853 entries), BDD val 10k, imgsz 640 | 50 | **0.553** | **0.309** | [6euheeu2](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/6euheeu2) | One Kaggle T4 session, 4.6 h, no resume needed. On the 7 shared classes: 0.560 / 0.321 (+0.20 mAP@0.5 over zero-shot). Per class AP50: car 0.775, sign 0.630, light 0.596, person 0.587, truck 0.582, bus 0.572, rider 0.426, bicycle 0.425, motorcycle 0.383. Weights: W&B artifact `yolo11s-bdd-best:candidate`. |

## Tracking (M2)
| Date | Tracker | Detector | Clip set | MOTA | IDF1 | ID sw | FPS | W&B |
|---|---|---|---|---|---|---|---|---|

## Lanes (M3)
| Date | Method | Eval set | IoU | F1 | Night F1 | FPS | W&B |
|---|---|---|---|---|---|---|---|
| 2026-11-13 | Classical (HLS + Sobel, sliding window), `configs/lanes_bdd.yaml` | BDD val 500 (200 day / 200 night / 100 dawn-dusk, seed 42) | 0.423 (measured-only 0.239) | 0.231 | 0.271 | 24.9 (CPU, Kaggle Xeon) | [bf4xp164](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/bf4xp164) |
| 2026-11-13 | YOLOP pretrained (ONNX Runtime FP32), same fitting | same | 0.415 (measured-only 0.224) | **0.515** | 0.493 | 16.3 (T4; model alone 22.9–24.9 ms) | [bf4xp164](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/bf4xp164) |

**How to read the lane numbers (M3.3–M3.4).** IoU = ego-lane mask vs BDD drivable area `direct` over image rows 0.58–0.74; single-side fallback on (the missing boundary is synthesized at 0.37 of the bird's-eye width), and 'measured-only' scores those images as no prediction. F1 = measured boundaries only vs the nearest BDD parallel lane lines, hit if mean error ≤ 20 px. Full table: `docs/figures/lanes_comparison.md`; failure cases: `docs/figures/lanes_failures.jpg`.
- **YOLOP finds the right lines far more often** (F1 0.52 vs 0.23, precision 0.60 vs 0.24). Classical is fooled by car edges, hood reflections and road text.
- **Ego-mask IoU is a tie (~0.42)** because YOLOP returns no line at all on 129/500 images (classical 28/500), mostly faint/worn markings, snow (21/38) and dense traffic. When it predicts, its IoU is higher (0.52 vs 0.45).
- **Night costs YOLOP little** (F1 0.49 vs 0.52 day). Snow is the worst condition for YOLOP (F1 0.37). Curves are weak for both (n = 20).
- **Speed:** classical 40 ms on CPU, YOLOP 61 ms end to end on T4 FP32 (42 ms in session.run, ~15 ms fit). M6 (TensorRT FP16) targets the YOLOP model time.
- **Decision:** use YOLOP as the default lane backend for the live pipeline; keep classical as the CPU fallback and for the report comparison. BDD lane width from 279 images is 0.407 W at the band bottom (≈0.34 bird's-eye), slightly below the 0.37 used; leave as is (affects only synthesized sides).

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
