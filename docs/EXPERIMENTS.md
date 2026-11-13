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

KITTI videos are qualitative only (MOT metrics below, M2.6). Fragmentation is visible: seq 0000 has 75 unique ids over 154 frames with ~7 objects active, and 23 tracks are shorter than 3 frames. M2.6 measures this properly (MOTA/IDF1/ID switches); if ID switches are high, try a longer `lost_track_buffer` and a lower activation threshold before trying DeepSORT.

**M2.6 tracker sweep (6 configs, same cached detections).** Requiring 2 matched frames before a track gets an id removes most one-frame ghost tracks: ID switches 310 → 176, unique ids 972 → 591, tracks shorter than 3 frames 220 → 67, IDF1 65.7 → 67.4, MOTA unchanged (52.3 → 52.5). Cost: recall 67.7 → 64.6, and a new id appears 2 frames after first detection (100 ms at 20 Hz in CARLA; the first 2 frames after a reset have no tracks), acceptable for AEB at urban speeds but M5 must not expect tracks on the first frames. Caveats: the config was chosen and scored on the same 8 sequences (in-sample, +1.7 IDF1); buffer 60 is only tested together with consecutive 2 (buffer 60 alone was slightly worse), so the next MOT run adds `consec2_buf30` and pins the old baseline as `baseline_m25`. Longer buffers alone and a lower activation threshold made things slightly worse. By group: vehicles MOTA 63.7 / IDF1 75.7, pedestrians 38.5 / 54.9 (recall 50%; KITTI Cyclist boxes include the bike while BDD rider boxes don't, so part of the pedestrian gap is label mismatch). Ground truth is KITTI tracking, not BDD MOT (not available on Kaggle); classes are mapped in `configs/tracker_mot.yaml`, matching ignores class.

**M2.6 follow-up sweep ([i3c2zmwu](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/i3c2zmwu), 2026-11-13).** Rerun with the pre-M2.6 baseline pinned reproduces both rows exactly (baseline 52.3 / 65.7 / 310, shipped default 52.5 / 67.4 / 176). Consecutive 2 at buffer 30 gives 52.5 / 67.1 / 173, so the gain comes from requiring 2 consecutive frames; buffer 60 vs 30 is within noise and is kept. Sweep entries now apply on top of the new default. The top entry, activation threshold 0.15 (52.7 / 68.0 / 169), is an in-sample +0.6 IDF1, which is too small to change the default.

## Lanes (M3)
| Date | Method | Eval set | IoU | F1 | Night F1 | FPS | W&B |
|---|---|---|---|---|---|---|---|
| 2026-11-13 | Classical (HLS + Sobel, sliding window), `configs/lanes_bdd.yaml` | BDD val 500 (200 day / 200 night / 100 dawn-dusk, seed 42) | 0.423 (measured-only 0.239) | 0.231 | 0.271 | 24.9 (CPU, Kaggle Xeon) | [bf4xp164](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/bf4xp164) |
| 2026-11-13 | YOLOP pretrained (ONNX Runtime FP32), same fitting | same | 0.415 (measured-only 0.224) | **0.515** | 0.493 | 16.3 (T4; model alone 22.9–24.9 ms) | [bf4xp164](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/bf4xp164) |

**How to read the lane numbers (M3.3–M3.4).** IoU = ego-lane mask vs BDD drivable area `direct` over image rows 0.58–0.74; single-side fallback on (the missing boundary is synthesized at 0.37 of the bird's-eye width), and 'measured-only' scores those images as no prediction. F1 = measured boundaries only vs the nearest BDD parallel lane lines, hit if mean error ≤ 20 px. Full table: `docs/figures/lanes_comparison.md`; failure cases: `docs/figures/lanes_failures.jpg`.
- **YOLOP finds the right lines far more often** (F1 0.52 vs 0.23, precision 0.60 vs 0.24). Classical is fooled by car edges, hood reflections and road text.
- **Ego-mask IoU is a tie (~0.42)** because YOLOP returns no line at all on 129/500 images (classical 28/500), mostly faint/worn markings, snow (21/38) and dense traffic. When it predicts, its IoU is higher (0.52 vs 0.45).
- **Night costs YOLOP little** (F1 0.49 vs 0.52 day). Snow is the worst condition for YOLOP (F1 0.37). Curves are weak for both (n = 20).
- **Speed:** classical 40 ms on CPU, YOLOP 61 ms end to end on T4 FP32. Two YOLOP model timings exist: 42.2 ms is `session.run` inside the eval loop (real 1280x720 frames, letterbox, host I/O); 22.9–24.9 ms is a fixed-tensor micro-benchmark (median of 50, with and without CUDA I/O binding). **M6 uses the in-loop 42.2 ms as the YOLOP baseline**, measured with the same harness.
- **Decision:** use YOLOP as the default lane backend for the live pipeline; keep classical as the CPU fallback and for the report comparison. BDD lane width from 279 images is 0.407 W at the band bottom (≈0.34 bird's-eye), slightly below the 0.37 used; leave as is (affects only synthesized sides).
- **Caveats for the report.** The BDD warp (`configs/lanes_bdd.yaml`) and the 0.37 fallback width were tuned on BDD val ground-truth statistics, and part of the 30-image tuning sample lies inside the 500 evaluated images; re-derive on the train split before any retune. About half of the ego masks use a synthesized side (classical 249/500, YOLOP 222/500), so the headline IoU depends on the fallback; the measured-only IoU is the conservative number. F1 counts a predicted line on an image with no scorable GT as a false positive (108 such images), which penalises classical more because it predicts more often.

## Depth / distance (M4)
| Date | Depth model | Scale method | GT | AbsRel 0–10m | AbsRel 10–30m | AbsRel 30–60m | RMSE | W&B |
|---|---|---|---|---|---|---|---|---|
| 2026-11-14 | none (geometric: box bottom + camera height/pitch) | – | KITTI LiDAR, 1000 frames, 2,887–3,008 GT-box objects | 0.085 | 0.151 | 0.234 | 7.58 m (coverage 0.96) | [0zvulu0t](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/0zvulu0t) |
| 2026-11-14 | MiDaS-small | per-frame affine fit in inverse depth on geometric anchors, leave-one-out | KITTI LiDAR, 1000 frames, 2,887–3,008 GT-box objects | 0.207 | 0.206 | 0.302 | 11.78 m (0.88) | [0zvulu0t](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/0zvulu0t) |
| 2026-11-14 | DPT-Hybrid | same | KITTI LiDAR, 1000 frames, 2,887–3,008 GT-box objects | 0.177 | 0.204 | 0.316 | 11.37 m (0.88) | [0zvulu0t](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/0zvulu0t) |
| 2026-11-14 | **Fused geometric + MiDaS-small (live default)** | same | KITTI LiDAR, 1000 frames, 2,887–3,008 GT-box objects | 0.121 | 0.153 | **0.218** | **6.76 m (1.00)** | [0zvulu0t](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/0zvulu0t) |
| 2026-11-14 | Fused geometric + DPT-Hybrid | same | KITTI LiDAR, 1000 frames, 2,887–3,008 GT-box objects | 0.121 | 0.159 | 0.237 | 7.90 m (1.00) | [0zvulu0t](https://wandb.ai/anish-grd2004-vellore-institute-of-technology/aiml04-perception/runs/0zvulu0t) |

**M4.4 notes.** GT is the median LiDAR depth inside the 2D box and the 3D label box (visible surface); scored on KITTI GT boxes so the numbers measure distance only, not detection. LiDAR projection was checked on smoke overlays (points land on poles, guardrails, cars). Overall AbsRel: geometric 0.168, fused MiDaS-small 0.168, MiDaS-small alone 0.236, DPT-Hybrid alone 0.234.
- **Geometry carries the accuracy; MiDaS adds coverage and range.** Fusion ties geometric AbsRel but estimates every object (coverage 1.00 vs 0.96: occluded or truncated boxes with no visible ground contact), lowers RMSE 7.58 → 6.76 m and is best at 30–60 m (0.218 vs 0.234). It is slightly worse at 0–10 m (0.121 vs 0.085), where geometry is already very good.
- **MiDaS-small is the live model.** DPT-Hybrid is no more accurate after alignment and costs 404 ms per frame (2.5 FPS) vs 25 ms (39 FPS) for MiDaS-small on T4. At the live input 256x448, MiDaS-small's model time is 13.8 ms (FP32, PyTorch), the M6 baseline.
- **Caveats:** KITTI camera height/pitch are known exactly, which flatters the geometric method; CARLA and BDD values in `configs/camera.yaml` are placeholders until M5. Kalman/TTC is unit-tested only (KITTI object frames are not sequences); it gets evaluated on CARLA recordings in M5. Full table: `docs/figures/depth_comparison.md`.

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
