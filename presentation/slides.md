---
title: "Autonomous Driving Perception Stack (AIML-04)"
subtitle: "Camera-only detection, tracking, lanes and distance, closed loop in CARLA through ROS2"
author: "[fill in: name], Nexvia Technologies AI/ML internship"
date: "[fill in: presentation date]"
---

<!--
How to use this file
- 14 slides, about 10 minutes. Timing per slide is in each notes block; the total is in the table
  at the end.
- Each slide: title, 3-5 bullets, a "Visual" line (first line of the notes) and speaker notes.
- Every number comes from docs/EXPERIMENTS.md or docs/benchmarks.md. W&B run ids are in the notes
  so a reviewer can look them up. Do not round them further on the slides.
- To make a PPTX (laptop, light; pandoc turns "::: notes" into speaker notes), from the repo root:
    pandoc presentation/slides.md -o presentation/AIML-04_presentation.pptx --slide-level=2
  Then insert the visuals named in each notes block by hand and export a PDF for Drive.
- Visuals that do not exist yet are marked TODO (diagram PNG export, demo video clips).
-->

## 1. Problem and brief

- Goal: a camera-only stack that detects, tracks and measures the distance to road users and finds the ego lane
- It must run inside the CARLA simulator through ROS2 and make the car react to obstacles
- Brief (Nexvia AIML-04): BDD100K/KITTI, fine-tuned YOLO, ByteTrack/DeepSORT, classical vs learned lanes, MiDaS, ONNX + TensorRT, W&B
- Result: in CARLA the car stopped without a collision in all 4 scripted scenarios

::: notes
Visual: title strip plus `docs/screenshots/stopped_car.jpg` (frame 178 of the final Session C recording `media/stopped_car_20261119_013806.mp4`; overlay shows the lead box, distance and ACC mode). Other scenario frames are in `docs/screenshots/`.

Time: 0:45.

The brief asked for a vision pipeline that detects lanes, vehicles and pedestrians, tracks them, estimates distance, and runs inside a driving simulator. It also named the tools: YOLO, OpenCV, ROS2, CARLA, TensorRT, ONNX and Weights and Biases.

I built it as one Python package that has no ROS or CARLA code in it. The same code runs on dataset videos in the cloud and live in CARLA, where ROS2 nodes feed it camera frames and a simple rule-based controller slows down or brakes.

The headline: in the four scripted CARLA scenarios, a stopped car, a pedestrian crossing, a cut-in and a stopped car at night in heavy rain, the ego car stopped every time without a collision. Each scenario was run once, which I come back to under limitations.
:::

## 2. Constraints

- Laptop: RTX 3060 Laptop, 6 GB VRAM; CARLA rendering and the models share the GPU
- Rule: anything that can run online runs online: Kaggle T4 for training and evals, GitHub Actions for tests, W&B for metrics and models
- Laptop only for CARLA + ROS2 (sessions of 30 min or less) and a one-time TensorRT engine build
- Kaggle budget about 30 GPU-hours per week; one deadline shared with a second project

::: notes
Visual: the compute-placement table from ARCHITECTURE.md 2.1, or the cloud half of `docs/diagrams/architecture.mmd` (PNG TODO).

Time: 0:40.

The constraints shaped most of the design. The laptop has a 6 GB GPU, and CARLA rendering takes a large part of it. So no PyTorch in the live loop, small "s" models, and TensorRT FP16 engines.

The second constraint was laptop load. Training, every dataset evaluation, the ONNX export and the T4 benchmark ran as Kaggle jobs launched from the laptop, tests ran in GitHub Actions, and every run is logged to W&B. The laptop never held the datasets. It only ran CARLA and ROS2 in short sessions, and built the TensorRT engines once, because engines only run on the GPU they were built on.

Kaggle gives about 30 GPU-hours a week, so long runs had to fit one session or resume from W&B.
:::

## 3. Architecture

- `perception/`: Detector -> Tracker -> LaneDetector + DepthEstimator -> DistanceFusion -> `PerceptionFrame`
- ROS2 nodes (WSL2): `carla_bridge_node`, `perception_node`, `control_node`, `recorder_node`; CARLA server on Windows
- CARLA in synchronous mode (0.05 s step): the bridge ticks only after the control for the last frame arrives
- Controller: AUTOPILOT / ACC / GAP / AEB / HOLD on a lead that stays in the ego lane for 3 frames

::: notes
Visual: `docs/diagrams/architecture.png` (TODO: export from `docs/diagrams/architecture.mmd` in the Mermaid Live Editor). Optional second slide or inset: `docs/diagrams/pipeline.mmd`.

Time: 1:00.

On the left is the cloud plane: GitHub, Actions, Kaggle and W&B. Kaggle jobs clone the repo at a commit and push metrics, weights and ONNX files to W&B.

On the right is the laptop. CARLA runs on Windows. ROS2 Humble runs in WSL2 with four nodes. The bridge is the only node that talks to CARLA: it publishes the camera image and odometry and applies the control. The perception node wraps the pipeline: YOLO detection, ByteTrack, YOLOP lanes and MiDaS depth every second frame, and distance fusion with a per-track Kalman filter that gives closing speed and time to collision. The control node decides the mode.

CARLA runs in synchronous mode, so slow perception never desyncs the simulation; the cost is that the loop runs slower than real time.
:::

## 4. Data

- BDD100K: 20,000-image train subset (stratified by time of day and weather, seed 42) + full 10,000-image val, converted to YOLO format on Kaggle
- Imbalance: car is 55 % of boxes; motorcycle 239x rarer than car, rider 148x
- Kept all 9 classes; repeat-factor sampling (motorcycle images x1.75, rider x1.36, bicycle x1.24): 20,853 entries
- KITTI: LiDAR for distance ground truth, 8 tracking sequences for MOT; CARLA: 3,000 recorded Town01 frames, 6 weathers

::: notes
Visual: `docs/figures/eda_class_counts.png` (left) and `docs/figures/eda_conditions.png` (right). Backup: `docs/figures/eda_box_sizes.png`.

Time: 0:40.

BDD100K was the main dataset. The full train split is 69,863 images with about 1.27 million boxes. To fit the GPU budget I took a 20,000-image subset, stratified by time of day and weather, and kept the full 10,000-image val split. About 40 % of images are at night and 55 % of boxes are small.

The class balance is very skewed: cars are 55 % of all boxes. I kept all nine classes, because rider versus person matters for driving, and used repeat-factor sampling so images with rare classes appear more often. That added only 853 entries per epoch (20,853 instead of 20,000).

KITTI gave LiDAR ground truth for distance and tracking sequences for MOT. Later I recorded 3,000 CARLA frames with ground truth on the laptop and uploaded them to Kaggle.
:::

## 5. Detection and the sim-to-real gap

- YOLO11s, COCO -> BDD, 50 epochs, 640 px, one 4.6 h T4 session: mAP@0.5 **0.553** (0.309 @0.5:0.95)
- COCO zero-shot on the 7 shared classes 0.361; fine-tuned on those 7 classes 0.560
- On CARLA frames the BDD model scored 0.530 vs zero-shot 0.485: fine-tuning gained at most +0.045 there
- M5.9: 5 more epochs on BDD + CARLA (25 % CARLA): CARLA-val **0.624 -> 0.819**, BDD val 0.553 -> 0.539

::: notes
Visual: a small table (base vs mix on BDD val and CARLA-val) from docs/EXPERIMENTS.md "BDD + CARLA mix fine-tune (M5.9)". Optional: W&B val gallery from run 8tzih6my.

Time: 0:50.

I fine-tuned YOLO11s from COCO weights. On the 10k BDD val images it reaches 0.553 mAP at 0.5 (W&B 6euheeu2, eval 8tzih6my). Compared fairly on the seven classes COCO shares with BDD, that is 0.560 against 0.361 zero-shot. Night costs only about 0.03.

The surprise came in CARLA. On 3,000 recorded frames the BDD model scored 0.530 and plain COCO 0.485, so most of the BDD gain did not transfer to rendered images. That +0.045 is probably an upper bound, because COCO calls CARLA riders "person" and those count as false positives.

So I fine-tuned five more epochs on a mix with 25 % CARLA frames (W&B i6dnlk0c). On 600 held-out CARLA frames mAP went from 0.624 to 0.819; on BDD it dropped 0.014. Caveat: the held-out frames come from the same town and session, so this is an in-domain gain.
:::

## 6. Tracking

- ByteTrack (`supervision`), scored on 8 KITTI tracking sequences (2,358 frames), IoU 0.5
- Default: MOTA 52.3, IDF1 65.7, 310 ID switches -> tuned (2 consecutive frames, buffer 60): **52.5 / 67.4 / 176**
- Vehicles 63.7 / 75.7, pedestrians 38.5 / 54.9 (MOTA / IDF1)
- Cost: recall 67.7 -> 64.6, and a new id appears 2 frames after first detection; detection + tracking 52 FPS on a T4

::: notes
Visual: a two-row table (default vs tuned). Optional: one KITTI tracking frame with ids from the W&B run f4sdcuqk.

Time: 0:40.

BDD's tracking split is not on Kaggle, so I evaluated on eight KITTI tracking sequences (W&B qvrkp7t1). The default ByteTrack produced many one-frame ghost tracks. Requiring two consecutive matched frames before a track gets an id cut ID switches from 310 to 176 and raised IDF1 from 65.7 to 67.4; MOTA barely moved.

The trade-off is a slightly lower recall and a 2-frame delay before a new object gets an id. At CARLA's 20 Hz that is 100 ms, which is fine for braking at urban speeds, but the controller cannot expect tracks on the first frames.

Pedestrians are much weaker than vehicles. Part of that is a label mismatch: KITTI cyclist boxes include the bike, BDD rider boxes do not. The settings were chosen on the same sequences, so these numbers are in-sample.
:::

## 7. Lanes: classical vs learned

- Classical: HLS + Sobel threshold, bird's-eye warp, sliding window, polynomial fit (CPU)
- Learned: pretrained YOLOP lane-line segmentation, same `LaneResult` output
- 500 BDD val images: lane-line F1 **YOLOP 0.515 vs classical 0.231**; ego-lane IoU a tie (0.415 vs 0.423)
- YOLOP finds no line on 129 of 500 images (worn markings, snow, dense traffic); snow F1 0.367

::: notes
Visual: `docs/figures/lanes_failures.jpg`, plus the "all / night / snowy / curve" rows of `docs/figures/lanes_comparison.md`.

Time: 0:45.

The brief asked for a classical lane detector compared with a segmentation model. Both produce the same output: measured left and right boundaries and an ego-lane mask (W&B bf4xp164).

On 500 BDD val images YOLOP finds the right lines far more often: F1 0.515 against 0.231, precision 0.60 against 0.24. The classical method is fooled by car edges, hood reflections and road text.

The ego-lane IoU is a tie at about 0.42, because YOLOP returns no line at all on 129 of 500 images. When it predicts, its IoU is higher, 0.52 against 0.45. Night costs YOLOP little, F1 0.49. Snow is its worst condition and curves are weak for both, on only 20 images.

YOLOP is the live default; classical stays as the CPU fallback.
:::

## 8. Distance: geometry first, MiDaS for coverage

- Geometric: flat-ground distance from the box bottom row, camera height and pitch; MiDaS-small aligned per frame on those anchors; Kalman filter gives closing speed and TTC
- KITTI vs LiDAR (1000 frames): AbsRel geometric 0.168, fused **0.170** with coverage 1.00 vs 0.96; 30-60 m 0.221 vs 0.234
- CARLA GT boxes: geometric 0.040, old fusion 0.087 -> `geometric_first` 0.057 (MiDaS-small is the noisy part in CARLA)
- Lesson: box-bottom calibration belongs to one detector. The BDD-detector fix over-corrected the mix detector; live uses none: **AbsRel 0.081** on CARLA-val

::: notes
Visual: two small tables, KITTI (geometric / MiDaS-small / fused, from `docs/figures/depth_comparison.md`) and CARLA-val (mix uncalibrated vs mix with the BDD calibration vs BDD calibrated).

Time: 0:55.

MiDaS gives relative depth only, so the metric scale comes from geometry: the box bottom touches the road, and with a known camera height and pitch that row gives the distance. MiDaS is aligned to those anchors every frame and fills the gaps.

On KITTI (W&B amgl803i) fusion adds coverage and helps at long range, with the same overall error. One bug I found there: averaging in metres pulled fused distances short, so fusion is done in inverse depth.

CARLA reversed the picture (W&B nfekm9vg). The flat-ground model is near exact, 0.040, and MiDaS made it worse, so CARLA uses geometric-first. Detected boxes read about 10 % long, so I fitted a box-bottom calibration for the BDD detector. When I switched to the mix detector, that calibration over-corrected it on all three hold-outs. The mix detector needs none: 0.081 AbsRel on 600 held-out frames (W&B 7cnlg0dt).
:::

## 9. Optimization: ONNX and TensorRT

| Model (T4, batch 1, p50 ms) | PyTorch FP32 | ONNX Runtime | TensorRT FP16 | Speed-up |
|---|---|---|---|---|
| YOLO11s 640 | 9.93 | 10.79 | **3.66** | 2.72x |
| MiDaS-small 256x448 | 10.95 | 9.04 | **2.58** | 4.25x |
| YOLOP 640 | 22.78 | 20.22 | **6.70** | 3.40x |

- FP16 accuracy cost: YOLO mAP@0.5 -0.0001, MiDaS output AbsRel 0.0096, YOLOP lane-mask IoU 0.9928
- TensorRT also uses the least VRAM: 173-189 MB per model (PyTorch 249-291, ONNX Runtime 383-639)

::: notes
Visual: this table, copied from `docs/benchmarks.md` (W&B 218me22j). Optional: the laptop row "offline TensorRT on 100 CARLA frames: 66.8 FPS stage time, 33.8 FPS wall".

Time: 0:45.

All three models were exported to ONNX on Kaggle and checked for parity against PyTorch, then built into TensorRT FP16 engines. The benchmark is 200 warm-up and 1000 timed iterations, batch 1, on a T4.

TensorRT is 2.7 to 4.3 times faster than PyTorch at model level, uses the least memory, and the accuracy cost is negligible. ONNX Runtime alone is close to PyTorch, and slightly slower for YOLO.

With TensorRT the bottleneck moves to the CPU: for YOLO, letterbox, decode and NMS take about 6 of the 9.6 ms module time. So further speed-ups should come from preprocessing, not the model.

On the laptop with CARLA rendering on the same GPU, the live perception stage averaged 35 to 44 ms per frame over 100-frame windows.
:::

## 10. Closed loop in CARLA (final recording, Session C)

| Scenario | Weather | Stop gap | AEB? |
|---|---|---|---|
| `stopped_car` | ClearNoon | 11.8 m | No (ACC) |
| `cut_in` | ClearNoon | 10.0 m | No (ACC, GAP) |
| `pedestrian_crossing` | ClearNoon | 9.6 m | Yes, TTC 1.49 s from 8.1 m/s |
| `night_rain` | HardRainNight | 11.3 m | No (ACC) |

- No collisions; mix detector as a TensorRT engine, no distance calibration, 10 m minimum gap, 3 s post-stop tail
- Live perception 38.2-49.2 ms per 100-frame window; at night the lead reads ~20 % long (13.6 m shown vs 11.3 m true)

::: notes
Visual: 30-40 s of `media/aiml04_demo.mp4` (built from `media/stopped_car_20261119_013806.mp4`, `cut_in_20261119_014200.mp4`, `pedestrian_crossing_20261119_014523.mp4`, `night_rain_20261119_020707.mp4`). Keep the table visible next to the clip. The published deck (presentation/deck/) already has these numbers.

Time: 1:30 (including the clip).

This is the final closed-loop recording from 19 November. In the stopped-car scenario the car slows down with adaptive cruise control and stops 11.8 m short, without emergency braking. The cut-in car comes in at 18 m and brakes; gap control stops us 10.0 m behind it.

For the pedestrian, who appears suddenly, emergency braking fires at a time to collision of 1.5 seconds from 8.1 m/s and the car stops 9.6 m short; after the walker clears it pulls away slowly.

At night in heavy rain the car stops smoothly 11.3 m short. The lead reads about 20 % long at night, 13.6 m shown against 11.3 m true.

Each scenario was recorded once, in one town. The 38-49 ms is perception time; the synchronous loop runs slower than real time, so the video plays at sim speed.
:::

## 11. Challenges and what went wrong

- ONNX Runtime's CUDA provider would not load in WSL2 (no cuBLAS/cuDNN): detector on the CPU, 130 ms -> went to TensorRT engines
- Kaggle limits: 0.6-0.7 MB/s reads from mounted inputs (copy to local disk first), a hidden torch.hub prompt that hung a job, CUDA OOM in pseudo-labelling
- MiDaS hurt in CARLA (0.040 -> 0.087 on GT boxes); fusion in metres biased KITTI distances short at range
- Calibration was detector-specific; mode chatter in ACC/GAP needed hysteresis, dwell and a rate-limited brake

::: notes
Visual: none, or a two-column "problem -> fix" table. Optional: the 2026-11-18 vs 2026-11-19 `stopped_car` gaps (6.9 m with AEB vs 10.1 m without).

Time: 0:55.

First, the runtime. ONNX Runtime's GPU provider would not load in WSL2, so the first live run had the detector on the CPU at 130 ms. I went straight to TensorRT engines built on the laptop.

Second, Kaggle. Reading the mounted dataset ran at 0.6 to 0.7 MB/s, so jobs copy data to local disk first. A MiDaS torch.hub dependency waited for a trust prompt that a batch job can't answer. Pseudo-labelling ran out of GPU memory until I batched it. [fill in: queue waits or session limits you hit.]

Third, assumptions that the data broke. MiDaS helped on KITTI but made CARLA distances worse. The box-bottom calibration was only right for the detector it was fitted on.

Last, control. The first runs switched between slow-down modes back and forth and ended in emergency braking. Hysteresis, a minimum dwell, a rate-limited brake and a 10 m minimum gap fixed that.
:::

## 12. Results summary

| Area | Result | Data |
|---|---|---|
| Detection | mAP@0.5 0.553 (zero-shot 0.361 on 7 classes) | BDD val 10k |
| Detection in CARLA | 0.819 mix vs 0.624 base | 600 held-out CARLA frames |
| Tracking | MOTA 52.5, IDF1 67.4, 176 ID switches | KITTI, 8 sequences |
| Lanes | F1 YOLOP 0.515 vs classical 0.231 | BDD val 500 |
| Distance | AbsRel 0.170 (KITTI LiDAR); 0.081 (CARLA-val, mix) | KITTI 1000 frames; CARLA 600 |
| TensorRT | 2.72x / 4.25x / 3.40x vs PyTorch | T4, batch 1 |
| Closed loop | 4 of 4 scenarios stopped, no collision | CARLA Town01 |

::: notes
Visual: this table (it mirrors the README results table).

Time: 0:35.

One slide with everything. Detection beats the zero-shot baseline on BDD, and the mix fine-tune fixes most of the sim-to-real gap in CARLA, at a small BDD cost.

Tracking is solid for vehicles and weaker for pedestrians. Learned lanes beat classical lanes clearly on line detection, but miss lines more often in bad conditions.

Distance is about 17 % relative error against LiDAR on KITTI, and about 8 % in CARLA with the mix detector.

TensorRT makes all three models two to four times faster at almost no accuracy cost, and the closed loop stops in all four scenarios.

Every number here has a W&B run behind it, listed in docs/EXPERIMENTS.md.
:::

## 13. Limitations and next steps

- One run per scenario, one town: repeat runs and test a second town
- Night distance read ~20 % long; pedestrian AP in CARLA did not improve (15 val boxes)
- Long-range false leads (KITTI: 47 of 111 lead frames beyond ~24 m) -> tighten the ego corridor
- Log p95 latency live; move preprocessing to the GPU
- Stretch goals not done: DeepSORT comparison, YOLO-seg lanes, INT8, Hugging Face Space

::: notes
Visual: none.

Time: 0:35.

What I would not claim yet. Each scenario ran once in one town, so the next step is repeated runs and a second town for the mix detector, whose CARLA gain is in-domain.

At night in rain the distance reads long, and pedestrians in CARLA did not get better with the mix fine-tune, though there were only 15 pedestrian boxes in the held-out set.

On KITTI, many lead frames at long range were cars at the kerb that the lane mask let in; the corridor needs tightening beyond about 24 m.

Latency: I log the mean live pipeline time but not p95, and the TensorRT module time is now limited by CPU preprocessing.

DeepSORT, the YOLO-seg lane model, INT8 and a hosted demo were stretch goals I did not do.
:::

## 14. Thank you / questions

- Repo: [fill in: GitHub URL]
- Demo video: [fill in: link]
- Experiments: W&B project `aiml04-perception`; `docs/EXPERIMENTS.md`, `docs/benchmarks.md`

::: notes
Visual: the architecture diagram again, small, for questions.

Time: 0:10.

Likely questions and where the answer is:
- "Why not DeepSORT?" Stretch goal; ByteTrack needs no appearance model and no PyTorch on the laptop.
- "Why is the CARLA gain an upper bound?" Same town, same session, held-out route blocks (EXPERIMENTS.md M5.9 caveats).
- "Why no calibration live?" Mix detector refit is ~identity; the BDD values make it worse (decisions log, 2026-11-19).
- "Real-time?" Perception 35-44 ms per frame; the synchronous loop runs at 4.9 sim steps/s.
:::

<!--
Timing plan (target 10:00)
| Slide | Time | Outline item in docs/SUBMISSION.md |
|---|---|---|
| 1 Problem | 0:45 | 1 Problem and goal |
| 2 Constraints | 0:40 | 2 Architecture |
| 3 Architecture | 1:00 | 2 Architecture |
| 4 Data | 0:40 | 3 Detection |
| 5 Detection | 0:50 | 3 Detection |
| 6 Tracking | 0:40 | 3 Tracking |
| 7 Lanes | 0:45 | 4 Lanes |
| 8 Distance | 0:55 | 5 Depth and distance |
| 9 Optimization | 0:45 | 7 TensorRT |
| 10 Closed loop | 1:30 | 6 CARLA demo clip |
| 11 Challenges | 0:55 | 8 Challenges |
| 12 Results | 0:35 | 8 Learnings |
| 13 Next steps | 0:35 | 8 Next steps |
| 14 Questions | 0:10 | - |
| Total | 10:25 | Cut slide 2 or 12 to 0:20 if rehearsal runs long |
-->
