# PLAN.md – AIML-04 Delivery Plan (cloud-first)

Start **2026-11-09**, AIML-04 code-complete **2026-11-21**, both projects submitted by **2026-11-29**
(one day of buffer before the 2026-11-30 portal deadline).

> **Low-load rule:** anything that can run online runs online. The laptop only runs what physically
> can't be done in the cloud: the CARLA simulator, the ROS2 closed loop, and building the TensorRT
> engine for its own GPU. See "Where each task runs" below.

> Constraint: AIML-01 (Fraud Detection) shares the deadline and the single Drive/GitHub submission.
> From 2026-11-18, AIML-04 work runs half-days so AIML-01 can start in parallel.

Legend: `[ ]` todo · `[x]` done · **(S)** stretch goal, skip if behind · ☁ runs online · 💻 runs on the laptop · agent in brackets.

## Where each task runs
| Work | Where | Why |
|---|---|---|
| Dataset download, conversion, EDA | ☁ Kaggle notebook | Data never touches the laptop (saves ~40 GB disk and bandwidth) |
| YOLO training and evaluation | ☁ Kaggle GPU (T4/P100) | The heaviest job, ~12–16 GPU-hours |
| Tracking, lane and depth evaluation | ☁ Kaggle | Needs the datasets, which live on Kaggle |
| ONNX export, TensorRT build, latency benchmarks | ☁ Kaggle T4 | Satisfies the "benchmark before and after" requirement |
| Unit tests and lint | ☁ GitHub Actions (CPU) | Runs on every push |
| Experiment tracking | ☁ Weights & Biases | Already online |
| README, slides, diagrams | ☁ / any | No compute |
| Optional public demo URL | ☁ Hugging Face Space **(S)** | Offline pipeline on uploaded video |
| CARLA simulator + ROS2 closed loop | 💻 laptop, short sessions | CARLA needs a local GPU with a display stack; free cloud GPUs can't host it |
| TensorRT engine for the live demo | 💻 laptop, ~5 min one-off | Engines are tied to the exact GPU they run on |
| Recording sim frames and demo video | 💻 laptop, during the CARLA sessions | Same sessions as above |

**Kaggle budget.** About 30 free GPU-hours per week, with sessions of up to 12 h (check the current limits on your account).
The plan uses about 22 GPU-hours spread across two weeks: training ~15 h, evals ~5 h, export and benchmarks ~2 h.
**Laptop budget.** About 5–6 h of CARLA time in total, in sessions of 30 minutes or less.

## Timeline at a glance
| Milestone | Dates | Where | Agent | Exit criteria |
|---|---|---|---|---|
| M0 Setup | Nov 9–10 | ☁ + 💻 | main + sim-ros-integrator | Kaggle/W&B/GitHub ready, CARLA runs, WSL2 ROS2 talks to it |
| M1 Data | Nov 10–11 | ☁ | data-engineer | BDD subset in YOLO format as a private Kaggle dataset, EDA notebook |
| M2 Detect + Track | Nov 11–14 | ☁ | detection-tracking | Fine-tuned YOLO with mAP logged, ByteTrack video, MOT metrics |
| M3 Lanes | Nov 13–15 | ☁ | lane-depth | Classical + learned lanes and a comparison table |
| M4 Depth + Distance | Nov 15–16 | ☁ | lane-depth | Metric distance with error table vs KITTI GT |
| M5 CARLA + ROS2 | Nov 16–19 | 💻 | sim-ros-integrator | Closed loop: car brakes for obstacles in sim |
| M6 ONNX/TensorRT | Nov 18–20 | ☁ + 💻 | edge-optimizer | Before/after benchmark table, TRT engine running locally |
| M7 Deliverables | Nov 20–22 | ☁ | docs-presenter + qa-reviewer | Demo video, README, diagrams, slides, reflection |
| Submit | Nov 28–29 | — | main | Drive folder + GitHub link submitted once on the portal |

## M0 – Setup (Nov 9–10)
Online:
- [x] M0.1 Create the GitHub repo (public, so Kaggle can clone it without a token) and push this scaffold: github.com/Outsideorbit29/aiml04-perception
- [x] M0.2 Kaggle account: phone verified, 30 h/week GPU quota available (username `anish2906`). The `WANDB_API_KEY` secret is attached per notebook on the first job push (see `kaggle/README.md`).
- [x] M0.3 Kaggle API token saved in `%USERPROFILE%\.kaggle\access_token`. The Kaggle CLI works, and `kaggle/smoke_test` ran on a Tesla T4 on 2026-11-09.
- [ ] M0.4 W&B: `wandb login`, project `aiml04-perception`
- [x] M0.5 GitHub Actions workflow running `pytest` and `ruff` on push
- [x] M0.6 Scaffold the package layout from ARCHITECTURE.md section 5, plus `perception/types.py` with tests and the `kaggle/_template` runner

On the laptop (one short session):
- [ ] M0.7 Download the base CARLA 0.9.15 package only (skip AdditionalMaps, ~8 GB saved), run `CarlaUE4.exe -quality-level=Low -RenderOffScreen` for 2 minutes and check VRAM and temperature
- [ ] M0.8 WSL2 Ubuntu 22.04 + ROS2 Humble (ros-base, not desktop) + `pip install carla==0.9.15`, then run `sim/smoke_test.py` to spawn a car and save one frame
- Exit: `docs/SETUP.md` steps verified end to end.

## M1 – Data (Nov 10–11) ☁ [data-engineer]
- [ ] M1.1 Attach BDD100K and KITTI to a Kaggle notebook as Kaggle datasets (community uploads; check the licence and contents) or download them inside the notebook
- [ ] M1.2 Build the 20k train subset (stratified by weather/time of day), convert to YOLO format, and save it as private Kaggle dataset `aiml04-bdd-yolo`
- [ ] M1.3 EDA notebook: class counts, box sizes, day/night split, sample grids; download only the plots to `docs/figures/`
- [ ] M1.4 Decide class handling (merge rider into person? oversample motorcycle?) and record it in EXPERIMENTS.md
- [ ] M1.5 Tiny fixtures (5 images + labels, a 3-second clip) committed to `tests/fixtures/`. These are the only data files on the laptop.

## M2 – Detection and tracking (Nov 11–14) ☁ [detection-tracking]
- [ ] M2.1 Baseline: COCO-pretrained YOLO11s zero-shot on BDD val, mAP logged
- [ ] M2.2 Fine-tune YOLO11s on Kaggle (640, batch 32 on T4 or P100, AMP, ~50 epochs). Split into 2 sessions with `resume=True`. The checkpoint is saved every epoch as a W&B artifact so a session timeout loses nothing.
- [ ] M2.3 Evaluate: per-class AP, confusion matrix, failure gallery (night, small objects)
- [ ] M2.4 `Detector` class with torch backend and a config switch for onnx/trt (filled in M6)
- [ ] M2.5 ByteTrack integration through the `Tracker` class; annotated video rendered on Kaggle
- [ ] M2.6 MOT eval on a BDD MOT subset (MOTA, IDF1, ID switches)
- [ ] M2.7 **(S)** DeepSORT comparison
- Exit: fine-tuned mAP@0.5 beats the zero-shot baseline, numbers are in EXPERIMENTS.md, and `best.pt` is a W&B artifact.

## M3 – Lane detection (Nov 13–15) ☁ [lane-depth]
- [ ] M3.1 Classical pipeline (ROI, HLS + Sobel, bird's-eye warp, sliding window, polyfit, smoothing). Develop it against the fixtures with CPU-only code.
- [ ] M3.2 Learned pipeline: YOLOP pretrained lane/drivable output in the same `LaneResult` format
- [ ] M3.3 Evaluation on 500 BDD val images on Kaggle: lane IoU/F1 by condition, plus FPS
- [ ] M3.4 Comparison table and a failure-case figure for the report
- [ ] M3.5 **(S)** Fine-tune a YOLO-seg lane model

## M4 – Depth and distance (Nov 15–16) ☁ [lane-depth]
- [ ] M4.1 MiDaS-small and DPT-Hybrid wrappers (`DepthEstimator`)
- [ ] M4.2 Geometric distance from camera intrinsics, height and pitch (`configs/camera.yaml` for KITTI and CARLA)
- [ ] M4.3 Scale alignment plus a per-track Kalman filter, giving distance, closing speed and TTC
- [ ] M4.4 Eval vs KITTI LiDAR on Kaggle: AbsRel and RMSE by distance bin
- [ ] M4.5 `DistanceFusion` produces a `PerceptionFrame` with an `in_ego_lane` flag
- [ ] M4.6 `perception.pipeline` runs the full offline pipeline on a video on Kaggle and writes an annotated MP4

## M5 – CARLA + ROS2 closed loop (Nov 16–19) 💻 [sim-ros-integrator]
Code is written and unit-tested without the simulator. The laptop runs CARLA only in short, planned sessions.
- [ ] M5.1 `perception_msgs` package (TrackedObject, TrackedObjectArray, LaneBoundaries, EgoControl)
- [ ] M5.2 `carla_bridge_node`: synchronous mode, spawn ego + RGB + depth cameras, publish topics, apply control
- [ ] M5.3 `perception_node` wrapping `perception.pipeline` on ONNX Runtime/TensorRT (no PyTorch on the laptop)
- [ ] M5.4 `control_node` with AEB/ACC rules (thresholds in `configs/carla.yaml`)
- [ ] M5.5 Scenarios: stopped car, pedestrian crossing, cut-in, night/rain (Town01/Town02, the lightest maps)
- [ ] M5.6 **Session A (~20 min):** record ~3k CARLA frames with GT, upload them as Kaggle dataset `aiml04-carla-frames`, and run the sim eval (mAP, distance error) on Kaggle
- [ ] M5.7 `stack.launch.py` and rosbag + MP4 recording
- [ ] M5.8 **Sessions B–D (~30 min each):** debug the loop, then record each scenario for the demo
- [ ] M5.9 **(S)** Fine-tune YOLO for 5 epochs on a BDD + CARLA mix, on Kaggle
- Exit: in every scenario the ego car stops before colliding. Screen-record each.

## M6 – ONNX / TensorRT optimization (Nov 18–20) [edge-optimizer]
- [ ] M6.1 ☁ Export YOLO, MiDaS-small and YOLOP to ONNX on Kaggle and verify output parity (max abs diff). The ONNX files are saved as W&B artifacts.
- [ ] M6.2 ☁ Build TensorRT FP16 engines on the Kaggle T4
- [ ] M6.3 ☁ `benchmark.py` on the T4: PyTorch vs ONNX Runtime vs TensorRT, p50/p95, FPS, VRAM, accuracy delta. This is the official before/after table.
- [ ] M6.4 💻 Build the local TensorRT engines from the ONNX files (~5 min, one time), because engines only run on the GPU model they were built for
- [ ] M6.5 💻 Measure end-to-end pipeline FPS during a CARLA session, before and after switching to TRT. TRT also lowers laptop load during the demo.
- [ ] M6.6 **(S)** INT8 calibration on Kaggle
- Exit: `docs/benchmarks.md` has the T4 table plus the laptop end-to-end numbers.

## M7 – Deliverables (Nov 20–22) ☁ [docs-presenter, qa-reviewer]
- [ ] M7.1 `qa-reviewer` full pass: CI green, lint clean, and a fresh clone works in a Kaggle notebook
- [ ] M7.2 README: overview, results table, setup, usage, demo GIF
- [ ] M7.3 Architecture diagram (Mermaid + PNG in `docs/diagrams/`) plus data/API docs for the messages and Python interfaces
- [ ] M7.4 Demo video, 3–4 min, edited from the session recordings: scenarios with overlay, ending on the benchmark table
- [ ] M7.5 10-minute presentation (problem, design, challenges, results, next steps)
- [ ] M7.6 Individual reflection (what I learned, what I owned)
- [ ] M7.7 Drive folder `Anish_Kumar_Gupta_Capstone_Projects/AIML-04/` with code zip, reports, screenshots, sample data, video. Link the full datasets rather than uploading them.
- [ ] M7.8 **(S)** Hugging Face Space running the offline pipeline on an uploaded video, used as the "Live Deployment URL"

## Submit (Nov 28–29)
- [ ] Both subfolders (AIML-04, AIML-01) complete; access set to "Anyone with the link"
- [ ] GitHub URL, deployment/demo URL and Drive URL ready
- [ ] Submit once on the portal (it cannot be resubmitted)

## Daily routine
1. Pick the next unchecked task and delegate to its agent with the milestone ID.
2. Cloud jobs: the agent pushes code to GitHub, then launches the Kaggle notebook with `kaggle kernels push`. Check progress in W&B instead of keeping the laptop busy.
3. Run `qa-reviewer` on the result.
4. Log numbers in `docs/EXPERIMENTS.md`, tick the box, commit.
5. If a task overruns its timebox by more than half a day, drop the related **(S)** items.
