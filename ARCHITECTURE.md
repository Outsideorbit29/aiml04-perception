# ARCHITECTURE.md – AIML-04 Autonomous Driving Perception Stack

## 1. Goal and scope
A **monocular camera** perception stack that, for every frame:
1. detects vehicles, pedestrians, cyclists, traffic lights/signs (fine-tuned YOLO),
2. tracks them across frames with stable IDs (ByteTrack, DeepSORT as comparison),
3. detects ego-lane boundaries (classical OpenCV **and** a segmentation model, compared),
4. estimates the distance to each tracked object (MiDaS monocular depth + geometric scale),
5. runs live in **CARLA** via **ROS2** nodes, and a simple controller brakes or slows for obstacles in the ego lane,
6. is exported to **ONNX** and optimized with **TensorRT**, with latency/FPS benchmarked before and after,
7. logs experiments to **Weights & Biases**.

Out of scope: path planning, LiDAR/radar fusion, full autonomy. The controller is a rule-based
emergency-braking / adaptive-speed demo, enough to show "the car reacting to obstacles".

## 2. System overview

### 2.1 Compute placement (cloud-first)
To keep load on the laptop low, the system is split into an **online offline-ML plane** and a
**local runtime plane**.

| Plane | Runs on | Contains |
|---|---|---|
| Online ML | Kaggle notebooks (T4/P100 GPU), GitHub Actions, W&B | Datasets, EDA, training, all dataset evaluations, ONNX export, T4 TensorRT benchmarks, offline video rendering, CI tests |
| Local runtime | Laptop: CARLA on Windows, ROS2 in WSL2 | Simulator, ROS2 nodes, live inference with ONNX Runtime/TensorRT, local engine build, demo recording |

Artifacts flow through **W&B Artifacts**. Kaggle logs `best.pt` and `*.onnx`. The laptop pulls only
the ONNX files and builds its own TensorRT engines. Sim frames recorded locally are uploaded as a
Kaggle dataset so sim evaluation and fine-tuning also run online. The laptop never holds the
datasets and never trains, and PyTorch is not installed there.

```
GitHub repo ──clone──► Kaggle notebook (GPU) ──metrics/artifacts──► W&B
     ▲                                                                │ ONNX
     │ push                                                           ▼
  laptop  ◄─────────────────────────────── pull ONNX, build TRT, run CARLA + ROS2
     │ upload recorded CARLA frames ──► Kaggle dataset ──► sim eval / fine-tune online
```

### 2.2 Runtime system (local)

```
                          ┌──────────────────── Windows 11 host (GPU) ───────────────────┐
                          │  CARLA 0.9.15 server  (-quality-level=Low, Town03/Town10HD)  │
                          │      ego vehicle + RGB camera (+ depth camera for GT)        │
                          └───────────────┬──────────────────────────────▲───────────────┘
                                          │ TCP 2000-2002 (CARLA client API)
┌───────────────────────────── WSL2 Ubuntu 22.04 · ROS2 Humble ────────┼────────────────┐
│                                                                       │                │
│  carla_bridge_node ──/carla/ego/rgb (Image)──────────┐                │                │
│        │          ──/carla/ego/depth_gt (Image)──┐   │      VehicleControl              │
│        │          ──/carla/ego/odom (Odometry)─┐ │   ▼                │                │
│        │                                       │ │ perception_node    │                │
│        │                                       │ │  ├ Detector (TRT)  │                │
│        │                                       │ │  ├ Tracker         │                │
│        │                                       │ │  ├ LaneDetector    │                │
│        │                                       │ │  ├ DepthEstimator  │                │
│        │                                       │ │  └ DistanceFusion  │                │
│        │                                       │ │   │ /perception/objects           │
│        │                                       │ │   │ /perception/lanes             │
│        │                                       │ │   │ /perception/debug_image       │
│        │                                       ▼ ▼   ▼                │                │
│        │                                  control_node (AEB / ACC rules)               │
│        │◄──────────── /carla/ego/control (custom EgoControl msg) ─────┘                │
│        │                                                                               │
│  recorder_node ── rosbag2 + annotated MP4 for the demo video                           │
│  eval_node ────── compares predicted distance vs depth_gt, logs to W&B                 │
└────────────────────────────────────────────────────────────────────────────────────────┘

Offline path (no ROS, no CARLA):  video / image folder ─► perception.pipeline ─► annotated MP4 + JSON
```

**Why this split.** CARLA's Unreal server needs a local GPU with a display stack, so free cloud
notebooks can't host it. It runs best natively on Windows.
ROS2 Humble is first-class on Ubuntu 22.04, and WSL2 can reach the Windows host over the
virtual network. The `perception/` package has no ROS or CARLA imports, so the same code runs
in unit tests, on dataset videos, and inside the ROS2 node.

## 3. Components

### 3.1 Data layer (`scripts/data/`, Kaggle datasets)
All datasets live on Kaggle, either as attached Kaggle datasets or downloaded inside a notebook.
Converted data is saved as private Kaggle datasets (`aiml04-bdd-yolo`, `aiml04-carla-frames`).
Only `tests/fixtures/` (a few frames) exists on the laptop.

| Dataset | Use | Notes |
|---|---|---|
| **BDD100K** (primary) | Detection fine-tune, lane eval, MOT eval | Use a 20k-image train subset plus the 10k val split to fit the timeline and 6 GB VRAM. Convert to YOLO format. |
| **KITTI** | Distance evaluation | LiDAR gives metric ground truth for the depth/distance module. |
| **CULane / TuSimple** | Lane-model comparison | Only if BDD lane labels prove too noisy. |
| **CARLA synthetic** | Domain adaptation and closed-loop eval | Recorded RGB + depth + semantic GT from the sim, roughly 3–5k frames. |

Classes, mapped from BDD100K: `car, truck, bus, person, rider, bicycle, motorcycle, traffic light, traffic sign`.
EDA output: class histogram, box-size distribution, day/night/weather split, and a decision on
class weighting or oversampling for rare classes (rider, motorcycle).

### 3.2 Detection (`perception/detection/`)
- Ultralytics **YOLO11s** (fall back to YOLOv8s if needed), pretrained on COCO, fine-tuned on BDD100K at 640 px on a **Kaggle T4/P100**, batch 32 with AMP, about 50 epochs over 2 sessions with checkpoint resume.
- `Detector` interface: `detect(frame) -> list[Detection]`. Backends are `torch`, `onnx` (ONNX Runtime) and `tensorrt`, chosen in config.
- Metrics: mAP@0.5 and mAP@0.5:0.95, per-class AP, plus mAP on CARLA frames (sim-to-real gap).

### 3.3 Tracking (`perception/tracking/`)
- **ByteTrack** is the default (`supervision` implementation, so the laptop runtime needs no PyTorch; config in `configs/tracker.yaml`). **DeepSORT** (`deep-sort-realtime`) is the comparison.
- `Tracker.update(detections, frame) -> list[Track]`, which keeps the track id, class, bbox history and age.
- Metrics on KITTI tracking training sequences (BDD100K MOT is not on Kaggle): MOTA, IDF1, ID switches (`motmetrics`, IoU 0.5), plus FPS. See `scripts/eval_mot.py` and `configs/tracker_mot.yaml`.

### 3.4 Lane detection (`perception/lanes/`)
| Approach | Method | Strength |
|---|---|---|
| Classical | ROI mask, HLS/Sobel threshold, perspective warp, sliding window, 2nd-order polyfit, temporal smoothing (undistortion skipped: BDD and CARLA frames ship without distortion parameters) | Fast, explainable; fails on worn markings, curves, night |
| Learned | **YOLOP** (pretrained on BDD100K drivable area + lane) or a YOLO-seg model fine-tuned on BDD lane masks | Robust; heavier |

Common output: `LaneResult(left: Polyline | None, right: Polyline | None, ego_mask, confidence)`. `left`/`right` hold measured boundaries only. When one side is found, `ego_mask` is built from it plus a side synthesized at a nominal lane width (`single_side_fallback`), and `confidence` is lowered. The mask covers only the measured rows; `contains()` uses it there and, outside those rows, the two measured polylines or (one-sided case) the mask edges extended linearly.
Comparison table: lane IoU/F1 on a BDD val subset, failure cases by condition (night, rain, curve), and FPS.

### 3.5 Depth and distance (`perception/depth/`, `perception/fusion/`)
MiDaS outputs **relative** inverse depth, so it needs a metric scale. The design:
1. **Geometric estimate** per object. Using the pinhole model with known camera height `h` and pitch, the distance to the bottom-center of the box is `Z ≈ f·h / (v_bottom − v_horizon)`. It is reliable for objects on a flat road.
2. **MiDaS (DPT-Hybrid or MiDaS-small)** gives a dense relative depth map. The median depth inside each box (lower 50%, to avoid background) is taken.
3. **Scale alignment** fits a per-frame scale and shift between MiDaS values and the geometric estimates of confident, near objects (robust least squares), then applies them to every object. This fixes cases where the geometric estimate breaks (hills, occlusion).
4. **Temporal filter** runs a per-track Kalman filter on distance, which also yields closing speed and time-to-collision `TTC = d / v_close`.

Evaluation: absolute relative error and RMSE against KITTI LiDAR and the CARLA depth camera, reported by distance bin (0–10, 10–30, 30–60 m).

### 3.6 Fusion output (`perception/fusion/`)
`DistanceFusion` merges tracks, distances and lanes into a `PerceptionFrame`. It flags each object
as `in_ego_lane` when the bottom-center of its box falls inside the ego-lane polygon.

### 3.7 Simulation and ROS2 (`sim/`, `ros2_ws/src/`)
Packages:
- `perception_msgs`: custom messages `TrackedObject`, `TrackedObjectArray`, `LaneBoundaries`, `EgoControl`.
- `perception_ros`: nodes `carla_bridge_node`, `perception_node`, `control_node`, `recorder_node`, `eval_node`, plus `stack.launch.py`.

| Topic | Type | Rate |
|---|---|---|
| `/carla/ego/rgb` | `sensor_msgs/Image` (800×450, bgr8) | 20 Hz |
| `/carla/ego/depth_gt` | `sensor_msgs/Image` (32FC1, metres) | 20 Hz |
| `/carla/ego/odom` | `nav_msgs/Odometry` | 20 Hz |
| `/perception/objects` | `perception_msgs/TrackedObjectArray` | per frame |
| `/perception/lanes` | `perception_msgs/LaneBoundaries` | per frame |
| `/perception/debug_image` | `sensor_msgs/Image` (annotated) | per frame |
| `/carla/ego/control` | `perception_msgs/EgoControl` | 20 Hz |

CARLA runs in **synchronous mode** with a fixed 0.05 s step, so perception latency never desyncs
the sim. The bridge ticks the world only after control for the previous frame arrives, or after a
timeout. A Traffic Manager autopilot drives the ego vehicle laterally. `control_node` overrides
throttle/brake longitudinally:
- if a lead object is in the ego lane with TTC < 2.0 s or distance < 8 m: full brake (AEB),
- if TTC is between 2.0 and 4.0 s: proportional slow-down (ACC-like),
- otherwise: autopilot throttle.

Demo scenarios (scripted in `sim/scenarios/`): a stopped car ahead, a pedestrian crossing, cut-in, night/rain.

### 3.8 Optimization (`deploy/`)
- `export.py` (runs on Kaggle) converts PyTorch to ONNX (opset 17, dynamic batch off, 640×640) and then to a TensorRT engine (FP16; INT8 with calibration on 500 BDD images as a stretch goal). It covers YOLO, MiDaS and the lane model.
- `benchmark.py` (runs on the Kaggle T4) does 200 warm-up and 1000 timed iterations per backend and reports p50/p95 latency, FPS, peak VRAM and the accuracy delta (mAP or depth AbsRel) against the PyTorch FP32 baseline. Results go to `docs/benchmarks.md` and W&B. This is the official before/after table.
- `build_local_engine.py` (runs on the laptop, once) builds RTX 3060 engines from the ONNX files, because TensorRT engines only run on the GPU model they were built for. The laptop then measures only end-to-end FPS in CARLA, before and after the switch to TRT.

| Backend | YOLO11s @640 on Kaggle T4 (to be measured) |
|---|---|
| PyTorch FP32 | ~20–30 ms |
| ONNX Runtime CUDA | ~12–18 ms |
| TensorRT FP16 | ~5–8 ms |

### 3.9 Experiment tracking
W&B project `aiml04-perception`, with run groups `detect`, `track`, `lanes`, `depth`, `bench`, `sim-eval`.
Every run logs its config YAML, git SHA, dataset version and final metrics. Model artifacts are
stored as W&B artifacts, not in git.

## 4. Data contracts (`perception/types.py`)
```python
@dataclass
class Detection:      bbox_xyxy: np.ndarray; score: float; cls: int; cls_name: str
@dataclass
class Track:          track_id: int; det: Detection; age: int; history: deque[np.ndarray]
@dataclass
class Polyline:       coeffs: np.ndarray  # x = a*y^2 + b*y + c, image coords
@dataclass
class LaneResult:     left: Polyline | None; right: Polyline | None; ego_mask: np.ndarray | None; confidence: float
@dataclass
class ObjectState:    track: Track; distance_m: float; distance_std: float; closing_speed_mps: float; ttc_s: float | None; in_ego_lane: bool
@dataclass
class PerceptionFrame:stamp: float; frame_id: int; objects: list[ObjectState]; lanes: LaneResult; timings_ms: dict[str, float]
```
Each module exposes a small class with `__init__(cfg: dict)` and one processing method. Backends
can be swapped by config without changing callers.

## 5. Repository layout
```
Project_ai/
├─ CLAUDE.md  AGENTS.md  ARCHITECTURE.md  PLAN.md  README.md
├─ .claude/agents/            # subagent definitions
├─ .github/workflows/ci.yml   # pytest + ruff on every push (online)
├─ kaggle/                    # one folder per cloud job: kernel-metadata.json + runner notebook
│  ├─ data_prep/  train_yolo/  eval_track/  eval_lanes/  eval_depth/  export_bench/  sim_eval/
├─ configs/                   # yolo_bdd.yaml, tracker.yaml, lanes.yaml, depth.yaml, carla.yaml, bench.yaml, camera.yaml
├─ perception/                # ROS-free core library
│  ├─ types.py  pipeline.py
│  ├─ detection/  tracking/  lanes/  depth/  fusion/
│  └─ viz.py                  # overlays for debug image / demo video
├─ scripts/                   # train_yolo.py, eval_*.py, data/ (download, convert, eda)
├─ deploy/                    # export.py, benchmark.py, trt_runtime.py
├─ sim/                       # carla_utils.py, scenarios/, record_dataset.py
├─ ros2_ws/src/
│  ├─ perception_msgs/
│  └─ perception_ros/         # nodes + launch
├─ tests/                     # pytest + fixtures (few frames)
├─ notebooks/                 # eda.ipynb, analysis
├─ docs/                      # PROJECT_BRIEF, SETUP, EXPERIMENTS, benchmarks, SUBMISSION, diagrams/
└─ models/ runs/ media/       # gitignored; local copies of ONNX/TRT and recordings only
```

## 6. Key risks and mitigations
| Risk | Mitigation |
|---|---|
| Laptop load and heat | Cloud-first split (section 2.1). CARLA sessions of 30 minutes or less, Town01/02, Low quality, off-screen rendering, 20 FPS cap |
| 6 GB VRAM shared by CARLA and models | 800×450 camera, TRT FP16 `s` models, MiDaS-small live (DPT-Hybrid offline only), lane model every 2nd frame |
| Kaggle quota or session timeouts | ~22 of ~30 weekly GPU-hours planned across two weeks; checkpoint every epoch to W&B and resume; Colab as a fallback |
| ROS2 on WSL2 networking to Windows CARLA | Use the host IP from `ip route`, open firewall ports 2000–2002, and verify early in M5.1 |
| BDD100K is large (~7 GB images) | 20k-image subset, convert once, cache labels |
| MiDaS scale ambiguity | Geometric scale alignment plus validation against KITTI LiDAR and CARLA depth GT |
| Sim-to-real gap for detection | Fine-tune 5 extra epochs on mixed BDD + CARLA frames and report both |
| Time (two projects, one deadline) | Strict timeboxes in PLAN.md; DeepSORT, INT8 and the YOLO-seg lane model are stretch goals |
