# API.md – Interfaces of the AIML-04 perception stack

This page documents the three interfaces of the stack: the ROS2 topics and messages, the Python
classes in `perception/`, and the config files. It was written by reading the code on 2026-11-18
(`ros2_ws/src/perception_msgs/msg/*.msg`, `ros2_ws/src/perception_ros/perception_ros/*.py`,
`perception/**/*.py`, `configs/*.yaml`). Where the code and ARCHITECTURE.md differ, the code is
described and the difference is listed at the end.

Diagrams: [docs/diagrams/](diagrams/README.md).

## 1. ROS2 interface

Packages in `ros2_ws/src/`:
- `perception_msgs`: the custom messages `TrackedObject`, `TrackedObjectArray`,
  `LaneBoundaries`, `EgoControl`.
- `perception_ros`: the nodes `carla_bridge_node`, `perception_node`, `control_node`,
  `recorder_node` and the launch file `launch/stack.launch.py`.

Every node takes one ROS parameter, `config` (default `configs/carla.yaml`), and reads its own
section of that file. The bridge also takes `carla_host`, `carla_port`, `town`, `load_town`; the
recorder takes `name` (video file prefix).

### 1.1 Topics

| Topic | Type | Publisher | Subscribers | Rate | QoS |
|---|---|---|---|---|---|
| `/carla/ego/rgb` | `sensor_msgs/Image`, 800x450 `bgr8` | carla_bridge_node | perception_node | one per sim step (20 Hz sim time) | reliable, depth 2 |
| `/carla/ego/depth_gt` | `sensor_msgs/Image`, `32FC1`, metres | carla_bridge_node, only if `depth_camera.enabled` (default false) | none in the live stack | one per sim step | reliable, depth 2 |
| `/carla/ego/odom` | `nav_msgs/Odometry` | carla_bridge_node | control_node | one per sim step | depth 10 |
| `/perception/objects` | `perception_msgs/TrackedObjectArray` | perception_node | control_node | one per RGB frame | depth 10 |
| `/perception/lanes` | `perception_msgs/LaneBoundaries` | perception_node | none (recorded with `record_bag:=true`) | one per RGB frame | depth 10 |
| `/perception/debug_image` | `sensor_msgs/Image`, `bgr8`, annotated | perception_node (if `perception.publish_debug`, every `debug_every_n` frames) | recorder_node | one per RGB frame by default | reliable, depth 1 |
| `/carla/ego/control` | `perception_msgs/EgoControl` | control_node | carla_bridge_node, recorder_node | one per objects message | depth 10 |

**Timing.** CARLA runs in synchronous mode with a 0.05 s step, so every topic is 20 Hz in
simulation time. In wall time the loop is slower, because the bridge ticks the world only after
`/carla/ego/control` for the current frame arrives (or after `bridge.control_timeout_s`). With
CARLA and the TensorRT models on the same laptop GPU it ran at 4.9 sim steps/s
(docs/EXPERIMENTS.md, "Closed loop in CARLA").

**Frame matching.** All messages of one frame carry the same `header.stamp`, the simulation time
of the camera image. perception_node copies the RGB header to its outputs, control_node copies the
objects header to its `EgoControl`, and the bridge waits for a control with that stamp. Before the
first control arrives (models still loading), the bridge republishes the frame every
`bridge.republish_s` seconds; perception_node drops frames whose stamp is not newer than the last
one it processed.

**Odometry fields used.** `header.frame_id` = `bridge.odom_frame` (`map`),
`child_frame_id` = `bridge.base_frame` (`ego`), `pose.pose` = ego pose converted from CARLA to
ROS axes, `twist.twist.linear.x` = forward speed in m/s, `twist.twist.angular.z` = yaw rate.
control_node reads only `twist.twist.linear.x`, and uses 0 if odometry is older than
`control.odom_timeout_s`.

### 1.2 Messages

**`TrackedObject.msg`** – one fused object (`perception.types.ObjectState` with its `Track` and
`Detection`; field-for-field `perception.pipeline.object_to_dict`).

| Field | Type | Meaning |
|---|---|---|
| `track_id` | int32 | ByteTrack id |
| `cls` | int32 | class index in `configs/data_bdd.yaml` order |
| `cls_name` | string | class name (`car`, `person`, ...) |
| `score` | float32 | detector confidence |
| `bbox_xyxy` | float32[4] | x1, y1, x2, y2 in RGB image pixels |
| `age` | int32 | frames since the track was first output |
| `distance_m` | float32 | Kalman-filtered longitudinal distance to the ground-contact point |
| `distance_std` | float32 | 1-sigma of `distance_m`, metres |
| `closing_speed_mps` | float32 | positive when the object gets closer |
| `has_ttc` | bool | false when `ObjectState.ttc_s` is None (not closing significantly, or track too young) |
| `ttc_s` | float32 | time to collision in seconds, NaN when `has_ttc` is false |
| `in_ego_lane` | bool | object is in the ego lane (lane test plus metric lateral corridor) |

**`TrackedObjectArray.msg`** – `perception.types.PerceptionFrame` without the lanes.

| Field | Type | Meaning |
|---|---|---|
| `header` | std_msgs/Header | copied from the `/carla/ego/rgb` image |
| `frame_id` | int64 | `PerceptionFrame.frame_id` (perception_node's frame counter) |
| `objects` | TrackedObject[] | all objects with a distance estimate |
| `lead_track_id` | int32 | `PerceptionFrame.lead_object()`: closest in-ego-lane object this frame, -1 if none. This is not the controller's confirmed lead (see `EgoControl.lead_track_id`). |
| `total_ms` | float32 | pipeline time of this frame (`timings_ms["total_ms"]`) |

**`LaneBoundaries.msg`** – `perception.types.LaneResult`. Boundaries are `x = a*y^2 + b*y + c`
in image pixels. The ego mask itself is not sent.

| Field | Type | Meaning |
|---|---|---|
| `header` | std_msgs/Header | copied from the RGB image |
| `has_left`, `has_right` | bool | a measured boundary exists on that side |
| `left_coeffs`, `right_coeffs` | float32[3] | a, b, c; zeros when the side is missing |
| `confidence` | float32 | `LaneResult.confidence` |
| `has_ego_mask` | bool | an ego-lane mask was built (it may come from one measured side) |
| `y_top`, `y_bottom` | float32 | image rows the fit measured, NaN if unknown |

**`EgoControl.msg`** – longitudinal command from control_node to the bridge.

| Field | Type | Meaning |
|---|---|---|
| `header` | std_msgs/Header | stamp of the `/perception/objects` message it reacts to |
| `override_autopilot` | bool | false: the Traffic Manager autopilot keeps throttle/brake |
| `throttle`, `brake` | float32 | [0, 1], applied only when `override_autopilot` is true |
| `mode` | string | `AUTOPILOT`, `ACC`, `GAP`, `AEB` or `HOLD` (`perception.control.MODES`) |
| `reason` | string | short text explaining the decision |
| `lead_track_id` | int32 | confirmed lead (after the persistence rule), -1 if none |
| `lead_distance_m` | float32 | NaN if no confirmed lead |
| `lead_ttc_s` | float32 | NaN if no lead or no TTC |
| `ego_speed_mps` | float32 | ego speed used for the decision |

### 1.3 Nodes

| Node | Reads (config section) | Does |
|---|---|---|
| `carla_bridge_node` | `carla`, `traffic_manager`, `ego`, `camera`, `depth_camera`, `bridge` | Connects to CARLA, sets synchronous mode, spawns the ego and camera(s), runs the Traffic Manager autopilot for steering (told to ignore vehicles, walkers, lights and signs, so only perception stops the car). Per step: tick, publish odom, depth GT and RGB, wait for control, apply it. An `override_autopilot` control switches the autopilot off and applies throttle/brake with the last steering angle; a non-override control switches it back on. On late control, `bridge.on_control_timeout` is `hold_last` or `autopilot`. Restores asynchronous mode and destroys its actors on shutdown. |
| `perception_node` | `perception` (+ the file named in `perception.pipeline_config`) | Builds `Pipeline.from_config` with the configured backend (TensorRT on the laptop), runs `Pipeline.process` on each new RGB frame, publishes objects, lanes and the annotated debug image. Logs the mean pipeline time every 100 frames. |
| `control_node` | `control` | Converts each `TrackedObjectArray` to `LeadCandidate`s, calls `LongitudinalController.update` with the odometry speed, publishes one `EgoControl` per objects message, logs mode changes. |
| `recorder_node` | `recorder` | Writes `/perception/debug_image` to `media/<name>_<time>.mp4` at `recorder.fps`; with `recorder.overlay_control` it draws the latest mode, brake, ego speed and lead. |

### 1.4 Launch file

```bash
# WSL2, CARLA already running on Windows
ros2 launch perception_ros stack.launch.py carla_host:=localhost town:=Town01 load_town:=true scenario:=stopped_car
```

| Argument | Default | Meaning |
|---|---|---|
| `carla_host`, `carla_port` | `localhost`, `''` | CARLA server; empty uses `configs/carla.yaml` |
| `town`, `load_town` | `''`, `false` | map, and whether to load it if another is active |
| `config` | `configs/carla.yaml` | config file for every node (relative to the repo root) |
| `scenario` | `''` | `stopped_car`, `pedestrian_crossing`, `cut_in`, `night_rain`, or empty for free driving. Runs `python3 -m sim.scenarios.run`; the launch shuts down when it exits. |
| `record` | `true` | start recorder_node |
| `record_bag` | `false` | run `ros2 bag record` on `recorder.bag_topics` into `runs/bags/` |
| `repo_root` | `''` | repo checkout; default `$AIML04_REPO_ROOT` or auto-detected |

The launch file sets `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and `ROS_LOCALHOST_ONLY=1` for its
nodes unless the shell already sets them (needed with WSL2 mirrored networking, docs/SETUP.md 6).

## 2. Python interfaces (`perception/`)

The package has no ROS or CARLA imports. Each module has a class built from a config dict with
one processing method; importing a module never imports torch (heavy backends load lazily).

### 2.1 Data contracts (`perception/types.py`)

| Dataclass | Fields | Methods |
|---|---|---|
| `Detection` | `bbox_xyxy: np.ndarray` (4, float32, pixels), `score: float`, `cls: int`, `cls_name: str` | `bottom_center -> (x, y)`: road-contact point used for distance and lane tests |
| `Track` | `track_id: int`, `det: Detection`, `age: int = 0`, `history: deque[np.ndarray]` (last boxes, maxlen 30 by default) | – |
| `Polyline` | `coeffs: np.ndarray` (3,): a, b, c of `x = a*y^2 + b*y + c` | `x_at(y)` |
| `LaneResult` | `left: Polyline \| None`, `right: Polyline \| None` (measured sides only), `ego_mask: np.ndarray \| None` (HxW bool), `confidence: float` | `contains(x, y) -> bool`: mask inside its measured rows; outside them the two polylines, or the mask edges extended linearly when only one side was measured |
| `ObjectState` | `track: Track`, `distance_m: float`, `distance_std: float`, `closing_speed_mps: float` (positive = closer), `ttc_s: float \| None`, `in_ego_lane: bool` | – |
| `PerceptionFrame` | `stamp: float` (s), `frame_id: int`, `objects: list[ObjectState]`, `lanes: LaneResult`, `timings_ms: dict[str, float]` | `lead_object() -> ObjectState \| None`: closest object with `in_ego_lane` |

`timings_ms` keys: `detect_ms`, `track_ms`, `lanes_ms`, `depth_ms`, `fusion_ms`, `total_ms`.

### 2.2 Detection (`perception/detection/`)

```python
from perception.detection import Detector, load_config
det = Detector(load_config("configs/yolo_detector.yaml"))
dets = det.detect(frame_bgr)            # HxWx3 uint8 BGR -> list[Detection]
```
- `Detector(cfg, backend=None)`. Required keys: `backend`, `weights`, `imgsz`, `conf`, `iou`.
  Optional: `max_det`, `device`, `half`, `classes`, `agnostic_nms`, `names`, `data_config`,
  `onnx_providers`.
- Backends (`BACKENDS`): `torch` (Ultralytics on `.pt`), `onnx` (ONNX Runtime), `tensorrt`
  (`.engine`). `weights` may keep the `.pt` name; the file with the backend's suffix next to it is
  used. ONNX and TensorRT share the same letterbox, decode, NMS and rescale code
  (`perception/detection/processing.py`).
- `load_config(path, **overrides)` reads the YAML and applies overrides.

### 2.3 Tracking (`perception/tracking/`)

```python
from perception.tracking import Tracker
tracker = Tracker(cfg)                  # cfg = configs/tracker.yaml as a dict
tracks = tracker.update(dets, frame)    # -> list[Track] matched this frame
tracker.reset()                         # between unrelated clips
```
- Required keys: `type` (`bytetrack`; `deepsort` raises NotImplementedError, M2.7 stretch),
  `frame_rate`, `track_activation_threshold`, `lost_track_buffer`, `minimum_matching_threshold`,
  `minimum_consecutive_frames`, `history_len`.
- ByteTrack comes from `supervision` (no torch). A new id is output only after
  `minimum_consecutive_frames` matched frames. `frame` is unused by ByteTrack.

### 2.4 Lanes (`perception/lanes/`)

```python
from perception.lanes import build_lane_detector, load_lane_config
lanes = build_lane_detector(load_lane_config("configs/lanes.yaml"))   # method: classical | yolop
result = lanes.detect(frame_bgr)                 # -> LaneResult, video mode (smoothing, frame skip)
result = lanes.detect(img, stateless=True)       # independent images (dataset eval)
lanes.reset()
```
- `LaneDetector` (abstract base): subclasses implement `lane_binary(frame_bgr) -> HxW bool`. The
  base class does the bird's-eye warp, sliding-window polyfit, smoothing, single-side fallback
  and ego mask, so `ClassicalLaneDetector` and `YolopLaneDetector` return the same `LaneResult`.
- `every_n_frames` in the backend's section lets a heavy backend run on every n-th frame and
  reuse the last result. Debug data of the last run: `last_debug`; stage times: `last_timing_ms`.
- `load_lane_config` supports override files with a top-level `base:` key (merged recursively).
- `YolopLaneDetector` loads an ONNX file with ONNX Runtime, or a TensorRT session for a `.engine`
  file.

### 2.5 Depth and geometry (`perception/depth/`)

```python
from perception.depth import DepthEstimator, load_depth_config, load_camera
cfg = load_depth_config("configs/depth.yaml")      # **overrides go into cfg["estimator"]
est = DepthEstimator(cfg)
inv = est.estimate(frame_bgr)        # HxW relative inverse depth (MiDaS), same size as the frame
inv = est.maybe_estimate(frame_bgr)  # None on frames skipped by estimator.every_n_frames
cam = load_camera("configs/camera.yaml", "carla")  # CameraModel
```
- `DepthEstimator(cfg, backend=None)`: `cfg["estimator"]` needs `model` (`midas_small` or
  `dpt_hybrid`) and `backend` (`torch`, `onnx`, `tensorrt`); `cfg["models"][model]` needs
  `input_hw`, `mean`, `std`, `multiple_of`. A skipped frame returns None, never a stale map.
- `CameraModel` (frozen dataclass): `width`, `height`, `fx`, `fy`, `cx`, `cy`, `height_m`,
  `pitch_rad`, `sigma_v_px`, `sigma_pitch_rad`. Methods: `ground_distance(v)` (flat-ground distance
  and its std for image row v, or None), `horizon_row()`, `scaled_to(hw)`,
  `with_intrinsics(P, hw)`, `distance_std(z)`, `inv_depth_std()`, `row_of_distance(z)`.
- `ScaleAligner(cfg["alignment"]).fit(m, z) -> Alignment | None`: per-frame fit
  `1/Z = s*m + t` of MiDaS values to geometric anchor distances; `Alignment.to_depth(m)`.
- `BottomCalibration.from_dict(cfg["geometry"]["calibration"])`: per-camera correction of the
  detector box-bottom row and/or distance scale; identity by default.

### 2.6 Distance fusion (`perception/fusion/`)

```python
from perception.fusion import DistanceFusion
fusion = DistanceFusion(load_depth_config("configs/depth.yaml"), load_camera(path, "carla"))
frame = fusion.update(tracks, lanes, stamp, frame_id, img.shape[:2], inv_depth=inv)  # -> PerceptionFrame
meas, align = fusion.measure(dets, hw, inv_depth)   # stateless per-frame estimates (evaluation)
fusion.reset()
```
Per object (only classes in `fusion.object_classes`, so traffic lights and signs are skipped):
1. Geometric distance from the (calibrated) box bottom row; invalid if the box touches the image
   bottom, the row is too close to the horizon, or the distance exceeds `geometry.max_distance_m`.
2. MiDaS value sampled in the lower-central part of the box, aligned to metres with the
   per-frame fit on confident near anchors (the last fit is reused for a few frames).
3. `fusion.mode`: `inverse_depth_weighted` (KITTI/BDD; inverse-variance mean in 1/Z) or
   `geometric_first` (CARLA; geometric when valid, aligned MiDaS otherwise).
4. Per-track Kalman filter (`FilterBank`, `DistanceKalman`) gives distance, closing speed and
   TTC. TTC is None until the track has been measured for `kalman.min_track_s_for_ttc` and the
   closing speed exceeds `ttc_speed_sigma_k` standard deviations.
5. `in_ego_lane(...)`: `LaneResult.contains` plus a metric lateral corridor (`fusion.ego_lane`).

`Measurement` (from `measure`) holds `det`, `geometric`, `midas_value`, `midas`, `fused`
(each estimate a `(distance_m, std_m)` tuple or None), `anchor`, `align_mode`, `align_anchors`.
Objects with no distance at all are left out of the frame and counted in `last_stats`.

### 2.7 Pipeline (`perception/pipeline.py`)

```python
from perception.pipeline import Pipeline, load_pipeline_config, draw_frame, frame_to_dict
cfg = load_pipeline_config("configs/pipeline.yaml")
pipe = Pipeline.from_config(cfg, backend="torch", camera_name="kitti", fps=10.0)
pf = pipe.process(frame_bgr, stamp=t, frame_id=i)          # -> PerceptionFrame
vis = draw_frame(frame_bgr, pf, cfg["viz"], pipe.lane_debug(), pipe.last_tracks)
d = frame_to_dict(pf, pipe.last_stats)                      # JSON-serialisable dict
```
- `Pipeline(detector, tracker, lanes, depth, fusion, fps)` composes any objects with the stage
  methods (the tests inject fakes). Order in `process`: detect, track, lanes, depth (may be None),
  fusion.
- `Pipeline.from_config(cfg, backend=None, camera_name=None, lanes_method=None, fps=None,
  camera=None, detector_weights=None, yolop_weights=None)` builds every stage; `backend`
  overrides detector and depth backends.
- `describe()` reports which backends actually run; `reset()` clears temporal state;
  `last_stats` has per-frame counts (`detections`, `tracks`, `objects`, `no_estimate`,
  `anchors`, `aligned`, `depth_ran`).
- CLI: `python -m perception.pipeline --source <video|dir> --out out.mp4 [--json frames.jsonl]
  [--summary summary.json] [--snapshots dir] [--config configs/pipeline.yaml]
  [--backend torch|onnx|tensorrt] [--camera kitti|carla|bdd] [--lanes classical|yolop] [--fps N]
  [--max-frames N] [--detector-weights path] [--yolop-weights path] [--no-depth]`.
  The summary reports `pipeline_fps` (stage time only) and `wall_fps` (includes decode, drawing
  and encoding).

### 2.8 Control (`perception/control.py`)

```python
from perception.control import LongitudinalController, candidates_from_frame
ctrl = LongitudinalController(cfg["control"])           # control section of configs/carla.yaml
decision = ctrl.update(candidates_from_frame(pf), ego_speed_mps)   # -> ControlDecision
```
- `LeadCandidate(track_id, distance_m, ttc_s, closing_speed_mps, in_ego_lane)`: one object as the
  controller sees it. `candidates_from_frame` builds them from a `PerceptionFrame`;
  `perception_ros.conversions.candidates_from_msg` from a `TrackedObjectArray`.
- `ControlDecision(override, throttle, brake, mode, lead, reason, extra)`; `mode` is one of
  `MODES = ("AUTOPILOT", "ACC", "GAP", "AEB", "HOLD")`. `override=False` hands throttle and
  brake back to the autopilot.
- Behaviour (ARCHITECTURE.md 3.7): an object becomes the lead only after it has been in the ego
  lane for several consecutive frames under one track id. AEB brakes fully on low TTC or short
  distance and latches into HOLD at standstill; ACC and GAP brake proportionally earlier;
  otherwise AUTOPILOT. The thresholds are in `control` of `configs/carla.yaml` and are being
  tuned (hysteresis work in progress), so they are not repeated here.
- `reset()` clears the persistence counters and the AEB latch; `latched` tells whether HOLD is
  active.

## 3. Configuration files (`configs/`)

All tunables live here; code has no magic numbers. Relative paths resolve from the repo root.
Files with a top-level `base:` key are overrides merged onto the named file.

| File | Read by | Controls |
|---|---|---|
| `data_bdd.yaml` | Ultralytics, `scripts/data/prepare_bdd.py` | BDD YOLO dataset path on Kaggle, 9 class names, `prep` block (subset size, stratification, seed) |
| `data_kitti.yaml` | depth, MOT and tracking evals | KITTI object and tracking folder layout on Kaggle |
| `yolo_bdd.yaml` | `scripts/train_yolo.py` (Kaggle) | Training: data search and staging paths, repeat-factor sampling, epochs, batch, image size, W&B, smoke mode |
| `yolo_detector.yaml` | `Detector` | Runtime detector: backend, weights, imgsz 640, conf, iou, NMS options |
| `yolo_eval.yaml` | `scripts/eval_detection.py` | Detection eval on BDD val, galleries |
| `tracker.yaml` | `Tracker` | ByteTrack settings (frame rate, activation threshold, lost-track buffer, matching threshold, consecutive frames, history length) |
| `tracker_eval.yaml`, `tracker_mot.yaml` | `scripts/eval_tracking.py`, `scripts/eval_mot.py` | Tracking videos and MOT eval on KITTI: sequences, class mapping, sweep |
| `lanes.yaml` | `build_lane_detector` | Lane method, ROI, thresholds, warp, fitting, smoothing, single-side fallback, YOLOP weights, sha256, `every_n_frames`, overlay style |
| `lanes_bdd.yaml`, `lanes_carla.yaml` | lane detector (override files) | Camera-specific warp and fallback for BDD (1280x720) and the CARLA 800x450 camera |
| `lanes_eval.yaml` | `scripts/eval_lanes.py` | Classical vs YOLOP comparison on BDD val |
| `camera.yaml` | `load_camera` | Intrinsics, height, pitch and noise for `kitti`, `carla`, `bdd` |
| `depth.yaml` | `DepthEstimator`, `DistanceFusion` | MiDaS model and backend, box sampling, geometry limits and calibration, anchors, alignment, fusion mode and ego-lane corridor, Kalman/TTC gates |
| `depth_carla.yaml` | live CARLA pipeline (override of `depth.yaml`) | `geometric_first` fusion and the CARLA detector-box calibration |
| `depth_eval.yaml` | `scripts/eval_depth.py` | Distance eval vs KITTI LiDAR: bins, models, leave-one-out |
| `pipeline.yaml` | `load_pipeline_config` | Which stage configs to combine, camera, per-stage overrides, source fps, output and overlay settings |
| `pipeline_carla.yaml` | perception_node (override of `pipeline.yaml`) | CARLA lanes/depth configs, TensorRT engines in `models/`, MiDaS every 2nd frame, 20 fps |
| `pipeline_kaggle.yaml` | `kaggle/run_pipeline/job.py` | Offline pipeline on a KITTI tracking sequence |
| `carla.yaml` | every ROS2 node, `sim/` scripts | CARLA connection and sync step, Traffic Manager, ego, camera pose, bridge timeouts, perception backend, `control` thresholds, recorder, scenarios, dataset recording |
| `carla_eval.yaml` | `sim/eval_frames.py` (Kaggle) | Sim eval on recorded CARLA frames: detection, distance, calibration study |
| `bench.yaml` | `deploy/export.py`, `deploy/benchmark.py` (Kaggle) | Export, engine build, parity, benchmark iterations, accuracy subsets |

## 4. Differences from ARCHITECTURE.md (for the main session)

- ARCHITECTURE.md 2.2 and 3.7 list an `eval_node`. There is none; distance is evaluated offline
  on recorded frames with `sim/eval_frames.py` on Kaggle.
- `/carla/ego/depth_gt` is published only when `depth_camera.enabled` is true (default false).
- ARCHITECTURE.md 3.7 describes the controller with AEB and ACC rules only; the code also has
  GAP and HOLD modes and a lead persistence rule.
- The `/carla/ego/*` rates are 20 Hz in simulation time; wall-clock rate is lower (4.9 sim
  steps/s measured).
