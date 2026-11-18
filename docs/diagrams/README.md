# Diagrams

The Mermaid sources are the `.mmd` files in this folder. GitHub renders the copies below directly,
so you can read them on the repo page without an image export. A PNG/SVG export for the slides and
the Drive folder is still open (see "Exporting" below).

| File | What it shows | Based on |
|---|---|---|
| `architecture.mmd` | Cloud plane (GitHub, Actions, Kaggle, W&B) and the laptop runtime (CARLA on Windows, ROS2 nodes in WSL2) with topics | ARCHITECTURE.md section 2, `ros2_ws/src/perception_ros` |
| `pipeline.mmd` | Stages of `perception.pipeline.Pipeline.process` and the types passed between them | `perception/pipeline.py`, `perception/types.py` |

If you change a `.mmd` file, copy it into the matching block below.

## System architecture

```mermaid
flowchart LR
    subgraph CLOUD["Cloud (online ML plane)"]
        direction TB
        GH["GitHub repo<br/>aiml04-perception"]
        CI["GitHub Actions<br/>ruff + pytest, Python 3.10"]
        KG["Kaggle T4 jobs (kaggle/*)<br/>data prep, YOLO training,<br/>evals, ONNX export,<br/>TensorRT benchmark"]
        WB["Weights and Biases<br/>project aiml04-perception<br/>metrics + artifacts<br/>(best.pt, *.onnx)"]
        GH -->|"push"| CI
        GH -->|"clone at commit"| KG
        KG -->|"metrics, best.pt, ONNX"| WB
    end

    subgraph WIN["Laptop: Windows 11 (RTX 3060 Laptop, 6 GB)"]
        CARLA["CARLA 0.9.15 server<br/>Town01, Low quality,<br/>synchronous mode, 0.05 s step<br/>ego + 800x450 RGB camera"]
    end

    subgraph WSL["Laptop: WSL2 Ubuntu 22.04, ROS2 Humble"]
        direction LR
        BRIDGE["carla_bridge_node<br/>ticks the world,<br/>applies control"]
        subgraph PN["perception_node (TensorRT FP16 engines)"]
            direction TB
            DET["Detector<br/>YOLO11s-BDD"]
            TRK["Tracker<br/>ByteTrack"]
            LANE["LaneDetector<br/>YOLOP, every 2nd frame"]
            DEPTH["DepthEstimator<br/>MiDaS-small, every 2nd frame"]
            FUS["DistanceFusion<br/>geometric-first + Kalman,<br/>in_ego_lane"]
            DET --> TRK --> FUS
            LANE --> FUS
            DEPTH --> FUS
        end
        CTRL["control_node<br/>LongitudinalController<br/>AUTOPILOT / ACC / GAP / AEB / HOLD"]
        REC["recorder_node<br/>annotated MP4 in media/"]
        BAG["rosbag2<br/>(record_bag:=true)"]
        BRIDGE -->|"/carla/ego/rgb (Image)"| PN
        BRIDGE -->|"/carla/ego/odom (Odometry)"| CTRL
        PN -->|"/perception/objects (TrackedObjectArray)"| CTRL
        PN -->|"/perception/lanes (LaneBoundaries)"| BAG
        PN -->|"/perception/debug_image (Image)"| REC
        CTRL -->|"/carla/ego/control (EgoControl)"| BRIDGE
        CTRL -->|"/carla/ego/control"| REC
    end

    CARLA <-->|"CARLA client API, TCP 2000-2002"| BRIDGE
    WB -->|"pull ONNX, build local TRT engines once"| PN
    CARLA -.->|"recorded frames uploaded as Kaggle dataset"| KG
```

Solid arrows inside WSL2 are ROS2 topics. Not drawn: `/carla/ego/depth_gt`, which the bridge
publishes only when `depth_camera.enabled` is true in `configs/carla.yaml` (off by default; nothing
live reads it). The dashed arrow is the offline path used for the M5.6 sim eval. Full topic and
message reference: [docs/API.md](../API.md).

## Perception pipeline data flow

```mermaid
flowchart TB
    IN["BGR frame (HxWx3 uint8)<br/>+ stamp, frame_id"]
    DET["1. Detector.detect(frame)<br/>letterbox 640, YOLO11s, NMS<br/>backend: torch / onnx / tensorrt"]
    TRK["2. Tracker.update(dets, frame)<br/>ByteTrack (supervision)<br/>id after 2 consecutive frames"]
    LANE["3. LaneDetector.detect(frame)<br/>classical (OpenCV) or YOLOP<br/>lane binary, bird's-eye warp, polyfit, smoothing"]
    DEPTH["4. DepthEstimator.maybe_estimate(frame)<br/>MiDaS-small relative inverse depth<br/>None on skipped frames (every_n_frames)"]
    subgraph FUSION["5. DistanceFusion.update(tracks, lanes, stamp, frame_id, hw, inv_depth)"]
        direction TB
        GEO["Geometric distance<br/>box bottom row + camera height/pitch<br/>(+ optional detector-box calibration)"]
        ALIGN["Per-frame scale/shift fit<br/>MiDaS -> metric on near, confident anchors"]
        MODE["fusion.mode<br/>inverse_depth_weighted (KITTI/BDD)<br/>geometric_first (CARLA)"]
        KF["Per-track Kalman filter<br/>distance, closing speed, gated TTC"]
        EGO["in_ego_lane<br/>LaneResult.contains + metric lateral corridor"]
        GEO --> MODE
        ALIGN --> MODE
        MODE --> KF --> EGO
    end
    OUT["PerceptionFrame<br/>objects: list of ObjectState<br/>lanes: LaneResult<br/>timings_ms: detect/track/lanes/depth/fusion/total"]
    LEAD["PerceptionFrame.lead_object()<br/>closest in-ego-lane object"]
    SINKS["Outputs<br/>offline: annotated MP4, frames.jsonl, summary.json<br/>live: ROS2 messages to control_node"]

    IN --> DET -->|"list of Detection"| TRK
    IN --> LANE
    IN --> DEPTH
    TRK -->|"list of Track"| FUSION
    LANE -->|"LaneResult"| FUSION
    DEPTH -->|"HxW inverse depth or None"| FUSION
    FUSION --> OUT --> LEAD
    OUT --> SINKS
```

Stages 1 to 4 each read the raw frame; only the tracker depends on the detector. Fusion is the one
place where they meet. `timings_ms` records the wall time of each call, so frames where YOLOP or
MiDaS is skipped show a small amortised cost.

## Exporting PNG/SVG (TODO for slides and Drive)
Rendering locally needs `@mermaid-js/mermaid-cli`, which downloads a headless Chromium. That is too
heavy for the laptop, so the export was not done. To export without installing anything:
1. Open this page on GitHub, or paste a `.mmd` file into the Mermaid Live Editor in a browser.
2. Export PNG (for slides) and SVG (for print), and save them as `docs/diagrams/architecture.png`
   and `docs/diagrams/pipeline.png`.
