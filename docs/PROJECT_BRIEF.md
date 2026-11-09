# Project Brief – AIML-04 Autonomous Driving Perception Stack

Source: Nexvia Technologies student portal (`/student/projects`), Project #1 of 2. Captured 2026-11-08.

| Field | Value |
|---|---|
| Code | AIML-04 |
| Track | AI / Machine Learning |
| Difficulty | Intermediate |
| Estimated effort | 14 days |
| Assigned | 2026-10-29 |
| Deadline | **2026-11-30** |
| Status | Assigned |

## Description
A vision pipeline that detects lanes, vehicles, and pedestrians, tracks them, estimates distance,
and runs inside a driving simulator.

## Instructions
1. Use the BDD100K or KITTI dataset. Explore the labels and class balance.
2. Fine-tune a YOLO model for object detection and track objects across frames with ByteTrack or DeepSORT.
3. Build lane detection using OpenCV (classical) and compare it with a segmentation model.
4. Add monocular depth estimation (MiDaS) to estimate distance to objects.
5. Install CARLA, attach a virtual camera, and stream frames to your pipeline through ROS2 nodes.
6. Export models to ONNX and optimize with TensorRT. Benchmark latency and FPS before and after.
7. Log experiments in Weights & Biases and record a demo video of the car reacting to obstacles.

## Tools required
YOLO · OpenCV · ROS2 · CARLA Simulator · TensorRT · ONNX · Weights & Biases

## Resources (datasets)
BDD100K · KITTI · Waymo Open Dataset · nuScenes · Cityscapes (segmentation) · CULane / TuSimple (lanes) · synthetic frames from CARLA.

## Submission requirements
- Working demo (deployed or screen-recorded)
- GitHub repository with clean commits, README, and setup instructions
- Architecture diagram and data or API documentation
- Test results and, where relevant, performance or accuracy numbers
- 10-minute final presentation covering problem, design, challenges, results, and next steps
- Individual reflection on what you learned and what you owned

## Portal submission rules (shared with Project #2, AIML-01)
- One Google Drive folder named `Anish_Kumar_Gupta_Capstone_Projects` with subfolders `AIML-04` and `AIML-01` (code, reports, screenshots, datasets).
- General access: "Anyone with the link can view / comment".
- Form fields: GitHub repository URL (required), live deployment URL (Vercel/Netlify/Render), Drive folder URL, reviewer notes.
- **Single submission only.** It is locked once submitted.
