# CLAUDE.md

Guidance for Claude Code in this repository. Shared rules for every coding agent live in
@AGENTS.md (imported). Read it first; this file adds Claude-specific notes only.

## Project in one line
**AIML-04 – Autonomous Driving Perception Stack** (Nexvia Technologies internship, AI/ML track).
A camera-only pipeline that detects vehicles/pedestrians, tracks them, detects lanes, estimates
distance, and runs live inside the CARLA simulator through ROS2 nodes. Deadline: **2026-11-30**.

## Key docs (read before large changes)
| File | What it holds |
|---|---|
| `PLAN.md` | Phased roadmap, dated milestones, checklist. Tick boxes when a task finishes. |
| `ARCHITECTURE.md` | Components, data flow, ROS2 topics, data contracts, repo layout. Source of truth for module boundaries. |
| `AGENTS.md` | Coding rules, commands, subagent roster and ownership. |
| `docs/PROJECT_BRIEF.md` | The original brief from the Nexvia portal (requirements, resources, submission rules). |
| `docs/SETUP.md` | Windows + WSL2 + CARLA + ROS2 + CUDA setup. |
| `docs/EXPERIMENTS.md` | Experiment log: every training/benchmark run with W&B link and numbers. |
| `docs/SUBMISSION.md` | Final deliverables checklist mapped to the portal requirements. |

## Using subagents
Project subagents are in `.claude/agents/`. Delegate by area instead of doing everything in the
main thread:

| Agent | Use it for |
|---|---|
| `data-engineer` | Dataset download/conversion, EDA, class balance, YOLO-format labels |
| `detection-tracking` | YOLO fine-tuning, ByteTrack/DeepSORT, detection & MOT metrics |
| `lane-depth` | Classical OpenCV lanes, segmentation lanes, MiDaS depth, distance fusion |
| `sim-ros-integrator` | CARLA client, ROS2 nodes, launch files, obstacle-reaction controller |
| `edge-optimizer` | ONNX export, TensorRT engines, latency/FPS benchmarks |
| `qa-reviewer` | Tests, code review, metric sanity checks; gate before any milestone is marked done |
| `docs-presenter` | README, diagrams, experiment write-ups, demo script, slides, reflection |

Rules for delegation:
- Give the agent the milestone ID from `PLAN.md` and the interface it must respect from `ARCHITECTURE.md`.
- One agent owns a directory at a time (ownership table in `AGENTS.md`). Never run two agents that write the same directory in parallel.
- After an agent finishes, run `qa-reviewer` on its changes before ticking the milestone.
- Safe parallel pairs: `data-engineer` + `sim-ros-integrator` (M1 with M5.1 setup), `detection-tracking` + `lane-depth` (M2 with M3).

## Claude-specific notes
- **Keep the laptop load as low as possible. This is the user's explicit request.** Training, dataset downloads, dataset-wide evals, exports and benchmarks run on Kaggle, tests run in GitHub Actions, and metrics go to W&B. Write code locally, push it, launch it with `kaggle kernels push`, and monitor from W&B. Don't run GPU jobs or long processes on the laptop.
- The laptop runs only: CARLA + ROS2 sessions (30 minutes or less, ask the user first), the one-time local TensorRT engine build, and demo recording.
- Laptop hardware is an **RTX 3060 Laptop, 6 GB VRAM**. Never propose CARLA Epic quality, large maps, YOLO `l/x` models, or PyTorch inference in the live loop.
- The host is Windows 11. CARLA runs on **Windows**; ROS2 Humble runs in **WSL2 Ubuntu 22.04**; cloud jobs run on **Kaggle**. Every shell command you write must say where it runs.
- The system Python is 3.14, which CARLA and many CUDA wheels don't support. Use Python 3.10 (WSL Ubuntu or the optional CPU venv).
- Never commit datasets, model weights (`*.pt`, `*.onnx`, `*.engine`), W&B run folders, videos or `kaggle.json`. Datasets live on Kaggle and weights live in W&B Artifacts.
- Record every training run and benchmark in `docs/EXPERIMENTS.md` as soon as it finishes.
