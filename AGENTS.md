# AGENTS.md

Instructions for any AI coding agent (Claude Code, Codex, Cursor, subagents) working in this repo.

## 1. Context
- Project: **AIML-04 Autonomous Driving Perception Stack**. Brief: `docs/PROJECT_BRIEF.md`.
- Design: `ARCHITECTURE.md`. Roadmap: `PLAN.md`. Do not add features outside them; propose changes in your report instead.
- Hard deadline **2026-11-30**. A second internship project (AIML-01 Fraud Detection) has the same deadline, so prefer the simplest approach that meets the brief.

## 2. Environment: cloud-first, low laptop load
**Rule: anything that can run online runs online.** The user wants the laptop load kept as low as
possible. Never run training, dataset downloads, dataset-wide evaluation or benchmarks on the
laptop. Write the code, push it, and launch it on Kaggle.

| Where | What runs there | Python |
|---|---|---|
| ☁ Kaggle notebook (T4/P100) | Data prep, EDA, training, all evaluations, ONNX export, T4 TensorRT benchmarks, offline video rendering | Kaggle image (clones the repo) |
| ☁ GitHub Actions | `pytest`, `ruff` on every push | 3.10 |
| ☁ W&B | Metrics and artifacts (`best.pt`, `*.onnx`) | — |
| 💻 Windows 11 | CARLA 0.9.15 server (short sessions only), `kaggle`/`wandb` CLIs | optional CPU-only 3.10 venv for quick tests |
| 💻 WSL2 Ubuntu 22.04 | ROS2 Humble, `ros2_ws` nodes, live inference on ONNX Runtime/TensorRT, local TRT engine build | 3.10 (ships with 22.04) |

- Laptop GPU: RTX 3060 Laptop, **6 GB VRAM**. With CARLA running, the pipeline gets about 2.5 GB, so live inference uses TensorRT FP16 `n/s` models.
- No PyTorch and no datasets on the laptop. Only `tests/fixtures/` and the ONNX/TRT files.
- Kaggle budget is about 30 GPU-hours per week with sessions of up to 12 h. Long jobs checkpoint every epoch to W&B and resume.
- Laptop CARLA sessions are 30 minutes or less, planned in advance (PLAN.md M5). Ask the user before starting CARLA.

## 3. Commands
```bash
# ☁ Launch a cloud job from the laptop (after git push)
kaggle kernels push -p kaggle/train_yolo          # each kaggle/<job>/ has kernel-metadata.json + notebook
kaggle kernels status <user>/aiml04-train-yolo
kaggle kernels output <user>/aiml04-train-yolo -p runs/train_yolo   # only when outputs are needed locally

# What the Kaggle runner notebook executes (never on the laptop)
python scripts/train_yolo.py --config configs/yolo_bdd.yaml
python -m perception.pipeline --source /kaggle/input/<clip.mp4 or image dir> --out /kaggle/working/out.mp4 \
    --json frames.jsonl --summary summary.json --snapshots snaps --backend torch --camera kitti \
    --detector-weights <best.pt> --yolop-weights <yolop-640-640.onnx>   # --backend torch until the M6 ONNX export
python deploy/export.py --weights best.pt --formats onnx engine --half
python deploy/benchmark.py --config configs/bench.yaml           # writes docs/benchmarks.md content

# 💻 Quick local tests (CPU only, optional; CI runs them anyway)
pytest -q

# 💻 CARLA server (Windows, short sessions)
CarlaUE4.exe -quality-level=Low -RenderOffScreen -fps=20

# 💻 WSL2
python3 deploy/build_local_engine.py --onnx models/*.onnx --fp16      # one time, ~5 min
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch perception_ros stack.launch.py carla_host:=<windows-host-ip>
```
Keep this section current when commands change.

## 4. Code rules
- Python 3.10, type hints on public functions, `ruff` + `black` (line length 100).
- Core logic lives in the `perception/` package and is **ROS-free and CARLA-free**, so it can be unit-tested and run on plain videos. ROS2 nodes in `ros2_ws/` are thin wrappers that convert messages and call `perception/`.
- All tunables (thresholds, model paths, image size, camera intrinsics) go in `configs/*.yaml`. No magic numbers in code.
- Data contracts are the dataclasses in `perception/types.py` (ARCHITECTURE.md section 4). Changing them means updating ARCHITECTURE.md and every consumer in the same change.
- Every module gets at least one test in `tests/` using small fixtures from `tests/fixtures/` (a few frames, never full datasets).
- Log training and benchmarks to Weights & Biases project `aiml04-perception`, and paste the run URL into `docs/EXPERIMENTS.md`.
- Scripts must run unchanged on Kaggle and locally. Paths come from config or CLI flags (`/kaggle/input/...` vs local), never hard-coded.
- Never commit `data/`, `models/`, `runs/`, `wandb/`, `media/`, `*.pt`, `*.onnx`, `*.engine`, `.env`, `kaggle.json`.
- Git: small commits with imperative messages (`Add ByteTrack wrapper`), one branch per milestone `m<id>-<slug>`.

## 5. Subagent roster and ownership
Definitions are in `.claude/agents/`. Each agent writes only to the paths it owns and may read anything.

| Agent | Owns (write access) | Milestones |
|---|---|---|
| `data-engineer` | `scripts/data/`, `notebooks/eda*.ipynb`, `configs/data_*.yaml`, `kaggle/data_prep/` | M1 |
| `detection-tracking` | `perception/detection/`, `perception/tracking/`, `scripts/train_yolo.py`, `configs/yolo_*.yaml`, `configs/tracker*.yaml`, `kaggle/train_yolo/`, `kaggle/eval_track/`, `kaggle/eval_mot/`, `scripts/eval_detection.py`, `scripts/eval_tracking.py`, `scripts/eval_mot.py` | M2 |
| `lane-depth` | `perception/lanes/`, `perception/depth/`, `perception/fusion/`, `configs/lanes*.yaml`, `configs/depth*.yaml`, `kaggle/eval_lanes/`, `kaggle/eval_depth/` | M3, M4 |
| `sim-ros-integrator` | `ros2_ws/`, `sim/`, `configs/carla*.yaml`, `configs/pipeline*.yaml`, `perception/pipeline.py`, `kaggle/sim_eval/`, `kaggle/run_pipeline/` | M5 |
| `edge-optimizer` | `deploy/`, `configs/bench*.yaml`, `docs/benchmarks.md`, `kaggle/export_bench/` | M6 |
| `qa-reviewer` | `tests/`, `.github/workflows/` | gate on every milestone |
| `docs-presenter` | `README.md`, `docs/` (except `benchmarks.md`), slides, demo script | M7 |

Shared files (`perception/types.py`, `requirements*.txt`, `ARCHITECTURE.md`, `PLAN.md`) are changed
only by the main session, or by an agent that lists the change in its handoff report.

## 6. Definition of done (every milestone)
1. Code runs from a clean checkout with the commands in section 3, on Kaggle for cloud jobs and on the laptop for the runtime.
2. CI is green and `qa-reviewer` reported no blocking issues.
3. Metrics are recorded in `docs/EXPERIMENTS.md` with a W&B link.
4. The checkbox is ticked in `PLAN.md`.

## 7. Handoff report format (subagents)
End every task with:
```
Milestone: M<x.y>
Changed: <files>
How to run: <command>
Results: <metrics or n/a>
Open issues / next step: <...>
```
