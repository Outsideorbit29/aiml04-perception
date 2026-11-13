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

## Progress snapshot (update at the end of every session)
Full checklist is in `PLAN.md`; numbers are in `docs/EXPERIMENTS.md`. This is the short version.

_Last updated: 2026-11-13_

| Milestone | Status | Notes |
|---|---|---|
| M0 Setup | Done except M0.7–M0.8 | GitHub, Kaggle, W&B, CI working. CARLA + WSL2/ROS2 local setup still open (needs user, laptop). |
| M1 Data | Done | 20k BDD subset + 10k val in YOLO format (kernel `anish2906/aiml04-data-prep`), EDA, fixtures, class decision. |
| M2 Detection/tracking | M2.1–M2.6 done (M2.7 stretch open) | YOLO11s mAP@0.5 **0.553** (zero-shot 0.361), artifact `yolo11s-bdd-best`, eval W&B `8tzih6my`. MOT on 8 KITTI sequences (W&B `qvrkp7t1`): MOTA 52.5, IDF1 67.4, IDSW 176 with tuned ByteTrack (buffer 60, 2 consecutive frames); vehicles 63.7 / 75.7, pedestrians 38.5 / 54.9. New ids appear 2 frames after first detection. Det+track 52 FPS on T4. |
| M3 Lanes | M3.1–M3.4 done (M3.5 stretch open) | 500 BDD val (W&B `bf4xp164`): lane F1 YOLOP **0.52** vs classical 0.23; ego-lane IoU tie ~0.42 (YOLOP finds no line on 129/500: faint markings, snow, traffic). YOLOP is the live default (switch `method` in M5). BDD uses `configs/lanes_bdd.yaml`; single-side fallback builds the ego mask from one line. Figures in `docs/figures/lanes_*`. |
| M4 Depth/distance | Done | 1000 KITTI frames vs LiDAR (W&B `amgl803i`): AbsRel geometric 0.168, fused geometric+MiDaS-small **0.170** with coverage 1.00 (geometric 0.96), bias −0.001; 30–60 m 0.221. MiDaS-small is live (27 ms at 256x832 on T4); DPT-Hybrid is no better and 424 ms. Table in `docs/figures/depth_comparison.md`. TTC gate: track ≥ 0.75 s and closing > 2σ. M4.6 offline pipeline on KITTI 0001 (W&B `e12sh0xi`): 11.4 FPS stages / 4.8 end-to-end on T4; in_ego_lane uses a metric lateral corridor; 1 AEB-trigger lead frame in 447 (a crossing car). |
| M5 CARLA + ROS2 | Not started | Needs M0.7–M0.8 first. |
| M6 ONNX/TensorRT | Not started | |
| M7 Deliverables | Not started | |

**Schedule:** M2, M3 and M4 done on 2026-11-13, ahead of PLAN.md dates. Open before/in M5: long-range kerbside false leads (47/111 lead frames, lane-depth to tighten the corridor beyond ~24 m) and a multi-frame lead persistence rule in `control_node`. M0.7–M0.8 local CARLA/ROS2 setup needed before M5 (Nov 16).

Useful facts learned so far:
- The Kaggle CLI is at `C:\Users\anish\AppData\Roaming\Python\Python314\Scripts\kaggle.exe` (not on PATH).
- Fetch only small outputs from a job: `kaggle kernels output <ref> -p <dir> --file-pattern '\.(log|json|csv|yaml)$'`.
- Kaggle jobs clone the repo shallowly; runners now `git fetch origin REF` + checkout `FETCH_HEAD`, so `REF` may be a branch. Set `REF` back to `main` before merging.
- Run a `MODE = "smoke"` job first and look at its debug outputs before any full run; implausible numbers (e.g. whole groups at 0.000) have meant eval bugs.
- Don't switch branches in the main checkout while a subagent is editing; merge through `git worktree` instead.
- `LaneResult.contains()` uses the ego mask only on its measured rows; below/above it uses the polylines or extends the mask edges. M4.5 `in_ego_lane` can call it with the box bottom-centre directly.
- YOLOP ONNX outputs probabilities (range ~0.41–1), not logits; `configs/lanes.yaml` sets `yolop.output_kind: probs` (`auto` also works).
- MiDaS via `torch.hub` loads a nested hub repo (`rwightman/gen-efficientnet-pytorch`) that must be pre-trusted (`trusted_hub_repos` in `configs/depth.yaml`), or the Kaggle job hangs on a stdin prompt. `configs/depth.yaml` defaults to `backend: onnx`; Kaggle scripts pass `backend="torch"`.
- The offline pipeline needs `--backend torch` on Kaggle until the M6 ONNX export (`configs/depth.yaml` defaults to onnx). Its `pipeline_fps` is stage time only; `wall_fps` includes decode/draw/encode.
- Distance fusion is done in inverse depth; averaging in metres biased fused distances short (−0.08 at 30–60 m).
- A 50-epoch YOLO11s run on 20k images at 640 takes ~4.6 h on a Kaggle T4 (fits one session).

## Claude-specific notes
- **Keep the laptop load as low as possible. This is the user's explicit request.** Training, dataset downloads, dataset-wide evals, exports and benchmarks run on Kaggle, tests run in GitHub Actions, and metrics go to W&B. Write code locally, push it, launch it with `kaggle kernels push`, and monitor from W&B. Don't run GPU jobs or long processes on the laptop.
- The laptop runs only: CARLA + ROS2 sessions (30 minutes or less, ask the user first), the one-time local TensorRT engine build, and demo recording.
- Laptop hardware is an **RTX 3060 Laptop, 6 GB VRAM**. Never propose CARLA Epic quality, large maps, YOLO `l/x` models, or PyTorch inference in the live loop.
- The host is Windows 11. CARLA runs on **Windows**; ROS2 Humble runs in **WSL2 Ubuntu 22.04**; cloud jobs run on **Kaggle**. Every shell command you write must say where it runs.
- The system Python is 3.14, which CARLA and many CUDA wheels don't support. Use Python 3.10 (WSL Ubuntu or the optional CPU venv).
- Never commit datasets, model weights (`*.pt`, `*.onnx`, `*.engine`), W&B run folders, videos or `kaggle.json`. Datasets live on Kaggle and weights live in W&B Artifacts.
- Record every training run and benchmark in `docs/EXPERIMENTS.md` as soon as it finishes.
