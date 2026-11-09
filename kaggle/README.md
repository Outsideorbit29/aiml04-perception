# kaggle/ – cloud job runners

Each subfolder is one Kaggle job: a `kernel-metadata.json` and a small `runner.py` that clones this
repo, installs `requirements.txt`, and runs a single command on Kaggle's GPU.

| Folder | Milestone | Owner agent |
|---|---|---|
| `_template/` | – | copy this to start a new job |
| `data_prep/` | M1 | data-engineer |
| `train_yolo/` | M2 | detection-tracking |
| `eval_track/` | M2 | detection-tracking |
| `eval_lanes/`, `eval_depth/` | M3, M4 | lane-depth |
| `sim_eval/` | M5 | sim-ros-integrator |
| `export_bench/` | M6 | edge-optimizer |

## Run a job (from the laptop)
```bash
git push                                   # Kaggle clones the pushed code
kaggle kernels push -p kaggle/<job>
kaggle kernels status anish2906/aiml04-<job>
kaggle kernels output anish2906/aiml04-<job> -p runs/<job>   # only if outputs are needed locally
```

## One-time per job
The first time a job is pushed, open it on kaggle.com → Add-ons → Secrets and attach
`WANDB_API_KEY`. Without it the runner logs W&B offline instead of failing.

Kaggle limits: about 30 GPU-hours per week and up to 12 h per session. Long jobs checkpoint to W&B and resume.
