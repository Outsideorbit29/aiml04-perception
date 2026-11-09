# kaggle/ – cloud job runners

Each subfolder is one Kaggle job: a `kernel-metadata.json` and a small `runner.py` that clones this
repo, installs `requirements.txt`, and runs a single command on Kaggle's GPU.

| Folder | Milestone | Owner agent |
|---|---|---|
| `_template/` | – | copy this to start a new job |
| `data_probe/` | M1 | data-engineer: lists the layout of candidate datasets (CPU) |
| `data_prep/` | M1 | data-engineer: BDD100K to YOLO subset, EDA, fixtures (CPU, `anish2906/aiml04-data-prep`) |
| `data_check/` | M1 | data-engineer: 1 tiny epoch on the prepared data (GPU, under 10 min) |
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

## Dataset mounts and chaining jobs
- Attached datasets are mounted at `/kaggle/input/datasets/<owner>/<slug>/` (probed 2026-11-09),
  not `/kaggle/input/<slug>/`. Scripts take paths from `configs/data_*.yaml` or CLI flags.
- The BDD YOLO data is the **output of the data-prep kernel**, not a separate dataset (kernels
  have no Kaggle API credentials). Later jobs attach it with
  `"kernel_sources": ["anish2906/aiml04-data-prep"]` and use `configs/data_bdd.yaml`, or run
  `python -m scripts.data.check_data_yaml` to write a YAML with the real mount path.
- The data-prep output holds about 60k files. Don't run `kaggle kernels output` on it: listing
  that many files trips the API rate limit. The small artifacts (EDA, fixtures, summaries) are
  copied into the `data_check` job output instead.

## W&B
Project `aiml04-perception`, entity `anish-grd2004-vellore-institute-of-technology` (the account's
default entity, so `wandb.init(project="aiml04-perception")` logs there without extra flags).

## W&B key on Kaggle
Kaggle Secrets are **not** passed to runs started with `kaggle kernels push` (verified 2026-11-09:
the secret was attached and ticked, and the run still couldn't read it). So `runner.py` looks for the key in this order:
1. Kaggle Secrets, which only works for runs saved from the browser editor.
2. A file `wandb_api_key` inside the private dataset `anish2906/aiml04-secrets`. Every job lists
   this dataset in `dataset_sources`.
3. Otherwise it logs offline into `/kaggle/working/wandb`. Upload such a run later with
   `kaggle kernels output ... -p runs/<job>` and then `wandb sync runs/<job>/wandb/offline-run-*`.

The secrets dataset lives in `%USERPROFILE%\.kaggle\aiml04-secrets\`, outside the repo. It is
private to the Kaggle account. If the key ever leaks, rotate it at wandb.ai/authorize and
re-upload with `kaggle datasets version -p %USERPROFILE%\.kaggle\aiml04-secrets -m rotate`.

Kaggle limits: about 30 GPU-hours per week and up to 12 h per session. Long jobs checkpoint to W&B and resume.
