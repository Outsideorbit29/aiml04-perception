"""Kaggle runner for M4.6: full offline pipeline on one KITTI tracking sequence (GPU).

Set MODE (and REF if the code is on a branch), push the code, then from the laptop:
    kaggle kernels push -p kaggle/run_pipeline
    smoke -> first 50 frames of the sequence; check summary.json and snapshots/*.jpg first
    full  -> the whole sequence (configs/pipeline_kaggle.yaml `sequence`)
KITTI tracking (image_02 + calib) comes from leducnhuan/kitti-tracking. Detector weights come
from the W&B artifact yolo11s-bdd-best:candidate, YOLOP ONNX from the hustvl/YOLOP repo and
MiDaS-small from torch.hub, all inside the job (internet on); nothing is downloaded to the
laptop. Outputs: /kaggle/working/m4_pipeline (out.mp4, frames.jsonl, summary.json,
snapshots/*.jpg) plus one W&B run (project aiml04-perception, group pipeline).
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "m4-pipeline"  # branch, tag or commit SHA to run; set back to "main" before merging
MODE = "smoke"  # smoke | full
EXTRA = ["--smoke"] if MODE == "smoke" else []
JOB_CMDS = [
    ["python", "kaggle/run_pipeline/job.py", "--config", "configs/pipeline_kaggle.yaml", *EXTRA],
]
# intel-isl/MiDaS (v3.1 hubconf) was written against timm 0.6.x; newer timm releases are known
# to break its DPT backbones. Pinned for this job only (requirements.txt keeps timm>=1.0),
# as in kaggle/eval_depth.
JOB_PIP = ["timm==0.6.13"]
WORKDIR = "/kaggle/temp/repo"  # outside /kaggle/working so the code is not saved as job output


def sh(cmd: list[str], check: bool = True, **kw) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, **kw).returncode


def load_wandb_key() -> None:
    """Find the W&B key, trying three sources in order.

    1. Kaggle Secrets (works for runs saved from the browser editor).
    2. A file named `wandb_api_key` in an attached private Kaggle dataset
       (works for runs pushed with `kaggle kernels push`, which get no secrets).
    3. Neither: log offline into /kaggle/working/wandb so the run is saved as job output
       and can be uploaded later with `wandb sync`.
    """
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        print("W&B key loaded from Kaggle Secrets")
        return
    except Exception as exc:
        print(f"Kaggle Secrets unavailable ({type(exc).__name__})")
    # Shallow patterns only: a recursive glob would walk the KITTI file trees.
    patterns = ["/kaggle/input" + "/*" * depth + "/wandb_api_key" for depth in range(1, 5)]
    for path in [p for pat in patterns for p in sorted(glob.glob(pat))]:
        with open(path) as fh:
            key = fh.read().strip()
        if key:
            os.environ["WANDB_API_KEY"] = key
            print("W&B key loaded from attached private dataset")
            return
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "/kaggle/working"
    print("No W&B key found; logging offline to /kaggle/working/wandb (sync later)")


def main() -> None:
    load_wandb_key()
    if not os.path.exists(WORKDIR):
        sh(["git", "clone", "--depth", "50", REPO, WORKDIR])
    # A shallow clone has only the default branch, so fetch REF explicitly.
    sh(["git", "fetch", "--depth", "50", "origin", REF], cwd=WORKDIR)
    sh(["git", "checkout", "FETCH_HEAD"], cwd=WORKDIR)
    # Commit under test; job.py logs it (summary.json git_sha and the W&B config) and falls
    # back to this env var if git is unavailable in its process.
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKDIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    os.environ["GIT_COMMIT"] = sha
    print(f"code commit {sha} (REF {REF})", flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORKDIR)
    if JOB_PIP:
        sh([sys.executable, "-m", "pip", "install", "-q", *JOB_PIP], cwd=WORKDIR)
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    # Run every step even if one fails, then fail the job so the status shows it.
    # job.py is a plain script, so the repo root must be on the import path.
    env = {**os.environ, "PYTHONPATH": WORKDIR}
    failed = [cmd[1] for cmd in JOB_CMDS if sh(cmd, check=False, cwd=WORKDIR, env=env) != 0]
    if failed:
        raise SystemExit(f"failed steps: {failed}")


if __name__ == "__main__":
    main()
