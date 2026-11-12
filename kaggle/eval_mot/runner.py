"""Kaggle runner for M2.6: MOT metrics (MOTA, IDF1, ID switches) on KITTI tracking (GPU).

Set REF and MODE below, push the code, then from the laptop:
    kaggle kernels push -p kaggle/eval_mot
    smoke -> sequence 0000, 50 frames, 1 tracker config (~5 min), proves the whole job
    full  -> 8 KITTI sequences (~2.4k frames) x 6 tracker configs (~10-15 min)
Weights come from the W&B artifact yolo11s-bdd-best:candidate; KITTI tracking (image_02 +
label_02) is attached through dataset_sources. The detector runs once per frame and the tracker
sweep replays the cached detections.
Outputs: /kaggle/working/m2_mot (results.json, m2_mot.md) and one W&B run (group track).
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "m2-mot-followup"  # branch, tag or commit SHA to run; set to "main" after the merge
MODE = "full"  # smoke | full
EXTRA = ["--smoke"] if MODE == "smoke" else []
JOB_CMD = ["python", "-m", "scripts.eval_mot", "--config", "configs/tracker_mot.yaml", *EXTRA]
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
       and can be uploaded later with `wandb sync`. Weights can't be fetched offline, so the
       job will fail in that case.
    """
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        print("W&B key loaded from Kaggle Secrets")
        return
    except Exception as exc:
        print(f"Kaggle Secrets unavailable ({type(exc).__name__})")
    # Shallow patterns only: a recursive glob would walk the 37 GB KITTI tree.
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
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORKDIR)
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    sh(JOB_CMD, cwd=WORKDIR)


if __name__ == "__main__":
    main()
