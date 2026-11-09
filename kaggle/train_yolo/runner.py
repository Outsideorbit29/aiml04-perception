"""Kaggle runner for M2.1-M2.3: zero-shot baseline + YOLO11s fine-tune on BDD100K (GPU).

Set MODE below, push the code, then from the laptop:
    kaggle kernels push -p kaggle/train_yolo
    smoke  -> 1 tiny epoch at imgsz 320 on 640 images (~10 min), proves the whole job
    train  -> zero-shot baseline, then the full 50-epoch run (stops itself at 11 h)
    resume -> continue from the newest W&B checkpoint in a new session
The data-prep output is attached through kernel_sources in kernel-metadata.json.
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "main"  # branch, tag or commit SHA to run
MODE = "train"  # smoke | train | resume
JOB_CMD = ["python", "-m", "scripts.train_yolo", "--config", "configs/yolo_bdd.yaml"]
if MODE != "train":
    JOB_CMD.append(f"--{MODE}")
WORKDIR = "/kaggle/temp/repo"  # outside /kaggle/working so the code is not saved as job output


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


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
    for path in glob.glob("/kaggle/input/**/wandb_api_key", recursive=True):
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
    sh(["git", "checkout", REF], cwd=WORKDIR)
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORKDIR)
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    sh(JOB_CMD, cwd=WORKDIR)


if __name__ == "__main__":
    main()
