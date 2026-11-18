"""Kaggle runner for M5.9: fine-tune the BDD YOLO11s for 5 epochs on a BDD + CARLA mix (GPU, T4).

Set MODE (and REF if the code is on a branch), push the code, then from the laptop:
    kaggle kernels push -p kaggle/train_mix
    smoke  -> 1 epoch at imgsz 320 on 640 BDD + every 10th CARLA-train frame, then the whole eval
              path and the ONNX export on small subsets (~15 min); check results.json, mix_eval.md
    train  -> pseudo labels, 5 epochs, base vs mix eval on BDD val (10k) and CARLA-val, W&B
              artifacts yolo11s-mix-best + yolo11s-mix-onnx
    resume -> continue from the newest W&B checkpoint (yolo11s-mix-ckpt) in a new session
BDD comes from the data-prep kernel output (kernel_sources), CARLA frames from the private dataset
anish2906/aiml04-carla-frames, the base weights from W&B yolo11s-bdd-best:latest. Outputs:
/kaggle/working/m5_9/<smoke|full> (results.json, mix_eval.md, mix.pt, Ultralytics plots).
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "m5-9-mix-finetune"  # branch, tag or commit SHA to run; set back to "main" before merging
MODE = "smoke"  # smoke | train | resume
EXTRA = [] if MODE == "train" else [f"--{MODE}"]
JOB_CMD = ["python", "-m", "scripts.train_mix", "--config", "configs/yolo_mix.yaml", *EXTRA]
JOB_PIP = ["onnxslim"]  # Ultralytics ONNX simplify (deploy/export.py export_yolo)
WORKDIR = "/kaggle/temp/repo"  # outside /kaggle/working so the code is not saved as job output


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def load_wandb_key() -> None:
    """Find the W&B key: Kaggle Secrets, else `wandb_api_key` in the attached private dataset
    anish2906/aiml04-secrets (pushed runs get no secrets), else log offline (sync later)."""
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        print("W&B key loaded from Kaggle Secrets")
        return
    except Exception as exc:
        print(f"Kaggle Secrets unavailable ({type(exc).__name__})")
    # Shallow patterns only: a recursive glob would walk the BDD and CARLA frame folders.
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
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKDIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    os.environ["GIT_COMMIT"] = sha
    print(f"code commit {sha} (REF {REF})", flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORKDIR)
    sh([sys.executable, "-m", "pip", "install", "-q", *JOB_PIP], cwd=WORKDIR)
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    sh(JOB_CMD, cwd=WORKDIR)


if __name__ == "__main__":
    main()
