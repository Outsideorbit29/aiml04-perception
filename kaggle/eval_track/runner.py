"""Kaggle runner for M2.3 + M2.5: detection eval on BDD val and ByteTrack videos on KITTI (GPU).

Set MODE below, push the code, then from the laptop:
    kaggle kernels push -p kaggle/eval_track
    smoke -> 300 val images + one 30-frame clip (~10 min), proves the whole job
    full  -> all 10k val images + 3 KITTI sequences (~25-35 min)
Weights come from the W&B artifact yolo11s-bdd-best:candidate. The BDD data-prep output is
attached through kernel_sources, KITTI tracking through dataset_sources.
Outputs: /kaggle/working/m2_eval (results.json, plots, galleries) and /kaggle/working/m2_track
(results.json, MP4s, previews), plus two W&B runs (groups detect and track).
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "m2-eval-track"  # branch, tag or commit SHA to run
MODE = "full"  # smoke | full
EXTRA = ["--smoke"] if MODE == "smoke" else []
JOB_CMDS = [
    ["python", "-m", "scripts.eval_tracking", "--config", "configs/tracker_eval.yaml", *EXTRA],
    ["python", "-m", "scripts.eval_detection", "--config", "configs/yolo_eval.yaml", *EXTRA],
]
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
    # Run both steps even if one fails, then fail the job so the status shows it.
    failed = [cmd[2] for cmd in JOB_CMDS if sh(cmd, check=False, cwd=WORKDIR) != 0]
    if failed:
        raise SystemExit(f"failed steps: {failed}")


if __name__ == "__main__":
    main()
