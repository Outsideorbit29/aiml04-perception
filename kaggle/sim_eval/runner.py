"""Kaggle runner for M5.6: perception stack on the recorded CARLA frames (GPU, T4).

Set MODE (and REF if the code is on a branch), push the code, then from the laptop:
    kaggle kernels push -p kaggle/sim_eval
    smoke -> every 30th frame (100, all weathers); check results.json, sim_eval.md and
             snapshots/*.jpg before "full"
    full  -> all 3000 frames
Frames come from the private dataset anish2906/aiml04-carla-frames (sim/record_dataset.py).
Detector weights come from the W&B artifact yolo11s-bdd-best:latest; COCO yolo11s and
MiDaS-small (torch.hub) are downloaded inside the job (internet on); nothing is downloaded to
the laptop. Outputs: /kaggle/working/m5_sim_eval/<mode> (results.json, sim_eval.md,
per_object.jsonl, snapshots/*.jpg, Ultralytics val plots) plus one W&B run (project
aiml04-perception, group sim_eval).
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "m5-sim-eval"  # branch, tag or commit SHA to run; set back to "main" before merging
MODE = "smoke"  # smoke | full
EXTRA = ["--smoke"] if MODE == "smoke" else []
JOB_CMDS = [
    ["python", "-m", "sim.eval_frames", "--config", "configs/carla_eval.yaml", *EXTRA],
]
# intel-isl/MiDaS (v3.1 hubconf) was written against timm 0.6.x; newer timm releases are known
# to break it. Pinned for this job only (requirements.txt keeps timm>=1.0), as in eval_depth.
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
    # Shallow patterns only: a recursive glob would walk the frame folders.
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


def probe_inputs() -> None:
    """Print the dataset mount layout (3 levels, at most 12 entries per folder)."""
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count("/") - 2
        if depth > 5:
            dirs[:] = []
            continue
        entries = sorted(dirs) + sorted(files)
        more = f" ... (+{len(entries) - 12})" if len(entries) > 12 else ""
        print(f"[input] {root}: {entries[:12]}{more}", flush=True)
        dirs[:] = [d for d in dirs if d not in ("images", "labels", "gt", "depth")]


def main() -> None:
    load_wandb_key()
    probe_inputs()
    if not os.path.exists(WORKDIR):
        sh(["git", "clone", "--depth", "50", REPO, WORKDIR])
    # A shallow clone has only the default branch, so fetch REF explicitly.
    sh(["git", "fetch", "--depth", "50", "origin", REF], cwd=WORKDIR)
    sh(["git", "checkout", "FETCH_HEAD"], cwd=WORKDIR)
    # Commit under test; sim/eval_frames.py logs it (results.json git_sha and the W&B config)
    # and falls back to this env var if git is unavailable in its process.
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
    failed = [cmd[2] for cmd in JOB_CMDS if sh(cmd, check=False, cwd=WORKDIR) != 0]
    if failed:
        raise SystemExit(f"failed steps: {failed}")


if __name__ == "__main__":
    main()
