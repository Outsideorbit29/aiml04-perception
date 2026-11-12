"""Kaggle runner for M3.3 + M3.4: classical vs YOLOP lanes on 500 BDD val images (GPU).

Set MODE (and REF if the code is on a branch), push the code, then from the laptop:
    kaggle kernels push -p kaggle/eval_lanes
    smoke -> 30 images (10 per time of day), proves the whole job (~10 min incl. pip install)
    full  -> 500 images stratified by time of day (~20 min)
BDD val images come from awsaf49/bdd100k-dataset; lane and drivable-area labels from the 2018
label file in solesensei/solesensei_bdd100k. YOLOP ONNX weights are downloaded from the
hustvl/YOLOP GitHub repo inside the job (internet on); nothing is downloaded to the laptop.
Outputs: /kaggle/working/m3_lanes (results.json, per_image.jsonl, lanes_comparison.md,
lanes_failures.jpg; smoke also debug/*.jpg overlays) plus one W&B run (project
aiml04-perception, group lanes). Check results.json "gt_geometry" and the diagnostics table
(GT empty / no pred) before launching the full run.
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "m3-lanes-eval"  # branch, tag or commit SHA to run
MODE = "smoke"  # smoke | full
EXTRA = ["--smoke"] if MODE == "smoke" else []
JOB_CMDS = [
    ["python", "-m", "scripts.eval_lanes", "--config", "configs/lanes_eval.yaml", *EXTRA],
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
       and can be uploaded later with `wandb sync`.
    """
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        print("W&B key loaded from Kaggle Secrets")
        return
    except Exception as exc:
        print(f"Kaggle Secrets unavailable ({type(exc).__name__})")
    # Shallow patterns only: a recursive glob would walk the BDD image trees.
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
    # Commit under test; scripts/eval_lanes.py logs it (results.json setup.git_sha and the W&B
    # config) and falls back to this env var if git is unavailable in its process.
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKDIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    os.environ["GIT_COMMIT"] = sha
    print(f"code commit {sha} (REF {REF})", flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORKDIR)
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    # Run every step even if one fails, then fail the job so the status shows it.
    failed = [cmd[2] for cmd in JOB_CMDS if sh(cmd, check=False, cwd=WORKDIR) != 0]
    if failed:
        raise SystemExit(f"failed steps: {failed}")


if __name__ == "__main__":
    main()
