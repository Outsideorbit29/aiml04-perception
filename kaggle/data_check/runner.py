"""Kaggle runner for the M1 sanity check: train 1 tiny epoch on the data-prep output.

Launch from the laptop after the data-prep job has finished and the code is pushed:
    kaggle kernels push -p kaggle/data_check
The data-prep output is attached through kernel_sources in kernel-metadata.json.
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "main"  # branch, tag or commit SHA to run
RESOLVED = "/kaggle/working/data_bdd_resolved.yaml"
COPY_ARTIFACTS = (
    "set -e; Y=$(find /kaggle/input -path '*bdd_yolo/data.yaml' | head -1); "
    'P=$(dirname "$(dirname "$Y")"); A=/kaggle/working/m1_artifacts; mkdir -p $A; '
    'cp -r "$P/eda" "$P/fixtures" $A/; cp "$P/bdd_yolo/meta/prep_summary.json" $A/; '
    "cat $A/prep_summary.json; cat $A/eda/eda_summary.md; cat $A/fixtures/fixtures_manifest.json; "
    "du -sh $A/*"
)
# M1 sanity check (GPU, <10 min): the data-prep output trains with Ultralytics.
JOB_CMDS = [
    [
        "bash",
        "-c",
        "find /kaggle/input -maxdepth 5 -type d | grep -vE '/(images|labels)/' | head -40",
    ],
    [
        "python",
        "-m",
        "scripts.data.check_data_yaml",
        "--config",
        "configs/data_bdd.yaml",
        "--search",
        "/kaggle/input",
        "--out",
        RESOLVED,
    ],
    # Copy the small data-prep artifacts into this job's output: the data-prep output has ~60k
    # files, and listing it through the API hits the rate limit.
    ["bash", "-c", COPY_ARTIFACTS],
    [
        "yolo",
        "train",
        "model=yolo11n.pt",
        f"data={RESOLVED}",
        "epochs=1",
        "fraction=0.01",
        "imgsz=320",
        "batch=32",
        "workers=2",
        "project=/kaggle/working/runs",
        "name=sanity",
        "exist_ok=True",
        "plots=False",
    ],
]
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
    os.environ["WANDB_MODE"] = "disabled"  # sanity check only, nothing to log
    for cmd in JOB_CMDS:
        sh(cmd, cwd=WORKDIR)


if __name__ == "__main__":
    main()
