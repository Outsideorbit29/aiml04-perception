"""Kaggle runner for M1 data prep: clone the repo, install deps, run the data scripts.

M1 data prep job (CPU). Launch from the laptop after git push:
    kaggle kernels push -p kaggle/data_prep
Outputs (/kaggle/working/bdd_yolo, eda/, fixtures/) are attached to later jobs with
kernel_sources: ["anish2906/aiml04-data-prep"].
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "main"  # branch, tag or commit SHA to run
KITTI_SEQ = "/kaggle/input/datasets/leducnhuan/kitti-tracking/kitti_tracking/training/image_02/0000"
# M1 data prep (CPU only): BDD100K -> YOLO subset, EDA figures, test fixtures.
JOB_CMDS = [
    ["python", "-m", "scripts.data.prepare_bdd", "--config", "configs/data_bdd.yaml"],
    [
        "python",
        "-m",
        "scripts.data.eda_bdd",
        "--data",
        "/kaggle/working/bdd_yolo",
        "--out",
        "/kaggle/working/eda",
    ],
    [
        "python",
        "-m",
        "scripts.data.make_fixtures",
        "--data",
        "/kaggle/working/bdd_yolo",
        "--kitti-seq",
        KITTI_SEQ,
        "--out",
        "/kaggle/working/fixtures",
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
    sh(["git", "fetch", "--depth", "50", "origin", REF], cwd=WORKDIR)  # shallow clone has only the default branch
    sh(["git", "checkout", "FETCH_HEAD"], cwd=WORKDIR)
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORKDIR)
    sh(["ls", "/kaggle/input", "/kaggle/input/datasets"])
    for cmd in JOB_CMDS:
        sh(cmd, cwd=WORKDIR)
    sh(["du", "-sh", "/kaggle/working/bdd_yolo", "/kaggle/working/eda", "/kaggle/working/fixtures"])


if __name__ == "__main__":
    main()
