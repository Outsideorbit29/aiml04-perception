"""Generic Kaggle runner: clone the repo, install deps, run one project command.

Copy this folder to kaggle/<job>/, edit JOB_CMD and kernel-metadata.json, then from the laptop:
    kaggle kernels push -p kaggle/<job>
Everything heavy runs on Kaggle's GPU; the laptop only sends this small folder.
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "main"  # branch, tag or commit SHA to run
CHECK = (
    "import torch, perception.types as t; "
    "print('cuda', torch.cuda.is_available(), "
    "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); "
    "print('repo ok', t.Detection.__name__); "
    "import wandb; r = wandb.init(project='aiml04-perception', job_type='smoke-test'); "
    "wandb.log({'smoke': 1}); print('wandb url', r.url, 'mode', r.settings.mode); r.finish()"
)
JOB_CMD = ["python", "-c", CHECK]
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
    sh(JOB_CMD, cwd=WORKDIR)


if __name__ == "__main__":
    main()
