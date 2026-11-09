"""Generic Kaggle runner: clone the repo, install deps, run one project command.

Copy this folder to kaggle/<job>/, edit JOB_CMD and kernel-metadata.json, then from the laptop:
    kaggle kernels push -p kaggle/<job>
Everything heavy runs on Kaggle's GPU; the laptop only sends this small folder.
"""

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
    """Read WANDB_API_KEY from Kaggle Secrets (Add-ons > Secrets in the notebook editor)."""
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        print("W&B key loaded from Kaggle Secrets")
    except Exception as exc:  # secret not attached: run offline instead of failing
        os.environ["WANDB_MODE"] = "offline"
        print(f"W&B secret unavailable ({exc}); logging offline")


def main() -> None:
    load_wandb_key()
    if not os.path.exists(WORKDIR):
        sh(["git", "clone", "--depth", "50", REPO, WORKDIR])
    sh(["git", "checkout", REF], cwd=WORKDIR)
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORKDIR)
    sh(JOB_CMD, cwd=WORKDIR)


if __name__ == "__main__":
    main()
