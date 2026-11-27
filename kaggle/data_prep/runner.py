"""Kaggle runner for F1: IEEE-CIS load + join, time split, EDA summary and figures (CPU job).

Push the code first (branch AIML-01), then from the laptop:
    kaggle kernels push -p kaggle/data_prep
    smoke -> first 50k transactions; check split_summary.json and eda_summary.json first
    full  -> all 590k labelled transactions
Needs the IEEE-CIS competition rules accepted on kaggle.com (competition_sources).
Outputs: /kaggle/working/f1_data (splits/*.parquet, split_summary.json, eda_summary.json,
figures/*.png). Later jobs attach this kernel's output via kernel_sources.
"""

import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "AIML-01"  # branch, tag or commit SHA
MODE = "full"  # smoke | full
WORKDIR = "/kaggle/temp/repo"  # outside /kaggle/working so the code is not saved as output


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> None:
    sh(["git", "clone", "--depth", "1", "--branch", REF, REPO, WORKDIR])
    sh(["git", "rev-parse", "HEAD"], cwd=WORKDIR)
    sh([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "pyarrow"])
    cmd = [sys.executable, "-m", "scripts.prep_data", "--config", "configs/data.yaml"]
    if MODE == "smoke":
        cmd.append("--smoke")
    sh(cmd, cwd=WORKDIR)


if __name__ == "__main__":
    main()
