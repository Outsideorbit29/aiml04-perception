"""Kaggle runner for F3: classifiers on the F2 features, tracked in MLflow on DagsHub (CPU job).

Attaches the F2 output (kernel anish2906/aiml01-features) and the private secrets dataset
(dagshub_token). From the laptop:
    kaggle kernels push -p kaggle/train
    smoke -> 60k training rows, 300 trees, no model registration
    full  -> all rows, early stopping, registers aiml01-xgb
Outputs: /kaggle/working/f3_model (train_summary.json, model.json, spec.json, threshold.json,
pr_curve_val.png, importance_*.csv).
"""

import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "AIML-01"
MODE = "smoke"  # smoke | full
WORKDIR = "/kaggle/temp/repo"


def sh(cmd: list[str], **kw) -> str:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main() -> None:
    sh(["git", "clone", "--depth", "1", "--branch", REF, REPO, WORKDIR])
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKDIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    print("commit", sha, flush=True)
    pkgs = [
        "pyyaml",
        "pyarrow",
        "xgboost>=2",
        "scikit-learn",
        "imbalanced-learn",
        "mlflow>=2.12,<3",
    ]
    sh([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    cmd = [sys.executable, "-m", "scripts.train", "--config", "configs/train.yaml"]
    if MODE == "smoke":
        cmd.append("--smoke")
    sh(cmd, cwd=WORKDIR, env={**os.environ, "GIT_SHA": sha})


if __name__ == "__main__":
    main()
