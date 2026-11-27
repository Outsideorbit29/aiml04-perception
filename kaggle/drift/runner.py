"""Kaggle runner for F6: weekly drift report (PSI + Evidently) over val + test vs train (CPU job).

Attaches the F1 output (kernel anish2906/aiml01-data-prep), the F2 features
(anish2906/aiml01-features), the F3 model (anish2906/aiml01-train) and the secrets dataset
(dagshub_token, for MLflow). From the laptop:
    kaggle kernels push -p kaggle/drift
    smoke -> 60k train rows as reference, first 40k val rows as current
    full  -> all of train vs every week of val + test
Outputs: /kaggle/working/f6_drift (drift_summary.json, drift_reference.json, evidently/*.json,
evidently/{all,first,last}.html).
"""

import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "AIML-01"
MODE = "full"  # smoke | full
WORKDIR = "/kaggle/temp/repo"


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main() -> None:
    sh(["git", "clone", "--depth", "1", "--branch", REF, REPO, WORKDIR])
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKDIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    print("commit", sha, flush=True)
    pkgs = ["pyyaml", "pyarrow", "xgboost>=2", "evidently==0.7.23", "mlflow>=2.12,<3"]
    sh([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    cmd = [sys.executable, "-m", "scripts.drift_report", "--config", "configs/drift.yaml"]
    if MODE == "smoke":
        cmd.append("--smoke")
    sh(cmd, cwd=WORKDIR, env={**os.environ, "GIT_SHA": sha})


if __name__ == "__main__":
    main()
