"""Kaggle runner for F4: GraphSAGE + fraud rings + blend with the F3 XGBoost (T4 GPU job).

Attaches F2 features (anish2906/aiml01-features), the F3 model (anish2906/aiml01-train) and the
secrets dataset (dagshub_token). From the laptop:
    kaggle kernels push -p kaggle/gnn
    smoke -> 50k/10k/10k rows, 40 epochs; full -> everything
Outputs: /kaggle/working/f4_gnn (gnn_summary.json, ring_stats.json, scores.parquet,
rings.parquet, client_risk.parquet, gnn_state.pt).
"""

import os
import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "AIML-01"
MODE = "full"  # smoke | full
WORKDIR = "/kaggle/temp/repo"


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> None:
    sh(["git", "clone", "--depth", "1", "--branch", REF, REPO, WORKDIR])
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKDIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    print("commit", sha, flush=True)
    pkgs = ["pyyaml", "pyarrow", "xgboost>=2", "torch_geometric", "mlflow>=2.12,<3"]
    sh([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    sh([sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"])
    cmd = [sys.executable, "-m", "scripts.train_gnn", "--config", "configs/gnn.yaml"]
    if MODE == "smoke":
        cmd.append("--smoke")
    sh(cmd, cwd=WORKDIR, env={**os.environ, "GIT_SHA": sha})


if __name__ == "__main__":
    main()
