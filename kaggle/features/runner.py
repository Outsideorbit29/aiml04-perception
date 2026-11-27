"""Kaggle runner for F2: card/device features, offline/online parity, Feast check (CPU job).

Attaches the F1 output (kernel anish2906/aiml01-data-prep) for splits/*.parquet.
    kaggle kernels push -p kaggle/features
    smoke -> first 20k transactions; check features_summary.json and feast_check.json
    full  -> all 590k
Outputs: /kaggle/working/f2_features (features/*.parquet, card_features.parquet,
features_summary.json, feast_check.json). The F3 training job attaches this kernel's output.
"""

import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "AIML-01"
MODE = "full"  # smoke | full
WORKDIR = "/kaggle/temp/repo"
OUT = "/kaggle/working/f2_features"


def sh(cmd: list[str], check: bool = True, **kw) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, **kw).returncode


def main() -> None:
    sh(["git", "clone", "--depth", "1", "--branch", REF, REPO, WORKDIR])
    sh(["git", "rev-parse", "HEAD"], cwd=WORKDIR)
    sh([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "pyarrow"])
    cmd = [sys.executable, "-m", "scripts.build_features", "--out-dir", OUT]
    if MODE == "smoke":
        cmd.append("--smoke")
    sh(cmd, cwd=WORKDIR)
    # Feast after the features are saved, so a Feast install problem can't lose them.
    if sh([sys.executable, "-m", "pip", "install", "-q", "feast"], check=False) == 0:
        sh(
            [sys.executable, "-m", "scripts.feast_check", "--features-dir", OUT],
            cwd=WORKDIR,
            check=False,
        )
    else:
        print("feast install failed; skipping the Feast check", flush=True)


if __name__ == "__main__":
    main()
