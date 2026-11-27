"""Kaggle runner for F5: bundle model, client risk, history and replay data for the serving stack.

    kaggle kernels push -p kaggle/serving_bundle
Output: /kaggle/working/serving_bundle (download with scripts/fetch_serving_bundle.sh).
"""

import subprocess
import sys

REPO = "https://github.com/Outsideorbit29/aiml04-perception.git"
REF = "AIML-01"
WORKDIR = "/kaggle/temp/repo"


def sh(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> None:
    sh(["git", "clone", "--depth", "1", "--branch", REF, REPO, WORKDIR])
    sh(["git", "rev-parse", "HEAD"], cwd=WORKDIR)
    sh([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "pyarrow"])
    sh([sys.executable, "-m", "scripts.export_serving_bundle"], cwd=WORKDIR)


if __name__ == "__main__":
    main()
