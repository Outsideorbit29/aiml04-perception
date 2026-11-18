"""Locate the AIML-04 repo so the nodes can import ``perception`` and ``sim`` and read configs/.

Order: ``$AIML04_REPO_ROOT``, then the parents of this file's real path (with
``colcon build --symlink-install`` the installed module links back to the repo checkout).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV = "AIML04_REPO_ROOT"


def _is_repo(p: Path) -> bool:
    return (p / "perception" / "types.py").is_file() and (p / "configs").is_dir()


def repo_root() -> Path:
    candidates: list[Path] = []
    if os.environ.get(ENV):
        candidates.append(Path(os.environ[ENV]))
    candidates += list(Path(os.path.realpath(__file__)).parents)
    for p in candidates:
        if _is_repo(p):
            return p.resolve()
    raise RuntimeError(
        f"AIML-04 repo not found: set {ENV}=/mnt/d/Project_ai (the folder holding perception/)"
    )


def ensure_repo_on_path() -> Path:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
