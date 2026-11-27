"""MLflow tracking setup: DagsHub when a token is available, local file store otherwise."""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def find_token(token_file: str, search_root: str | Path = "/kaggle/input") -> str | None:
    """Token from MLFLOW_TRACKING_PASSWORD, else from `token_file` in an attached dataset."""
    if os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        return os.environ["MLFLOW_TRACKING_PASSWORD"]
    # Shallow patterns only: a recursive glob would walk large competition trees.
    for depth in range(1, 5):
        pattern = os.path.join(str(search_root), *["*"] * depth, token_file)
        for path in sorted(glob.glob(pattern)):
            token = Path(path).read_text(encoding="utf-8").strip()
            if token:
                return token
    return None


def configure(cfg: Mapping[str, Any], search_root: str | Path = "/kaggle/input") -> str:
    """Set the MLflow env vars and return the tracking URI in use (remote or fallback).

    Only environment variables are set, so this works before `mlflow` is imported and is
    testable without mlflow installed.
    """
    token = find_token(cfg["token_file"], search_root)
    if token:
        os.environ["MLFLOW_TRACKING_URI"] = cfg["tracking_uri"]
        os.environ["MLFLOW_TRACKING_USERNAME"] = cfg["username"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        uri = cfg["tracking_uri"]
        print(f"MLflow: logging to {uri}", flush=True)
    else:
        uri = cfg["fallback_uri"]
        os.environ["MLFLOW_TRACKING_URI"] = uri
        print(f"MLflow: no token found, logging to {uri}", flush=True)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg["experiment"]
    return uri
