"""Shared helpers for the services: config with env overrides, JSON-safe rows, percentiles."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fraud.config import load_config


def serving_config(path: str | None = None) -> dict[str, Any]:
    """configs/serving.yaml with each key overridable by an upper-case environment variable."""
    cfg = load_config(path or os.environ.get("SERVING_CONFIG", "configs/serving.yaml"))
    for key, value in list(cfg.items()):
        env = os.environ.get(key.upper())
        if env is not None:
            cfg[key] = type(value)(env) if isinstance(value, (int, float)) else env
    return cfg


def json_safe(row: Mapping[str, Any]) -> dict[str, Any]:
    """NaN / numpy scalars -> JSON-friendly Python values (NaN becomes None)."""
    out = {}
    for k, v in row.items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and math.isnan(v):
            v = None
        out[k] = v
    return out


def percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values, dtype="float64")
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }
