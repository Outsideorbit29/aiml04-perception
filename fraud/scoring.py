"""Online scoring: F3 XGBoost + F4 client GNN risk, blended exactly as evaluated offline.

The scorer is given the raw transaction and its F2 features (fetched from the online store) and
returns the fraud probability and the alert decision. No Kafka, HTTP or Feast here, so it is
unit-testable; `services/scorer` wraps it in FastAPI.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from fraud.model import FeatureSpec


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


class Scorer:
    """Loads the serving artefacts from `model_dir`.

    Required: model.json (XGBoost), spec.json, threshold.json.
    Optional: blend.json {"weight_xgb": w, "threshold": t} and client_risk.parquet
    (client_key, gnn_risk). Without them the XGBoost score and threshold are used alone.
    """

    def __init__(self, model_dir: str | Path):
        import xgboost as xgb

        model_dir = Path(model_dir)
        self.spec = FeatureSpec.load(model_dir / "spec.json")
        # Plain Booster (no scikit-learn needed at serving); inplace_predict returns P(fraud).
        self.booster = xgb.Booster()
        self.booster.load_model(model_dir / "model.json")
        meta = json.loads((model_dir / "threshold.json").read_text())
        self.model_name = meta.get("experiment", "xgb")
        self.threshold = float(meta["threshold"])
        self.weight_xgb = 1.0
        blend_path = model_dir / "blend.json"
        if blend_path.exists():
            blend = json.loads(blend_path.read_text())
            self.weight_xgb = float(blend["weight_xgb"])
            self.threshold = float(blend["threshold"])
        self.risk: dict[str, float] = {}
        risk_path = model_dir / "client_risk.parquet"
        if risk_path.exists() and self.weight_xgb < 1.0:
            import pandas as pd

            r = pd.read_parquet(risk_path)
            self.risk = dict(zip(r["client_key"], r["gnn_risk"].astype(float), strict=True))

    def info(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "n_features": len(self.spec.columns),
            "threshold": self.threshold,
            "weight_xgb": self.weight_xgb,
            "clients_with_risk": len(self.risk),
        }

    def score(self, tx: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        row = {**tx, **features}
        x = self.spec.transform_row(row)
        p_xgb = float(self.booster.inplace_predict(x)[0])
        risk = self.risk.get(str(features.get("client_key", tx.get("client_key"))))
        if risk is not None and self.weight_xgb < 1.0:
            z = self.weight_xgb * _logit(p_xgb) + (1 - self.weight_xgb) * _logit(risk)
            p = 1 / (1 + math.exp(-z))
        else:
            p = p_xgb  # unseen client: XGBoost alone (as evaluated offline)
        return {
            "score": p,
            "p_xgb": p_xgb,
            "gnn_risk": risk,
            "alert": bool(p > self.threshold),
            "threshold": self.threshold,
            "model_ms": (time.perf_counter() - t0) * 1000,
        }


def score_matrix(scorer: Scorer, x: np.ndarray) -> np.ndarray:
    """Batch XGBoost probabilities (used by tests and the load test)."""
    return scorer.booster.inplace_predict(x)
