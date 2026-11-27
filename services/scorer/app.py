"""FastAPI fraud scorer.

POST /score  {raw transaction}  -> fetches the transaction's F2 features from the online store
(pushed by the consumer), applies the F3 model + F4 client risk, returns score and alert.
GET /health, GET /model. Run: uvicorn services.scorer.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from fraud.config import load_config
from fraud.features import feature_names, row_keys
from fraud.scoring import Scorer
from fraud.store import FeastStore, MemoryStore, OnlineStore
from services.common import serving_config

STATE: dict[str, Any] = {}


def build_store(cfg: dict, feat_cfg: dict) -> OnlineStore:
    if os.environ.get("ONLINE_STORE", "feast") == "memory":
        return MemoryStore()
    return FeastStore(
        cfg["feature_repo"],
        feature_names(feat_cfg["windows"]),
        push_source="client_features_push",
        view="client_features",
        fs_yaml=cfg["feature_store_yaml"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = serving_config()
    feat_cfg = load_config(os.environ.get("FEATURES_CONFIG", "configs/features.yaml"))
    STATE.update(
        cfg=cfg,
        feat_cfg=feat_cfg,
        scorer=Scorer(cfg["model_dir"]),
        store=STATE.get("store") or build_store(cfg, feat_cfg),
        started=time.time(),
        scored=0,
    )
    yield


app = FastAPI(title="AIML-01 fraud scorer", version="0.1.0", lifespan=lifespan)


class Transaction(BaseModel):
    """Raw transaction as replayed from Kafka; extra fields (C1, card4, …) are allowed."""

    model_config = ConfigDict(extra="allow")
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float


class ScoreResponse(BaseModel):
    TransactionID: int
    client_key: str
    score: float
    p_xgb: float
    gnn_risk: float | None
    alert: bool
    threshold: float
    fetch_ms: float
    model_ms: float
    total_ms: float


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "uptime_s": time.time() - STATE["started"], "scored": STATE["scored"]}


@app.get("/model")
def model() -> dict[str, Any]:
    return STATE["scorer"].info()


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction) -> ScoreResponse:
    t0 = time.perf_counter()
    raw = tx.model_dump()
    client_key, _, _ = row_keys(raw, STATE["feat_cfg"])
    features = STATE["store"].get(client_key)
    fetch_ms = (time.perf_counter() - t0) * 1000
    if features is None or int(features.get("TransactionID", -1)) != tx.TransactionID:
        raise HTTPException(
            status_code=409,
            detail=f"features for transaction {tx.TransactionID} not in the online store yet",
        )
    out = STATE["scorer"].score(raw, features)
    STATE["scored"] += 1
    return ScoreResponse(
        TransactionID=tx.TransactionID,
        client_key=client_key,
        fetch_ms=fetch_ms,
        total_ms=(time.perf_counter() - t0) * 1000,
        **{k: out[k] for k in ("score", "p_xgb", "gnn_risk", "alert", "threshold", "model_ms")},
    )
