"""Heterogeneous GraphSAGE over transactions and the client / card / device they belong to.

Torch + PyTorch Geometric; runs on Kaggle only (not imported by the CPU test suite).
Graph: one `tx` node per transaction with its model features; one node per distinct client,
card and device key; edges tx -> entity and back. Entity nodes start from a learned per-type
vector, so after one layer an entity holds the mean of its transactions and after two layers a
transaction sees its siblings through shared clients, cards and devices.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from fraud.metrics import average_precision


def node_type(key_column: str) -> str:
    return key_column.removesuffix("_key")


def prepare_tx_features(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Signed log1p, then standardise with training-split statistics; NaN -> 0 (the mean)."""
    z = np.sign(x) * np.log1p(np.abs(x))
    mean = np.nanmean(z[train_mask], axis=0)
    std = np.nanstd(z[train_mask], axis=0)
    mean = np.nan_to_num(mean)
    std = np.where((std > 0) & np.isfinite(std), std, 1.0)
    return np.nan_to_num((z - mean) / std).astype("float32")


def build_hetero(
    x_tx: np.ndarray, enc: Mapping[str, tuple[np.ndarray, np.ndarray, int]]
) -> HeteroData:
    data = HeteroData()
    data["tx"].x = torch.from_numpy(x_tx)
    for col, (tx_idx, ent_idx, n) in enc.items():
        t = node_type(col)
        data[t].num_nodes = n
        edge = torch.from_numpy(np.vstack([tx_idx, ent_idx]))
        data["tx", f"to_{t}", t].edge_index = edge
        data[t, f"of_{t}", "tx"].edge_index = edge.flip(0)
    return data


class HeteroSAGE(nn.Module):
    def __init__(self, data: HeteroData, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.entity_types = [t for t in data.node_types if t != "tx"]
        self.tx_in = nn.Linear(data["tx"].x.shape[1], hidden)
        self.entity_init = nn.ParameterDict(
            {t: nn.Parameter(torch.zeros(1, hidden)) for t in self.entity_types}
        )
        self.convs = nn.ModuleList(
            HeteroConv(
                {et: SAGEConv((hidden, hidden), hidden) for et in data.edge_types}, aggr="sum"
            )
            for _ in range(layers)
        )
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, data: HeteroData) -> torch.Tensor:
        h = {"tx": F.relu(self.tx_in(data["tx"].x))}
        for t in self.entity_types:
            h[t] = self.entity_init[t].expand(data[t].num_nodes, -1)
        for conv in self.convs:
            out = conv(h, data.edge_index_dict)
            h = {k: F.dropout(F.relu(v), self.dropout, self.training) for k, v in out.items()}
        return self.head(h["tx"]).squeeze(-1)


def train_gnn(
    data: HeteroData,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Mapping[str, Any],
    device: str,
    log: Callable[[dict], None] | None = None,
) -> tuple[HeteroSAGE, np.ndarray, dict]:
    """Full-batch training with early stopping on val PR-AUC; returns (model, probs, info)."""
    data = data.to(device)
    model = HeteroSAGE(data, cfg["hidden"], cfg["layers"], cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    y_t = torch.from_numpy(y.astype("float32")).to(device)
    tr = torch.from_numpy(train_idx).to(device)
    pos = float(y[train_idx].sum())
    pos_weight = torch.tensor((len(train_idx) - pos) / max(pos, 1.0), device=device)
    best = {"ap": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        opt.zero_grad()
        logits = model(data)
        loss = F.binary_cross_entropy_with_logits(logits[tr], y_t[tr], pos_weight=pos_weight)
        loss.backward()
        opt.step()
        if epoch % cfg["eval_every"] == 0:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(data)).cpu().numpy()
            ap = average_precision(y[val_idx], p[val_idx])
            row = {"epoch": epoch, "loss": float(loss), "val_pr_auc": ap}
            history.append(row)
            if log:
                log(row)
            if ap > best["ap"]:
                best = {"ap": ap, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            elif epoch - best["epoch"] >= cfg["patience"]:
                break
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data)).cpu().numpy()
    return (
        model,
        probs,
        {"best_epoch": best["epoch"], "best_val_pr_auc": best["ap"], "history": history},
    )
