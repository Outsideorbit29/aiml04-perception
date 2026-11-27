"""Graph construction, fraud-ring detection and score blending (numpy only; no torch here).

- `encode_entities`: map each key column to integer node ids and transaction->entity edges
- `find_rings`: clients connected through specific (low-degree) shared device fingerprints
- `blend_logits` / `best_blend`: combine XGBoost and GNN probabilities on the logit scale
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from fraud.metrics import average_precision


def encode_entities(
    df: pd.DataFrame, columns: Sequence[str]
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """For each key column: (tx_index, entity_id, n_entities) for rows whose key is not NaN."""
    out = {}
    for col in columns:
        s = df[col]
        mask = s.notna().to_numpy()
        codes, uniques = pd.factorize(s[mask], sort=False)
        out[col] = (np.flatnonzero(mask).astype(np.int64), codes.astype(np.int64), len(uniques))
    return out


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, a: int) -> int:
        p = self.parent
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:  # path compression
            p[a], a = root, p[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def find_rings(
    clients: pd.Series,
    devices: pd.Series,
    max_clients_per_device: int,
    min_size: int,
) -> pd.DataFrame:
    """Groups of clients linked by shared specific devices.

    A device links its clients only if it was used by 2..max_clients_per_device distinct clients
    (generic fingerprints such as "Windows|Chrome" would otherwise join everyone).
    Returns one row per client in a ring: client_key, ring_id, ring_size, ring_devices.
    """
    pairs = pd.DataFrame({"client": clients.values, "device": devices.values}).dropna()
    pairs = pairs.drop_duplicates()
    per_device = pairs.groupby("device")["client"].nunique()
    specific = per_device[(per_device >= 2) & (per_device <= max_clients_per_device)].index
    pairs = pairs[pairs["device"].isin(specific)]
    if pairs.empty:
        return pd.DataFrame(columns=["client_key", "ring_id", "ring_size", "ring_devices"])
    client_codes, client_names = pd.factorize(pairs["client"])
    uf = _UnionFind(len(client_names))
    for _, idx in pairs.groupby("device").indices.items():
        members = client_codes[idx]
        for other in members[1:]:
            uf.union(int(members[0]), int(other))
    roots = np.array([uf.find(i) for i in range(len(client_names))])
    clients_df = pd.DataFrame({"client_key": client_names, "root": roots})
    sizes = clients_df.groupby("root")["client_key"].transform("size")
    clients_df = clients_df[sizes >= min_size].copy()
    if clients_df.empty:
        return pd.DataFrame(columns=["client_key", "ring_id", "ring_size", "ring_devices"])
    root_of = dict(zip(client_names, roots, strict=True))
    pairs = pairs.assign(root=pairs["client"].map(root_of))
    n_dev = pairs.groupby("root")["device"].nunique()
    ring_ids = {r: i for i, r in enumerate(sorted(clients_df["root"].unique()))}
    clients_df["ring_id"] = clients_df["root"].map(ring_ids)
    clients_df["ring_size"] = clients_df.groupby("root")["client_key"].transform("size")
    clients_df["ring_devices"] = clients_df["root"].map(n_dev).astype(int)
    return clients_df.drop(columns="root").reset_index(drop=True)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return np.log(p / (1 - p))


def blend_logits(p_xgb: np.ndarray, p_gnn: np.ndarray, w: float) -> np.ndarray:
    """Probability from w * logit(xgb) + (1 - w) * logit(gnn)."""
    z = w * logit(p_xgb) + (1 - w) * logit(p_gnn)
    return 1 / (1 + np.exp(-z))


def best_blend(
    y: np.ndarray, p_xgb: np.ndarray, p_gnn: np.ndarray, grid: Sequence[float]
) -> tuple[float, Mapping[float, float]]:
    """Weight from `grid` with the highest PR-AUC on (y, scores); returns (w, {w: pr_auc})."""
    scores = {float(w): average_precision(y, blend_logits(p_xgb, p_gnn, w)) for w in grid}
    return max(scores, key=scores.get), scores
