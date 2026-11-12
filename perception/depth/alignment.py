"""Per-frame scale/shift alignment of relative MiDaS inverse depth to metric depth (M4.3).

MiDaS predicts inverse depth up to an unknown scale and shift, so for every frame we fit

    1 / Z = s * m + t            (m = MiDaS value, Z = metric depth, s > 0)

against anchors: objects whose flat-ground geometric distance is trusted (near, ground contact,
confident box). The fit is done in inverse-depth space, where the geometric error is roughly
constant (std(1/Z) = std(Z) / Z^2 = sqrt((sigma_v / fy)^2 + sigma_pitch^2) / h), so anchors get
equal weight. Robustness: iteratively reweighted least squares with Huber weights (residual
scale from the MAD, floored at ``sigma_floor``), then one refit on the inliers.

With fewer than ``min_points_affine`` anchors the shift cannot be estimated; we fall back to a
scale-only fit (t = 0, s = median(1/(Z m))) if ``allow_scale_only``. With no anchors there is no
alignment and MiDaS gives no metric distance. Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

REQUIRED_KEYS = (
    "min_points_affine",
    "allow_scale_only",
    "huber_k",
    "reject_k",
    "irls_iters",
    "sigma_floor",
    "min_inv_depth",
)


@dataclass(frozen=True)
class Alignment:
    """Fitted ``1/Z = s*m + t``. ``rms`` is the inlier residual RMS in 1/m."""

    s: float
    t: float
    rms: float
    n_anchors: int
    n_inliers: int
    mode: str  # "affine" | "scale"
    sigma_floor: float
    min_inv_depth: float

    def inv_depth(self, m: np.ndarray | float) -> np.ndarray:
        return self.s * np.asarray(m, dtype=np.float64) + self.t

    def to_depth(self, m: np.ndarray | float) -> np.ndarray:
        """Metric depth for MiDaS values m; NaN where 1/Z falls below ``min_inv_depth``."""
        inv = self.inv_depth(m)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(inv > self.min_inv_depth, 1.0 / inv, np.nan)

    def depth_std(self, z: np.ndarray | float) -> np.ndarray:
        """1-sigma error of an aligned depth z: the inverse-depth residual propagated to Z."""
        z = np.asarray(z, dtype=np.float64)
        return z * z * max(self.rms, self.sigma_floor)


class ScaleAligner:
    """``ScaleAligner(cfg["alignment"]).fit(m, z) -> Alignment | None``."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in cfg]
        if missing:
            raise KeyError(f"alignment config is missing {missing}")
        self.min_affine = int(cfg["min_points_affine"])
        if self.min_affine < 2:
            raise ValueError("min_points_affine must be >= 2 (two unknowns)")
        self.allow_scale_only = bool(cfg["allow_scale_only"])
        self.huber_k = float(cfg["huber_k"])
        self.reject_k = float(cfg["reject_k"])
        self.iters = int(cfg["irls_iters"])
        self.sigma_floor = float(cfg["sigma_floor"])
        self.min_inv_depth = float(cfg["min_inv_depth"])

    def fit(self, m: np.ndarray, z: np.ndarray) -> Alignment | None:
        """Fit on anchors with MiDaS values ``m`` and metric depths ``z`` (same length)."""
        m = np.asarray(m, dtype=np.float64).ravel()
        z = np.asarray(z, dtype=np.float64).ravel()
        if m.shape != z.shape:
            raise ValueError(f"m and z differ in length: {m.shape} vs {z.shape}")
        ok = np.isfinite(m) & np.isfinite(z) & (z > 0)
        m, y = m[ok], 1.0 / z[ok]
        n = m.size
        if n >= self.min_affine:
            res = self._fit_affine(m, y)
            if res is not None:
                return res
        if n >= 1 and self.allow_scale_only:
            return self._fit_scale(m, y)
        return None

    def _make(self, s: float, t: float, r: np.ndarray, n: int, mode: str) -> Alignment | None:
        if not np.isfinite(s) or not np.isfinite(t) or s <= 0:
            return None
        rms = float(np.sqrt(np.mean(r * r))) if r.size else 0.0
        return Alignment(s, t, rms, n, int(r.size), mode, self.sigma_floor, self.min_inv_depth)

    def _scale(self, r: np.ndarray) -> float:
        mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        return max(mad, self.sigma_floor)

    @staticmethod
    def _wls(m: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float] | None:
        a = np.stack([m, np.ones_like(m)], axis=1) * np.sqrt(w)[:, None]
        b = y * np.sqrt(w)
        if np.linalg.matrix_rank(a) < 2:
            return None
        (s, t), *_ = np.linalg.lstsq(a, b, rcond=None)
        return float(s), float(t)

    def _fit_affine(self, m: np.ndarray, y: np.ndarray) -> Alignment | None:
        w = np.ones_like(m)
        st = self._wls(m, y, w)
        if st is None:
            return None
        for _ in range(self.iters):
            r = y - (st[0] * m + st[1])
            u = np.abs(r) / (self.huber_k * self._scale(r))
            w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
            st = self._wls(m, y, w)
            if st is None:
                return None
        r = y - (st[0] * m + st[1])
        inl = np.abs(r) <= self.reject_k * self._scale(r)
        if inl.sum() >= self.min_affine and not inl.all():
            refit = self._wls(m[inl], y[inl], np.ones(int(inl.sum())))
            if refit is not None:
                st = refit
                r = y - (st[0] * m + st[1])
        return self._make(st[0], st[1], r[inl], m.size, "affine")

    def _fit_scale(self, m: np.ndarray, y: np.ndarray) -> Alignment | None:
        pos = m > 0
        if not pos.any():
            return None
        s = float(np.median(y[pos] / m[pos]))
        return self._make(s, 0.0, y[pos] - s * m[pos], m.size, "scale")
