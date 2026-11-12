"""Temporal smoothing of lane boundaries across video frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.types import Polyline


@dataclass
class _SideState:
    coeffs: np.ndarray | None = None
    conf: float = 0.0
    missed: int = 0


class LaneSmoother:
    """Per-side EMA on image-space coefficients with hold-over on missing frames.

    - A new fit is blended as `alpha * new + (1 - alpha) * previous` (linear in x(y)).
    - A fit whose bottom-row x jumps more than `max_jump_frac * W` is treated as missing.
    - A missing side keeps its last value for up to `max_missed` frames, with confidence
      multiplied by `missed_conf_decay` each frame, then it is dropped (None, confidence 0).

    `cfg` is the `smoothing` section of configs/lanes.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self.enabled = bool(cfg["enabled"])
        self.alpha = float(cfg["alpha"])
        self.max_missed = int(cfg["max_missed"])
        self.decay = float(cfg["missed_conf_decay"])
        self.max_jump_frac = float(cfg["max_jump_frac"])
        self._state = {"left": _SideState(), "right": _SideState()}

    def reset(self) -> None:
        self._state = {"left": _SideState(), "right": _SideState()}

    def update(
        self,
        left: Polyline | None,
        right: Polyline | None,
        conf_left: float,
        conf_right: float,
        y_ref: float,
        width: int,
    ) -> tuple[Polyline | None, Polyline | None, float, float]:
        """Blend this frame's fits into the running state and return the smoothed sides."""
        if not self.enabled:
            c_l = float(conf_left) if left is not None else 0.0
            c_r = float(conf_right) if right is not None else 0.0
            return left, right, c_l, c_r
        jump_px = self.max_jump_frac * width
        out_l, c_l = self._step(self._state["left"], left, conf_left, y_ref, jump_px)
        out_r, c_r = self._step(self._state["right"], right, conf_right, y_ref, jump_px)
        return out_l, out_r, c_l, c_r

    def _step(
        self, st: _SideState, poly: Polyline | None, conf: float, y_ref: float, jump_px: float
    ) -> tuple[Polyline | None, float]:
        if poly is not None and st.coeffs is not None:
            dx = abs(float(poly.x_at(y_ref)) - float(np.polyval(st.coeffs, y_ref)))
            if dx > jump_px:
                poly = None  # outlier: handled like a missed detection
        if poly is not None:
            new = np.asarray(poly.coeffs, dtype=np.float64)
            if st.coeffs is None:
                st.coeffs = new.copy()
            else:
                st.coeffs = self.alpha * new + (1.0 - self.alpha) * st.coeffs
            st.conf = float(conf)
            st.missed = 0
        elif st.coeffs is not None:
            st.missed += 1
            if st.missed > self.max_missed:
                st.coeffs, st.conf, st.missed = None, 0.0, 0
            else:
                st.conf *= self.decay
        if st.coeffs is None:
            return None, 0.0
        return Polyline(st.coeffs.copy()), st.conf
