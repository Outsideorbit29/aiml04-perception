"""YOLOP lane backend (M3.2) - placeholder.

Plan: run pretrained YOLOP (lane-line + drivable-area heads) on Kaggle, resize the lane-line
mask to the frame size, threshold it with `yolop.lane_threshold`, and return it from
`lane_binary`. The inherited `_lanes_from_binary` then fits polylines with the same
`LaneFitter`/`LaneSmoother` as the classical backend, so both return the same `LaneResult`.
`yolop.every_n_frames: 2` keeps the live path inside the latency budget.
"""

from __future__ import annotations

import numpy as np

from perception.lanes.base import LaneDetector


class YolopLaneDetector(LaneDetector):
    name = "yolop"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        raise NotImplementedError("YOLOP lane backend is milestone M3.2 and not implemented yet")

    def lane_binary(self, frame_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError("YOLOP lane backend is milestone M3.2 and not implemented yet")
