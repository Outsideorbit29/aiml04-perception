"""Classical lane detector (M3.1): HLS + Sobel threshold, then the shared warp/fit/smooth stages.

Pure NumPy/OpenCV on the CPU; no ROS, CARLA or torch.
"""

from __future__ import annotations

import numpy as np

from perception.lanes.base import LaneDetector
from perception.lanes.threshold import threshold_binary


class ClassicalLaneDetector(LaneDetector):
    """ROI -> HLS/Sobel threshold -> bird's-eye warp -> sliding window -> polyfit -> EMA."""

    name = "classical"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.threshold_cfg = cfg["classical"]["threshold"]

    def lane_binary(self, frame_bgr: np.ndarray) -> np.ndarray:
        return threshold_binary(frame_bgr, self.threshold_cfg)
