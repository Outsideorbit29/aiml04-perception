"""Lane detection (M3): classical OpenCV pipeline and learned backends behind one interface.

Usage:
    cfg = load_lane_config("configs/lanes.yaml")
    detector = build_lane_detector(cfg)        # backend chosen by cfg["method"]
    result = detector.detect(frame_bgr)        # -> perception.types.LaneResult
    result = detector.detect(img, stateless=True)  # independent images (dataset eval)
"""

from perception.lanes.base import LaneDetector, build_lane_detector, load_lane_config
from perception.lanes.classical import ClassicalLaneDetector

__all__ = ["ClassicalLaneDetector", "LaneDetector", "build_lane_detector", "load_lane_config"]
