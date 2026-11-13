"""Distance fusion and temporal filtering (M4.3, M4.5): tracks + depth + lanes -> PerceptionFrame.

fusion = DistanceFusion(load_depth_config("configs/depth.yaml"), load_camera(path, "carla"))
frame = fusion.update(tracks, lanes, stamp, frame_id, hw, inv_depth)
"""

from perception.fusion.fusion import DistanceFusion, Measurement, fuse, fuse_inverse
from perception.fusion.kalman import DistanceKalman, FilterBank, FilterOutput

__all__ = [
    "DistanceFusion",
    "DistanceKalman",
    "FilterBank",
    "FilterOutput",
    "Measurement",
    "fuse",
    "fuse_inverse",
]
