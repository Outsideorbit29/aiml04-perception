"""Distance fusion and temporal filtering (M4.3, M4.5): tracks + depth + lanes -> PerceptionFrame.

fusion = DistanceFusion(load_depth_config("configs/depth.yaml"), load_camera(path, "carla"))
frame = fusion.update(tracks, lanes, stamp, frame_id, hw, inv_depth)
"""

from perception.fusion.fusion import (
    FUSION_MODES,
    DistanceFusion,
    EgoCorridor,
    Measurement,
    fuse,
    fuse_by_mode,
    fuse_inverse,
    in_ego_lane,
    inverse,
    lateral_offset,
)
from perception.fusion.kalman import DistanceKalman, FilterBank, FilterOutput

__all__ = [
    "FUSION_MODES",
    "DistanceFusion",
    "DistanceKalman",
    "EgoCorridor",
    "FilterBank",
    "FilterOutput",
    "Measurement",
    "fuse",
    "fuse_by_mode",
    "fuse_inverse",
    "in_ego_lane",
    "inverse",
    "lateral_offset",
]
