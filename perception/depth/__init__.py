"""Depth and geometric distance (M4.1-M4.3). Importing this package never imports torch.

cam = load_camera("configs/camera.yaml", "kitti")       # flat-ground geometric distance
est = DepthEstimator(load_depth_config("configs/depth.yaml"))  # MiDaS relative inv. depth
# (shipped backend: onnx for the laptop runtime; pass backend="torch" on Kaggle)
align = ScaleAligner(cfg["alignment"]).fit(m, z_geo)    # relative -> metric
"""

from perception.depth.alignment import Alignment, ScaleAligner
from perception.depth.camera import CameraModel, load_camera
from perception.depth.estimator import DepthEstimator, load_depth_config, sample_box

__all__ = [
    "Alignment",
    "CameraModel",
    "DepthEstimator",
    "ScaleAligner",
    "load_camera",
    "load_depth_config",
    "sample_box",
]
