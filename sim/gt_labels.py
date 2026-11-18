"""Ground-truth 2D boxes and distances from CARLA actors (M5.6), CARLA-free math.

sim/record_dataset.py gives these functions numpy arrays taken from CARLA (bounding-box world
vertices, the camera's world-to-sensor matrix, the depth GT image), so they are unit-tested in
CI without the simulator.

CARLA sensor axes: x forward, y right, z up (left-handed, Unreal). The image projection is
``u = fx * y / x + cx``, ``v = -fy * z / x + cy``; ``x`` is the camera-frame depth, the same
quantity as the depth camera and as ``ObjectState.distance_m`` at zero pitch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# CARLA blueprint `base_type` (0.9.15) / actor kind -> BDD class (configs/data_bdd.yaml).
BASE_TYPE_TO_BDD = {
    "car": "car",
    "van": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "pedestrian": "person",
}
MIN_DEPTH_M = 0.3  # vertices closer than this (or behind the camera) make the box invalid


def to_camera(points_world: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray:
    """(N, 3) world points -> (N, 3) sensor-frame points with a 4x4 world-to-sensor matrix."""
    p = np.asarray(points_world, np.float64)
    h = np.hstack([p, np.ones((len(p), 1))])
    return (np.asarray(world_to_cam, np.float64) @ h.T).T[:, :3]


def project_box(
    cam_pts: np.ndarray, fx: float, fy: float, cx: float, cy: float, width: int, height: int
) -> tuple[np.ndarray, float, float] | None:
    """Tight image box of the projected points, clipped to the image.

    Returns ``(xyxy, depth_min, depth_max)`` or None if any point is behind / too close to the
    camera or the clipped box is empty."""
    x = cam_pts[:, 0]
    if np.any(x < MIN_DEPTH_M):
        return None
    u = fx * cam_pts[:, 1] / x + cx
    v = -fy * cam_pts[:, 2] / x + cy
    x1, x2 = float(np.clip(u.min(), 0, width - 1)), float(np.clip(u.max(), 0, width - 1))
    y1, y2 = float(np.clip(v.min(), 0, height - 1)), float(np.clip(v.max(), 0, height - 1))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return np.array([x1, y1, x2, y2], np.float32), float(x.min()), float(x.max())


def visible_fraction(
    depth: np.ndarray, box: np.ndarray, dmin: float, dmax: float, tol: float, grid: int = 7
) -> float:
    """Fraction of a ``grid x grid`` sample inside ``box`` whose GT depth lies on the object
    (within ``[dmin - tol, dmax + tol]``). Occluded samples are nearer, background samples
    farther; both count as not visible."""
    x1, y1, x2, y2 = (float(v) for v in box)
    us = np.linspace(x1, x2, grid + 2)[1:-1].round().astype(int)
    vs = np.linspace(y1, y2, grid + 2)[1:-1].round().astype(int)
    h, w = depth.shape
    d = depth[np.clip(vs, 0, h - 1)][:, np.clip(us, 0, w - 1)]
    return float(np.mean((d >= dmin - tol) & (d <= dmax + tol)))


def yolo_line(cls_id: int, box: np.ndarray, width: int, height: int) -> str:
    x1, y1, x2, y2 = (float(v) for v in box)
    return (
        f"{cls_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def label_objects(
    objects: list[dict[str, Any]],
    world_to_cam: np.ndarray,
    depth: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """GT labels for one frame.

    ``objects``: ``{"id", "cls_name", "vertices" (8, 3) world}``. ``cfg``: the ``dataset`` block
    of configs/carla.yaml (max_box_distance_m, min_box_px, min_visible_frac, depth_tolerance_m).
    Each label: id, cls_name, bbox_xyxy, distance_m (nearest vertex depth = rear face for a car
    ahead), depth_max_m, visible_frac.
    """
    fx, fy, cx, cy = intrinsics
    h, w = depth.shape
    out = []
    for o in objects:
        cam = to_camera(o["vertices"], world_to_cam)
        proj = project_box(cam, fx, fy, cx, cy, w, h)
        if proj is None:
            continue
        box, dmin, dmax = proj
        if dmin > float(cfg["max_box_distance_m"]):
            continue
        if min(box[2] - box[0], box[3] - box[1]) < float(cfg["min_box_px"]):
            continue
        vis = visible_fraction(depth, box, dmin, dmax, float(cfg["depth_tolerance_m"]))
        if vis < float(cfg["min_visible_frac"]):
            continue
        out.append({
            "id": int(o["id"]),
            "cls_name": str(o["cls_name"]),
            "bbox_xyxy": [round(float(v), 2) for v in box],
            "distance_m": round(dmin, 3),
            "depth_max_m": round(dmax, 3),
            "visible_frac": round(vis, 3),
        })
    return out
