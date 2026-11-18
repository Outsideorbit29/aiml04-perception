"""sim.gt_labels (M5.6): projection of CARLA bounding boxes, occlusion test, YOLO lines."""

from __future__ import annotations

import numpy as np
import pytest

from sim.carla_utils import camera_intrinsics, load_carla_config
from sim.gt_labels import (
    BASE_TYPE_TO_BDD,
    label_objects,
    project_box,
    to_camera,
    visible_fraction,
    yolo_line,
)

CFG = load_carla_config("configs/carla.yaml")
W, H = 800, 450
INTR = camera_intrinsics(W, H, 90.0)  # fx = fy = 400, centre (400, 225)


def box_vertices(cx: float, cy: float, cz: float, hx: float, hy: float, hz: float) -> np.ndarray:
    return np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def test_to_camera_identity_and_translation():
    m = np.eye(4)
    m[0, 3] = -10.0  # world -> camera: camera sits at x = 10
    p = to_camera(np.array([[15.0, 1.0, 2.0]]), m)
    assert p.tolist() == [[5.0, 1.0, 2.0]]


def test_project_box_centre_and_height():
    # 1.6 m tall object (z -1.6..0 relative to the camera, i.e. standing on the road) at 20 m
    cam = box_vertices(20.0, 0.0, -0.8, 0.2, 0.3, 0.8)
    box, dmin, dmax = project_box(cam, *INTR, W, H)
    assert dmin == pytest.approx(19.8) and dmax == pytest.approx(20.2)
    # bottom edge = road contact at the near face: v = 225 + 400 * 1.6 / 19.8
    assert box[3] == pytest.approx(225 + 400 * 1.6 / 19.8, abs=0.01)
    assert (box[0] + box[2]) / 2 == pytest.approx(400.0)


def test_project_box_rejects_behind_and_clips():
    assert project_box(box_vertices(0.1, 0, 0, 0.5, 0.5, 0.5), *INTR, W, H) is None
    box, _, _ = project_box(box_vertices(3.0, 4.0, 0.0, 1.0, 1.0, 0.5), *INTR, W, H)
    assert box[2] == W - 1  # clipped at the right image edge


def test_visible_fraction():
    depth = np.full((H, W), 100.0, np.float32)
    box = np.array([300, 200, 500, 300], np.float32)
    assert visible_fraction(depth, box, 20.0, 22.0, 1.0) == 0.0  # only background
    depth[200:301, 300:501] = 21.0
    assert visible_fraction(depth, box, 20.0, 22.0, 1.0) == 1.0
    depth[200:301, 300:501] = 5.0  # something nearer covers it
    assert visible_fraction(depth, box, 20.0, 22.0, 1.0) == 0.0


def test_label_objects_filters_and_distance():
    d = CFG["dataset"]
    depth = np.full((H, W), 200.0, np.float32)
    # car ahead at 15 m (rear face at 13 m), visible in the depth map
    car = box_vertices(15.0, 0.0, -0.8, 2.0, 0.9, 0.75)
    box, dmin, _ = project_box(car, *INTR, W, H)
    x1, y1, x2, y2 = box.astype(int)
    depth[y1:y2 + 1, x1:x2 + 1] = 14.0
    hidden = box_vertices(40.0, 3.0, -0.8, 2.0, 0.9, 0.75)  # not in the depth map: occluded
    far = box_vertices(150.0, 0.0, -0.8, 2.0, 0.9, 0.75)  # beyond max_box_distance_m
    objs = [{"id": 1, "cls_name": "car", "vertices": car},
            {"id": 2, "cls_name": "car", "vertices": hidden},
            {"id": 3, "cls_name": "car", "vertices": far}]
    labels = label_objects(objs, np.eye(4), depth, INTR, d)
    assert [lb["id"] for lb in labels] == [1]
    assert labels[0]["distance_m"] == pytest.approx(13.0)
    assert labels[0]["visible_frac"] == 1.0


def test_yolo_line_and_class_map():
    line = yolo_line(0, np.array([200, 100, 600, 350]), W, H)
    assert line == "0 0.500000 0.500000 0.500000 0.555556"
    assert BASE_TYPE_TO_BDD["van"] == "car" and BASE_TYPE_TO_BDD["pedestrian"] == "person"
