"""Unit tests for the BDD100K -> YOLO helpers (CPU only, tiny fixtures)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.data.bdd_yolo import (
    CLASSES,
    allocate,
    area_bucket,
    class_id,
    clip_box,
    format_yolo_rows,
    frame_attributes,
    frame_to_yolo,
    stratified_sample,
    stratum_counts,
    xyxy_to_yolo,
)

FIXTURES = Path(__file__).parent / "fixtures"
W, H = 1280, 720


@pytest.fixture()
def frames() -> list[dict]:
    return json.loads((FIXTURES / "bdd_labels_sample.json").read_text())


def test_class_order_matches_architecture():
    assert CLASSES == (
        "car",
        "truck",
        "bus",
        "person",
        "rider",
        "bicycle",
        "motorcycle",
        "traffic light",
        "traffic sign",
    )


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("car", 0),
        ("truck", 1),
        ("bus", 2),
        ("person", 3),
        ("pedestrian", 3),
        ("rider", 4),
        ("bike", 5),
        ("bicycle", 5),
        ("motor", 6),
        ("motorcycle", 6),
        ("traffic light", 7),
        ("Traffic Sign ", 8),
        ("train", None),
        ("trailer", None),
        ("other vehicle", None),
        ("other person", None),
        ("lane", None),
        ("drivable area", None),
    ],
)
def test_class_id(category, expected):
    assert class_id(category) == expected


def test_xyxy_to_yolo_centre_and_size():
    cx, cy, w, h = xyxy_to_yolo(100, 200, 300, 400, W, H)
    assert cx == pytest.approx(200 / W)
    assert cy == pytest.approx(300 / H)
    assert w == pytest.approx(200 / W)
    assert h == pytest.approx(200 / H)


def test_xyxy_to_yolo_clips_and_swaps():
    assert clip_box(-5, 730, 50, 600, W, H) == (0.0, 600.0, 50.0, 720.0)
    cx, cy, w, h = xyxy_to_yolo(1300, -10, 1200, 10, W, H)
    assert 0 <= cx - w / 2 and cx + w / 2 <= 1
    assert 0 <= cy - h / 2 and cy + h / 2 <= 1
    assert w == pytest.approx(80 / W)


def test_xyxy_to_yolo_rejects_bad_size():
    with pytest.raises(ValueError):
        xyxy_to_yolo(0, 0, 1, 1, 0, 10)


def test_area_bucket_coco_thresholds():
    assert area_bucket(32 * 32 - 1) == "small"
    assert area_bucket(32 * 32) == "medium"
    assert area_bucket(96 * 96 - 1) == "medium"
    assert area_bucket(96 * 96) == "large"


def test_frame_to_yolo_2018_release(frames):
    rows = frame_to_yolo(frames[0], W, H)
    # car, person, clipped bike; the 0.5 px wide motor box, train and polygons are dropped
    assert [r[0] for r in rows] == [0, 3, 5]
    for _, cx, cy, w, h in rows:
        assert all(0.0 <= v <= 1.0 for v in (cx, cy, w, h))
    bike = rows[2]
    assert bike[3] == pytest.approx(50 / W)  # x clipped at 0
    assert bike[4] == pytest.approx(120 / H)  # y clipped at 720


def test_frame_to_yolo_det20_and_per_frame_layouts(frames):
    assert [r[0] for r in frame_to_yolo(frames[1], W, H)] == [3, 6, 7]
    assert [r[0] for r in frame_to_yolo(frames[2], W, H)] == [8]
    assert frame_to_yolo(frames[3], W, H) == []


def test_min_size_filter(frames):
    rows = frame_to_yolo(frames[0], W, H, min_size_px=25)
    assert [r[0] for r in rows] == [0, 5]  # the 20 px wide person is dropped


def test_format_yolo_rows():
    text = format_yolo_rows([(3, 0.5, 0.25, 0.1, 0.2)])
    assert text == "3 0.500000 0.250000 0.100000 0.200000\n"
    assert format_yolo_rows([]) == ""


def test_frame_attributes(frames):
    assert frame_attributes(frames[1]) == ("night", "rainy")
    assert frame_attributes(frames[3]) == ("undefined", "undefined")
    assert stratum_counts(frames)[("daytime", "clear")] == 1


def test_allocate_proportional_and_exact_total():
    counts = {"a": 700, "b": 200, "c": 95, "d": 5}
    alloc = allocate(counts, 100, min_per_stratum=1)
    assert sum(alloc.values()) == 100
    assert alloc["a"] == 70 and alloc["b"] == 20
    assert alloc["d"] >= 1
    assert all(alloc[k] <= counts[k] for k in counts)


def test_allocate_minimum_and_caps():
    counts = {"big": 1000, "tiny": 2, "one": 1}
    alloc = allocate(counts, 50, min_per_stratum=5)
    assert alloc["tiny"] == 2 and alloc["one"] == 1
    assert sum(alloc.values()) == 50


def test_allocate_edge_cases():
    assert allocate({"a": 3, "b": 4}, 10) == {"a": 3, "b": 4}
    assert allocate({"a": 3, "b": 4}, 0) == {"a": 0, "b": 0}
    # more strata than samples: minimums are dropped, total is still exact
    assert sum(allocate({"a": 3, "b": 4, "c": 5}, 2, min_per_stratum=1).values()) == 2


def _population() -> list[dict]:
    items = []
    spec = {
        ("daytime", "clear"): 600,
        ("night", "clear"): 300,
        ("night", "rainy"): 90,
        ("dawn/dusk", "foggy"): 10,
    }
    for (tod, w), n in spec.items():
        items += [
            {"name": f"{tod}-{w}-{i}", "attributes": {"timeofday": tod, "weather": w}}
            for i in range(n)
        ]
    return items


def test_stratified_sample_is_deterministic_and_stratified():
    items = _population()
    s1 = stratified_sample(items, frame_attributes, total=100, seed=42)
    s2 = stratified_sample(items, frame_attributes, total=100, seed=42)
    s3 = stratified_sample(items, frame_attributes, total=100, seed=7)
    assert [x["name"] for x in s1] == [x["name"] for x in s2]
    assert [x["name"] for x in s1] != [x["name"] for x in s3]
    assert len(s1) == 100 and len({x["name"] for x in s1}) == 100
    strata = Counter(frame_attributes(x) for x in s1)
    assert strata == {
        ("daytime", "clear"): 60,
        ("night", "clear"): 30,
        ("night", "rainy"): 9,
        ("dawn/dusk", "foggy"): 1,
    }


def test_stratified_sample_keeps_input_order():
    items = _population()
    sample = stratified_sample(items, frame_attributes, total=50, seed=42)
    idx = [items.index(x) for x in sample]
    assert idx == sorted(idx)
