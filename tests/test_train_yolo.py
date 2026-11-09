"""Pure helpers of scripts/train_yolo.py (no torch, no Ultralytics needed)."""

import math
from pathlib import Path

import yaml

from scripts.train_yolo import (
    bdd_to_coco_ids,
    label_classes,
    label_path,
    remap_label_text,
    repeat_factors,
    repeat_list,
    shared_mean,
)

FIX = Path(__file__).parent / "fixtures"


def test_label_classes_reads_ids_and_skips_blank_lines():
    assert label_classes("2 0.5 0.5 0.1 0.1\n\n7 0.1 0.1 0.2 0.2\n2 0 0 1 1\n") == {2, 7}
    assert label_classes("") == set()


def test_repeat_factors_lvis_formula():
    # 40 images: class 0 in all, class 1 in 4 (f=0.1, at threshold), class 2 in 1 (f=0.025).
    images = [{0} for _ in range(40)]
    for i in range(4):
        images[i] = {0, 1}
    images[39] = {0, 2}
    r = repeat_factors(images, threshold=0.1)
    assert r[10] == 1.0  # only the common class
    assert math.isclose(r[0], 1.0)  # f == t gives sqrt(1) = 1
    assert math.isclose(r[39], math.sqrt(0.1 / 0.025))  # 2.0


def test_repeat_factors_rare_class_and_empty_image():
    images = [set(), {0}, {0}, {0}, {1}]  # class 1 in 1/5 images -> below t=0.5
    r = repeat_factors(images, threshold=0.5)
    assert r[0] == 1.0
    assert r[1] == 1.0
    assert math.isclose(r[4], math.sqrt(0.5 / 0.2))


def test_repeat_list_is_seeded_and_keeps_integer_part():
    items = ["a", "b", "c"]
    out1 = repeat_list(items, [1.0, 2.0, 1.5], seed=0)
    out2 = repeat_list(items, [1.0, 2.0, 1.5], seed=0)
    assert out1 == out2
    assert out1.count("a") == 1 and out1.count("b") == 2 and out1.count("c") in (1, 2)
    many = repeat_list(["x"] * 2000, [1.25] * 2000, seed=1)
    assert 2400 < len(many) < 2600  # expectation 2500


def test_remap_label_text_maps_and_drops():
    text = "0 0.5 0.5 0.1 0.1\n8 0.2 0.2 0.1 0.1\n4 0.3 0.3 0.1 0.1\n"
    out = remap_label_text(text, {0: 2, 4: 0})
    assert out == "2 0.5 0.5 0.1 0.1\n0 0.3 0.3 0.1 0.1\n"
    assert remap_label_text("8 0.1 0.1 0.1 0.1\n", {0: 2}) == ""


def test_bdd_to_coco_mapping_from_config():
    cfg = yaml.safe_load(Path("configs/yolo_bdd.yaml").read_text())
    names = yaml.safe_load(Path("configs/data_bdd.yaml").read_text())["names"]
    mapping = bdd_to_coco_ids({int(k): v for k, v in names.items()}, cfg["zeroshot"]["bdd_to_coco"])
    assert mapping == {0: 2, 1: 7, 2: 5, 3: 0, 4: 0, 5: 1, 6: 3, 7: 9}  # traffic sign (8) dropped


def test_label_path_matches_fixture_layout():
    img = next((FIX / "images").glob("*.jpg"))
    lbl = label_path(Path("/root/bdd_yolo/images/val") / img.name)
    assert lbl == Path("/root/bdd_yolo/labels/val") / (img.stem + ".txt")


def test_repeat_factors_on_fixture_labels():
    classes = [label_classes(p.read_text()) for p in sorted((FIX / "labels").glob("*.txt"))]
    r = repeat_factors(classes, threshold=0.1)
    assert len(r) == len(classes) and all(x >= 1.0 for x in r)


def test_shared_mean_ignores_missing_and_extra_classes():
    pc = {"car": {"ap50": 0.8}, "person": {"ap50": 0.4}, "traffic sign": {"ap50": 0.0}}
    assert math.isclose(shared_mean(pc, "ap50"), 0.6)
    assert shared_mean({}, "ap50") == 0.0
