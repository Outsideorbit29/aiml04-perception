"""Pure helpers of scripts/train_mix.py (M5.9 split + mix list; no torch, no Ultralytics)."""

import math
from pathlib import Path

import yaml

from scripts.train_mix import (
    block_split,
    delta_rows,
    flat,
    frame_index,
    keep_fraction,
    merge_label_text,
    pos_repeat,
    pseudo_rows,
    split_ranges,
    weather_segments,
)

ROOT = Path(__file__).resolve().parents[1]
LOG = [{"weather": w, "first_index": 500 * i} for i, w in enumerate("ABCDEF")]


def test_frame_index_and_segments_from_meta():
    assert frame_index("Town01_00042") == 42
    segs = weather_segments(list(reversed(LOG)), 3000)  # order in the file does not matter
    assert segs[0] == ("A", 0, 500) and segs[-1] == ("F", 2500, 3000)
    assert weather_segments([], 10) == [("unknown", 0, 10)]


def test_block_split_holds_out_last_block_with_guards():
    roles = block_split(weather_segments(LOG, 3000), 3000, block_len=100, val_block=-1, guard=20)
    assert split_ranges(roles, "val") == [(400 + 500 * i, 500 + 500 * i) for i in range(6)]
    # Guard before each val block and after it, across the segment boundary.
    g = split_ranges(roles, "guard")
    assert g[0] == (380, 400) and g[1] == (500, 520) and g[-1] == (2880, 2900)
    assert roles.count("val") == 600 and roles.count("guard") == 6 * 20 + 5 * 20
    assert roles.count("train") == 3000 - 600 - 220


def test_block_split_no_train_frame_within_guard_of_val():
    roles = block_split(weather_segments(LOG, 3000), 3000, 100, 2, 15)
    val = [i for i, r in enumerate(roles) if r == "val"]
    train = [i for i, r in enumerate(roles) if r == "train"]
    assert min(abs(t - v) for t in train for v in val[::10] + val[9::10]) > 15
    assert split_ranges(roles, "val")[0] == (200, 300)


def test_block_split_rejects_missing_block():
    try:
        block_split([("A", 0, 50)], 50, 100, 1, 0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_pos_repeat_hits_target_share():
    n_base, n_pos, n_neg = 20853, 1700, 300
    r = pos_repeat(n_base, n_pos, n_neg, 0.25)
    carla = n_pos * r + n_neg
    assert math.isclose(carla / (carla + n_base), 0.25)
    assert pos_repeat(100, 1000, 0, 0.25) == 1.0  # never below 1
    assert pos_repeat(100, 0, 5, 0.25) == 0.0


def test_keep_fraction_seeded_and_sorted():
    items = [f"f{i:03d}" for i in range(100)]
    a = keep_fraction(items, 0.5, 42)
    assert len(a) == 50 and a == sorted(a) and a == keep_fraction(items, 0.5, 42)
    assert keep_fraction(items, 0.0, 1) == [] and len(keep_fraction(items, 2.0, 1)) == 100


def test_pseudo_rows_filter_class_and_conf():
    rows = pseudo_rows([7, 0, 8, 4], [[0.5, 0.5, 0.1, 0.2]] * 4, [0.9, 0.95, 0.3, 0.41],
                       keep=[4, 7, 8], conf_thr=0.4)
    assert rows == ["7 0.500000 0.500000 0.100000 0.200000",
                    "4 0.500000 0.500000 0.100000 0.200000"]


def test_merge_label_text():
    assert merge_label_text("0 0.1 0.1 0.1 0.1\n\n", ["7 0.2 0.2 0.1 0.1"]) == (
        "0 0.1 0.1 0.1 0.1\n7 0.2 0.2 0.1 0.1\n")
    assert merge_label_text("", []) == ""


def test_delta_rows_and_flat():
    rows = delta_rows({"car": 0.5, "bus": 0.4}, {"bus": 0.6, "truck": 0.3})
    assert rows[0][0] == "car" and math.isnan(rows[0][2])
    assert rows[1] == ("bus", 0.4, 0.6) and rows[2][0] == "truck"
    assert flat({"a": {"b": 1, "c": True, "d": "x"}, "e": 2.5}) == {"a/b": 1.0, "e": 2.5}


def test_mix_config_is_consistent():
    cfg = yaml.safe_load((ROOT / "configs/yolo_mix.yaml").read_text())
    names = yaml.safe_load((ROOT / cfg["data_config"]).read_text())["names"]
    assert set(cfg["pseudo"]["classes"]) <= set(names.values())
    assert cfg["wandb"]["best_artifact"] != "yolo11s-bdd-best"
    assert cfg["train"]["imgsz"] == 640 and 0 < cfg["mix"]["carla_share"] < 0.5
    assert "model" not in cfg["train"]  # the base comes from the W&B artifact
