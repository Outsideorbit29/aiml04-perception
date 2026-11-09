"""Tests for locating the data-prep output and resolving the data YAML."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.data.check_data_yaml import check_split, find_dataset_root, main


def _make_dataset(base: Path) -> Path:
    root = base / "notebooks" / "someone" / "aiml04-data-prep" / "bdd_yolo"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "x.jpg").write_bytes(b"")
        (root / "labels" / split / "x.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (root / "data.yaml").write_text("train: images/train\nval: images/val\nnc: 9\n")
    return root


def test_find_and_resolve(tmp_path: Path):
    root = _make_dataset(tmp_path / "input")
    assert find_dataset_root(tmp_path / "input") == root
    assert check_split(root, "val") == {"images": 1, "labels": 1, "unmatched": 0}
    out = tmp_path / "resolved.yaml"
    main(
        [
            "--config",
            "configs/data_bdd.yaml",
            "--search",
            str(tmp_path / "input"),
            "--out",
            str(out),
        ]
    )
    data = yaml.safe_load(out.read_text())
    assert data["path"] == str(root) and "prep" not in data
    assert data["names"][0] == "car" and data["nc"] == 9


def test_missing_dataset(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_dataset_root(tmp_path)
