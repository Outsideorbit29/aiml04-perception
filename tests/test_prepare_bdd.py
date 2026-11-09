"""End-to-end test of the BDD converter on 3 synthetic images (CPU, runs in CI)."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest
import yaml

from scripts.data import prepare_bdd

FIXTURES = Path(__file__).parent / "fixtures"


def _write_jpeg(path: Path, width: int, height: int) -> None:
    import cv2
    import numpy as np

    cv2.imwrite(str(path), np.full((height, width, 3), 127, np.uint8))


@pytest.fixture()
def bdd_root(tmp_path: Path) -> Path:
    frames = json.loads((FIXTURES / "bdd_labels_sample.json").read_text())
    root = tmp_path / "in"
    for split in ("train", "val"):
        (root / "img" / split).mkdir(parents=True)
        for fr in frames:
            _write_jpeg(root / "img" / split / fr["name"], 1280, 720)
        # one labelled frame has no image: it must be skipped
        labels = frames + [{"name": "missing.jpg", "labels": []}]
        (root / f"{split}.json").write_text(json.dumps(labels))
    attrs = [{"name": "d.jpg", "attributes": {"timeofday": "night", "weather": "foggy"}}]
    (root / "attrs.json").write_text(json.dumps(attrs))
    cfg = yaml.safe_load(Path("configs/data_bdd.yaml").read_text())
    cfg["prep"].update(
        input_root=str(root),
        out_dir=str(tmp_path / "out"),
        train_size=3,
        max_side=640,
        source={
            "images_train": "img/train",
            "images_val": "img/val",
            "labels_train": "train.json",
            "labels_val": "val.json",
            "attributes_train": "attrs.json",
        },
    )
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path


def test_prepare_end_to_end_and_idempotent(bdd_root: Path):
    import cv2

    out = bdd_root / "out"
    argv = ["--config", str(bdd_root / "cfg.yaml"), "--workers", "2"]
    prepare_bdd.main(argv)
    assert sorted(p.name for p in (out / "images" / "val").iterdir()) == [
        "a.jpg",
        "b.jpg",
        "c.jpg",
        "d.jpg",
    ]
    assert len(list((out / "images" / "train").iterdir())) == 3
    img = cv2.imread(str(out / "images" / "val" / "a.jpg"))
    assert img.shape[:2] == (360, 640)  # resized to max_side
    lines = (out / "labels" / "val" / "a.txt").read_text().splitlines()
    assert [ln.split()[0] for ln in lines] == ["0", "3", "5"]
    assert (out / "labels" / "val" / "d.txt").read_text() == ""

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert "path" not in data and data["nc"] == 9 and data["names"][6] == "motorcycle"
    summary = json.loads((out / "meta" / "prep_summary.json").read_text())
    assert summary["train"]["labelled_frames_in_source"] == 5
    assert summary["train"]["attributes_filled_from_fallback"] == 1
    with gzip.open(out / "meta" / "frames_train_full.jsonl.gz", "rt") as fh:
        full = [json.loads(x) for x in fh]
    assert len(full) == 4 and {f["name"]: f["weather"] for f in full}["d.jpg"] == "foggy"

    # rerun with a smaller subset: stale train files are removed, val is unchanged
    prepare_bdd.main([*argv, "--train-size", "2"])
    assert len(list((out / "images" / "train").iterdir())) == 2
    assert len(list((out / "labels" / "train").iterdir())) == 2
    shutil.rmtree(out)
