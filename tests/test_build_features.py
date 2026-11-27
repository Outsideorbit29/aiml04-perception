import ast
import json
from pathlib import Path

import numpy as np
import pytest

from fraud.config import load_config
from fraud.features import feature_names
from scripts.build_features import auc
from scripts.build_features import main as build_main
from scripts.prep_data import main as prep_main

FIXTURES = Path(__file__).parent / "fixtures"


def test_feast_definitions_list_matches_builder():
    # Parse feature_repo/definitions.py without importing feast (not installed in CI).
    tree = ast.parse(Path("feature_repo/definitions.py").read_text(encoding="utf-8"))
    names = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FEATURES"
    )
    assert names == feature_names(load_config("configs/features.yaml")["windows"])


def test_auc_rank():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(np.array([0, 0]), np.array([1.0, 2.0])) is None


def test_build_features_end_to_end(tmp_path):
    pytest.importorskip("pyarrow")
    prep_main(["--data-dir", str(FIXTURES), "--out-dir", str(tmp_path), "--no-figures"])
    out = tmp_path / "f2"
    summary = build_main(["--input-dir", str(tmp_path), "--out-dir", str(out)])
    assert summary["parity_ok"]
    assert sum(summary["rows_per_split"].values()) == 300
    saved = json.loads((out / "features_summary.json").read_text())
    assert saved["rows"] == 300
    for name in ("train", "val", "test"):
        assert (out / "features" / f"{name}.parquet").exists()
    assert (out / "card_features.parquet").exists()
