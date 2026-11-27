import numpy as np
import pandas as pd
import pytest

from fraud.metrics import (
    average_precision,
    confusion_at,
    evaluate,
    recall_at_fpr,
    roc_auc,
    threshold_at_fpr,
)
from fraud.model import FeatureSpec


def test_average_precision_known_values():
    y = np.array([1, 0, 1, 0])
    # ranks: 1 (P=1), 0, 1 (P=2/3), 0 -> AP = 0.5*1 + 0.5*2/3
    assert average_precision(y, np.array([0.9, 0.8, 0.7, 0.1])) == pytest.approx(5 / 6)
    assert average_precision(np.array([1, 1, 0]), np.array([0.9, 0.8, 0.1])) == 1.0


def test_average_precision_ties_like_sklearn():
    # all scores tied: one block, precision = base rate, recall 0 -> 1
    y = np.array([1, 0, 0, 0])
    assert average_precision(y, np.ones(4)) == pytest.approx(0.25)


def test_roc_auc():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)


def test_threshold_respects_fpr_budget():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    s = rng.random(5000) + y * 0.5
    for budget in (0.0, 0.01, 0.05):
        thr = threshold_at_fpr(y, s, budget)
        assert confusion_at(y, s, thr)["fpr"] <= budget + 1e-12
    assert recall_at_fpr(y, s, 0.05) > recall_at_fpr(y, s, 0.01)


def test_evaluate_keys_and_nan_check():
    y = np.array([0, 1, 0, 1])
    out = evaluate(y, np.array([0.1, 0.9, 0.2, 0.8]), 0.5, "val_")
    assert out["val_pr_auc"] == 1.0 and out["val_recall"] == 1.0 and out["val_alerts"] == 2
    with pytest.raises(ValueError):
        roc_auc(y, np.array([0.1, np.nan, 0.2, 0.3]))


def test_feature_spec_roundtrip(tmp_path):
    train = pd.DataFrame(
        {"amt": [1.0, 2.0, np.nan] * 10, "prod": ["W", "C", None] * 10, "rare": ["x"] + ["y"] * 29}
    )
    spec = FeatureSpec.fit(train, ["amt"], ["prod", "rare"], min_count=5)
    assert spec.vocab["prod"] == ["C", "W"] and spec.vocab["rare"] == ["y"]
    x = spec.transform(pd.DataFrame({"amt": [5.0], "prod": ["W"], "rare": ["x"]}))
    assert x[0, 0] == 5.0 and x[0, 1] == 1.0 and np.isnan(x[0, 2])  # rare/unknown -> NaN
    spec.save(tmp_path / "spec.json")
    again = FeatureSpec.load(tmp_path / "spec.json")
    row = {"amt": 5.0, "prod": "W", "rare": "x"}
    assert np.array_equal(again.transform_row(row), x, equal_nan=True)
    assert again.columns == ["amt", "prod", "rare"]
