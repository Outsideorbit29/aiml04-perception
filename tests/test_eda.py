import json
from pathlib import Path

from fraud.data import add_time_columns
from fraud.eda import eda_summary, fraud_rate_by
from fraud.synthetic import make_synthetic
from scripts.prep_data import main as prep_main

FIXTURES = Path(__file__).parent / "fixtures"


def test_eda_summary_basic():
    tx, _ = make_synthetic(500, fraud_rate=0.1, seed=3)
    df = add_time_columns(tx, "2017-12-01")
    s = eda_summary(df)
    assert s["rows"] == 500 and s["frauds"] == int(tx["isFraud"].sum())
    assert 0 < s["fraud_rate"] < 1
    assert set(s["by_category"]) >= {"ProductCD", "card4", "card6", "P_emaildomain"}
    assert len(s["fraud_rate_by_hour"]) <= 24
    json.dumps(s)  # must be JSON-serialisable


def test_fraud_rate_by_handles_missing():
    tx, _ = make_synthetic(300, seed=4)
    rows = fraud_rate_by(tx, "P_emaildomain")
    assert "NA" in {r["value"] for r in rows}
    assert sum(r["count"] for r in rows) == 300


def test_prep_data_end_to_end(tmp_path):
    result = prep_main(
        [
            "--config",
            "configs/data.yaml",
            "--data-dir",
            str(FIXTURES),
            "--out-dir",
            str(tmp_path),
            "--no-figures",
        ]
    )
    for name in ("train", "val", "test"):
        assert (tmp_path / "splits" / f"{name}.parquet").exists()
    assert json.loads((tmp_path / "split_summary.json").read_text()) == result["split_summary"]
    assert result["eda_summary"]["rows"] == result["split_summary"]["train"]["rows"]
