import os

import pytest

from fraud.config import load_config
from fraud.tracking import configure, find_token

ENV = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_EXPERIMENT_NAME",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV:
        monkeypatch.delenv(key, raising=False)


def test_token_from_attached_dataset(tmp_path):
    secrets = tmp_path / "datasets" / "user" / "secrets"
    secrets.mkdir(parents=True)
    (secrets / "dagshub_token").write_text("abc123\n")
    assert find_token("dagshub_token", tmp_path) == "abc123"


def test_configure_remote(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "dagshub_token").write_text("tok")
    cfg = load_config("configs/mlflow.yaml")
    assert configure(cfg, tmp_path) == cfg["tracking_uri"]
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == cfg["username"]
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "tok"
    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "aiml01-fraud"


def test_configure_fallback(tmp_path):
    cfg = load_config("configs/mlflow.yaml")
    assert configure(cfg, tmp_path) == cfg["fallback_uri"]
    assert "MLFLOW_TRACKING_PASSWORD" not in os.environ
