"""drift_check.py のテスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gene_plug_voltage_predictor.datarobot.drift_check import run_adversarial_validation


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def test_av_same_distribution_no_drift(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    test = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_csv(train_path, train)
    _write_csv(test_path, test)

    result = run_adversarial_validation(train_path, test_path, threshold=0.58)

    assert 0.4 <= result["overall_auc"] <= 0.65
    assert result["drifted_features"] == []
    assert result["threshold"] == 0.58


def test_av_shifted_column_is_drifted(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    train = pd.DataFrame(
        {"a": rng.normal(size=500), "b": rng.normal(size=500)}
    )
    test = pd.DataFrame(
        {"a": rng.normal(size=500), "b": rng.normal(loc=5.0, size=500)}
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_csv(train_path, train)
    _write_csv(test_path, test)

    result = run_adversarial_validation(train_path, test_path, threshold=0.58)

    assert "b" in result["drifted_features"]
    assert "a" not in result["drifted_features"]


def test_av_excludes_listed_cols(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    train = pd.DataFrame(
        {"id": range(500), "a": rng.normal(size=500)}
    )
    test = pd.DataFrame(
        {"id": range(500, 1000), "a": rng.normal(size=500)}
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_csv(train_path, train)
    _write_csv(test_path, test)

    result = run_adversarial_validation(
        train_path, test_path, threshold=0.58, exclude_cols=["id"]
    )

    assert "id" not in result["per_feature_auc"]
    assert "a" in result["per_feature_auc"]


def test_av_raises_when_no_shared_numeric_cols(tmp_path: Path) -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"b": [4.0, 5.0, 6.0]})
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False, encoding="utf-8-sig")
    test.to_csv(test_path, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="No shared numeric columns"):
        run_adversarial_validation(train_path, test_path)
