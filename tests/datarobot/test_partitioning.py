"""partitioning.py のテスト."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from gene_plug_voltage_predictor.datarobot import partitioning as part_mod
from gene_plug_voltage_predictor.datarobot.partitioning import FOLD_COL_NAME


def _base_partitioning(**overrides: Any) -> dict[str, Any]:
    base = {
        "cv_type": "stratified_cv",
        "n_folds": 5,
        "seed": 42,
        "datetime_col": None,
        "validation_duration": None,
        "group_col": None,
    }
    base.update(overrides)
    return base


def test_build_partition_stratified_cv(mocker: MockerFixture) -> None:
    strat = mocker.patch.object(part_mod.dr, "StratifiedCV")
    part_mod.build_partition(_base_partitioning(cv_type="stratified_cv"))
    strat.assert_called_once_with(holdout_pct=0, reps=5, seed=42)


def test_build_partition_random_cv(mocker: MockerFixture) -> None:
    random = mocker.patch.object(part_mod.dr, "RandomCV")
    part_mod.build_partition(_base_partitioning(cv_type="random_cv"))
    random.assert_called_once_with(holdout_pct=0, reps=5, seed=42)


def test_build_partition_datetime_cv(mocker: MockerFixture) -> None:
    dt_spec = mocker.patch.object(part_mod.dr, "DatetimePartitioningSpecification")
    part_mod.build_partition(
        _base_partitioning(
            cv_type="datetime_cv",
            datetime_col="graphpt_ptdatetime",
            validation_duration="P30D",
        )
    )
    dt_spec.assert_called_once()
    kwargs = dt_spec.call_args.kwargs
    assert kwargs["datetime_partition_column"] == "graphpt_ptdatetime"
    assert kwargs["validation_duration"] == "P30D"
    assert kwargs["number_of_backtests"] == 5


def test_build_partition_datetime_cv_missing_col_raises() -> None:
    with pytest.raises(ValueError, match="datetime_col"):
        part_mod.build_partition(
            _base_partitioning(cv_type="datetime_cv", datetime_col=None)
        )


def test_build_partition_datetime_cv_missing_duration_raises() -> None:
    with pytest.raises(ValueError, match="validation_duration"):
        part_mod.build_partition(
            _base_partitioning(
                cv_type="datetime_cv",
                datetime_col="graphpt_ptdatetime",
                validation_duration=None,
            )
        )


def test_build_partition_group_cv(mocker: MockerFixture) -> None:
    user_cv = mocker.patch.object(part_mod.dr, "UserCV")
    part_mod.build_partition(
        _base_partitioning(cv_type="group_cv", group_col="group_key"),
        fold_col=FOLD_COL_NAME,
    )
    user_cv.assert_called_once()
    kwargs = user_cv.call_args.kwargs
    assert kwargs["user_partition_col"] == FOLD_COL_NAME


def test_build_partition_group_cv_without_fold_col_raises() -> None:
    with pytest.raises(ValueError, match="fold_col"):
        part_mod.build_partition(
            _base_partitioning(cv_type="group_cv", group_col="group_key"),
            fold_col=None,
        )


def test_build_partition_invalid_cv_type_raises() -> None:
    with pytest.raises(ValueError, match="cv_type"):
        part_mod.build_partition(_base_partitioning(cv_type="kfold"))


def test_prepare_train_dataset_non_group_returns_original(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    config = {
        "data": {
            "train_path": str(fixtures_dir / "train_sample.csv"),
            "test_path": None,
            "id_col": "target_id",
            "target_col": "plug_voltage",
        },
        "partitioning": _base_partitioning(cv_type="stratified_cv"),
        "output": {"outputs_dir": str(tmp_path / "outputs"), "metrics_dir": "metrics"},
    }
    path, fold_col = part_mod.prepare_train_dataset(config)
    assert path == Path(fixtures_dir / "train_sample.csv")
    assert fold_col is None


def test_prepare_train_dataset_group_cv_assigns_folds(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    config = {
        "data": {
            "train_path": str(fixtures_dir / "train_sample.csv"),
            "test_path": None,
            "id_col": "target_id",
            "target_col": "plug_voltage",
        },
        "partitioning": _base_partitioning(
            cv_type="group_cv", group_col="group_key", n_folds=5, seed=42
        ),
        "output": {"outputs_dir": str(tmp_path / "outputs"), "metrics_dir": "metrics"},
    }
    path, fold_col = part_mod.prepare_train_dataset(config)

    assert fold_col == FOLD_COL_NAME
    assert path == Path(tmp_path / "outputs" / "_tmp_train.csv")
    assert path.exists()

    df = pd.read_csv(path)
    assert FOLD_COL_NAME in df.columns
    # 同一 group は同一 fold
    for _, group_df in df.groupby("group_key"):
        assert group_df[FOLD_COL_NAME].nunique() == 1


def test_prepare_train_dataset_group_cv_missing_column_raises(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    config = {
        "data": {
            "train_path": str(fixtures_dir / "train_sample.csv"),
            "test_path": None,
            "id_col": "target_id",
            "target_col": "plug_voltage",
        },
        "partitioning": _base_partitioning(
            cv_type="group_cv", group_col="missing_col", n_folds=5, seed=42
        ),
        "output": {"outputs_dir": str(tmp_path / "outputs"), "metrics_dir": "metrics"},
    }
    with pytest.raises(ValueError, match="missing_col"):
        part_mod.prepare_train_dataset(config)
