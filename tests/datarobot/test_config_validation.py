"""config バリデータのテスト."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gene_plug_voltage_predictor.datarobot import config as config_mod


@pytest.fixture
def valid_config(fixtures_dir: Path) -> dict[str, Any]:
    return {
        "project": {"name_prefix": "test_job", "endpoint": None},
        "data": {
            "train_path": str(fixtures_dir / "train_sample.csv"),
            "test_path": str(fixtures_dir / "test_sample.csv"),
            "id_col": "target_id",
            "target_col": "plug_voltage",
        },
        "task": {
            "task_type": "regression",
            "metric": None,
            "positive_class": None,
        },
        "partitioning": {
            "cv_type": "stratified_cv",
            "n_folds": 5,
            "seed": 42,
            "datetime_col": None,
            "validation_duration": None,
            "group_col": None,
        },
        "autopilot": {
            "mode": "quick",
            "worker_count": -1,
            "max_wait_minutes": 180,
            "text_mining": False,
            "exclude_columns": [],
        },
        "output": {"outputs_dir": "outputs", "metrics_dir": "metrics"},
    }


def test_validate_config_accepts_valid(valid_config: dict[str, Any]) -> None:
    config_mod.validate_config(valid_config)  # 例外なし


def test_validate_config_rejects_missing_target_col(valid_config: dict[str, Any]) -> None:
    valid_config["data"].pop("target_col")
    with pytest.raises(ValueError, match="target_col"):
        config_mod.validate_config(valid_config)


def test_validate_config_rejects_missing_train_path(valid_config: dict[str, Any]) -> None:
    valid_config["data"].pop("train_path")
    with pytest.raises(ValueError, match="train_path"):
        config_mod.validate_config(valid_config)


def test_validate_config_rejects_invalid_task_type(valid_config: dict[str, Any]) -> None:
    valid_config["task"]["task_type"] = "clustering"
    with pytest.raises(ValueError, match="task_type"):
        config_mod.validate_config(valid_config)


def test_validate_config_rejects_invalid_cv_type(valid_config: dict[str, Any]) -> None:
    valid_config["partitioning"]["cv_type"] = "kfold"
    with pytest.raises(ValueError, match="cv_type"):
        config_mod.validate_config(valid_config)


def test_validate_config_rejects_invalid_autopilot_mode(valid_config: dict[str, Any]) -> None:
    valid_config["autopilot"]["mode"] = "turbo"
    with pytest.raises(ValueError, match="autopilot.mode"):
        config_mod.validate_config(valid_config)


def test_validate_config_rejects_binary_f1(valid_config: dict[str, Any]) -> None:
    valid_config["task"]["task_type"] = "binary"
    valid_config["task"]["metric"] = "F1"
    with pytest.raises(ValueError, match="F1"):
        config_mod.validate_config(valid_config)


def test_validate_config_requires_datetime_col_for_datetime_cv(
    valid_config: dict[str, Any],
) -> None:
    valid_config["partitioning"]["cv_type"] = "datetime_cv"
    valid_config["partitioning"]["datetime_col"] = None
    valid_config["partitioning"]["validation_duration"] = "P30D"
    with pytest.raises(ValueError, match="datetime_col"):
        config_mod.validate_config(valid_config)


def test_validate_config_requires_validation_duration_for_datetime_cv(
    valid_config: dict[str, Any],
) -> None:
    valid_config["partitioning"]["cv_type"] = "datetime_cv"
    valid_config["partitioning"]["datetime_col"] = "graphpt_ptdatetime"
    valid_config["partitioning"]["validation_duration"] = None
    with pytest.raises(ValueError, match="validation_duration"):
        config_mod.validate_config(valid_config)


def test_validate_config_requires_group_col_for_group_cv(
    valid_config: dict[str, Any],
) -> None:
    valid_config["partitioning"]["cv_type"] = "group_cv"
    valid_config["partitioning"]["group_col"] = None
    with pytest.raises(ValueError, match="group_col"):
        config_mod.validate_config(valid_config)


def test_validate_config_rejects_missing_train_file(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    valid_config["data"]["train_path"] = str(tmp_path / "nonexistent.csv")
    with pytest.raises(ValueError, match="train_path"):
        config_mod.validate_config(valid_config)


def test_validate_config_rejects_missing_test_file(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    valid_config["data"]["test_path"] = str(tmp_path / "nonexistent.csv")
    with pytest.raises(ValueError, match="test_path"):
        config_mod.validate_config(valid_config)


def test_validate_config_rejects_missing_top_level_section(
    valid_config: dict[str, Any],
) -> None:
    valid_config.pop("partitioning")
    with pytest.raises(ValueError, match="partitioning"):
        config_mod.validate_config(valid_config)


def test_validate_config_rejects_non_dict_section(
    valid_config: dict[str, Any],
) -> None:
    valid_config["project"] = None
    with pytest.raises(ValueError, match="must be an object, got NoneType"):
        config_mod.validate_config(valid_config)


def test_load_config_parses_json(
    valid_config: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_config), encoding="utf-8")
    loaded = config_mod.load_config(path)
    assert loaded["data"]["target_col"] == "plug_voltage"


def test_load_config_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        config_mod.load_config(tmp_path / "nope.json")
