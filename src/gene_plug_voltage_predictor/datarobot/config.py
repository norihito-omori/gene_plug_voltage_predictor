"""パイプライン設定（config.json）の TypedDict 定義と検証."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypedDict

TaskType = Literal["binary", "multiclass", "regression", "timeseries"]
CvType = Literal["stratified_cv", "random_cv", "datetime_cv", "group_cv"]
AutopilotMode = Literal["quick", "full_auto", "comprehensive"]

VALID_TASK_TYPES: tuple[str, ...] = ("binary", "multiclass", "regression", "timeseries")
VALID_CV_TYPES: tuple[str, ...] = ("stratified_cv", "random_cv", "datetime_cv", "group_cv")
VALID_AUTOPILOT_MODES: tuple[str, ...] = ("quick", "full_auto", "comprehensive")


class ProjectConfig(TypedDict):
    name_prefix: str
    endpoint: str | None


class DataConfig(TypedDict):
    train_path: str
    test_path: str | None
    id_col: str
    target_col: str


class TaskConfig(TypedDict):
    task_type: TaskType
    metric: str | None
    positive_class: str | None


class PartitioningConfig(TypedDict):
    cv_type: CvType
    n_folds: int
    seed: int
    datetime_col: str | None
    validation_duration: str | None
    group_col: str | None


class AutopilotConfig(TypedDict):
    mode: AutopilotMode
    worker_count: int
    max_wait_minutes: int
    text_mining: bool
    exclude_columns: list[str]


class OutputConfig(TypedDict):
    outputs_dir: str
    metrics_dir: str


class PipelineConfig(TypedDict):
    project: ProjectConfig
    data: DataConfig
    task: TaskConfig
    partitioning: PartitioningConfig
    autopilot: AutopilotConfig
    output: OutputConfig


def load_config(path: Path) -> PipelineConfig:
    """config.json を読み込み、validate_config を呼んで返す.

    Raises:
        FileNotFoundError: パスが存在しない場合.
        ValueError: 検証エラー.
    """
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    validate_config(data)
    return data  # type: ignore[return-value]


def _require(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section or section[key] in (None, ""):
        raise ValueError(f"{section_name}.{key} is required")
    return section[key]


def validate_config(config: dict[str, Any]) -> None:
    """必須キー・enum・条件付き必須の検証を行う.

    Raises:
        ValueError: 不正値や欠落があった場合.
    """
    for top_key in ("project", "data", "task", "partitioning", "autopilot", "output"):
        if top_key not in config:
            raise ValueError(f"top-level key '{top_key}' is required")
        if not isinstance(config[top_key], dict):
            raise ValueError(
                f"top-level key '{top_key}' must be an object, "
                f"got {type(config[top_key]).__name__}"
            )

    _require(config["project"], "name_prefix", "project")
    train_path_str = _require(config["data"], "train_path", "data")
    _require(config["data"], "id_col", "data")
    _require(config["data"], "target_col", "data")

    if not Path(train_path_str).exists():
        raise ValueError(f"data.train_path does not exist: {train_path_str}")

    test_path = config["data"].get("test_path")
    if test_path and not Path(test_path).exists():
        raise ValueError(f"data.test_path does not exist: {test_path}")

    task_type = _require(config["task"], "task_type", "task")
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"task.task_type must be one of {VALID_TASK_TYPES}, got {task_type!r}"
        )

    metric = config["task"].get("metric")
    if task_type == "binary" and metric == "F1":
        raise ValueError(
            "task.metric=F1 is not allowed for binary: "
            "F1 depends on Top-K threshold and cannot be used in get_best_model. "
            "Use LogLoss or Area Under PR Curve."
        )

    cv_type = _require(config["partitioning"], "cv_type", "partitioning")
    if cv_type not in VALID_CV_TYPES:
        raise ValueError(
            f"partitioning.cv_type must be one of {VALID_CV_TYPES}, got {cv_type!r}"
        )

    if cv_type == "datetime_cv":
        if not config["partitioning"].get("datetime_col"):
            raise ValueError("partitioning.datetime_col is required for datetime_cv")
        if not config["partitioning"].get("validation_duration"):
            raise ValueError(
                "partitioning.validation_duration is required for datetime_cv"
            )

    if cv_type == "group_cv" and not config["partitioning"].get("group_col"):
        raise ValueError("partitioning.group_col is required for group_cv")

    mode = config["autopilot"].get("mode", "quick")
    if mode not in VALID_AUTOPILOT_MODES:
        raise ValueError(
            f"autopilot.mode must be one of {VALID_AUTOPILOT_MODES}, got {mode!r}"
        )
