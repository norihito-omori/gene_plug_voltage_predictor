"""DataRobot プロジェクトの作成・Autopilot 起動・完了待機."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import datarobot as dr

logger = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")
_POLL_INTERVAL_SECONDS = 60
_AUTOPILOT_COMPLETED_STATUS = "completed"


def create_project(config: dict[str, Any], train_csv_path: Path) -> Any:
    """CSV をアップロードし、JST タイムスタンプ付きのプロジェクト名で作成する."""
    name_prefix = config["project"]["name_prefix"]
    timestamp = datetime.now(tz=JST).strftime("%Y%m%d_%H%M")
    project_name = f"{name_prefix}_{timestamp}"
    logger.info("Creating DataRobot project: %s (from %s)", project_name, train_csv_path)
    project = dr.Project.create(
        sourcedata=str(train_csv_path),
        project_name=project_name,
    )
    logger.info("Project created: %s (id=%s)", project_name, project.id)
    return project


def _build_feature_list_excluding_text(
    project: Any, extra_exclude: list[str]
) -> Any:
    """text 列と extra_exclude を除いた Feature List を作成する."""
    features = project.get_features()
    excluded_types = {"Text"}
    keep = [
        f.name
        for f in features
        if f.feature_type not in excluded_types
        and f.name not in extra_exclude
    ]
    logger.info(
        "Building feature list (kept=%d, excluded text types + %d cols)",
        len(keep),
        len(extra_exclude),
    )
    return project.create_featurelist(name="no_text_features", features=keep)


_MODE_MAP = {
    "quick": dr.enums.AUTOPILOT_MODE.QUICK,
    "full_auto": dr.enums.AUTOPILOT_MODE.FULL_AUTO,
    "comprehensive": dr.enums.AUTOPILOT_MODE.COMPREHENSIVE,
}


def configure_and_start(
    project: Any,
    partition: Any,
    config: dict[str, Any],
) -> Any:
    """Feature List 作成と set_target を実行して Autopilot を起動する."""
    autopilot = config["autopilot"]
    task_type = config["task"]["task_type"]
    metric = config["task"].get("metric")

    featurelist_id = None
    if not autopilot.get("text_mining", False):
        fl = _build_feature_list_excluding_text(
            project, autopilot.get("exclude_columns") or []
        )
        featurelist_id = fl.id

    kwargs: dict[str, Any] = {
        "target": config["data"]["target_col"],
        "worker_count": int(autopilot.get("worker_count", -1)),
        "mode": _MODE_MAP[autopilot.get("mode", "quick")],
    }
    if metric:
        kwargs["metric"] = metric
    if featurelist_id:
        kwargs["featurelist_id"] = featurelist_id

    kwargs["partitioning_method"] = partition

    positive_class = config["task"].get("positive_class")
    if task_type == "binary" and positive_class is not None:
        kwargs["positive_class"] = positive_class

    logger.info(
        "Starting Autopilot (mode=%s, target=%s, metric=%s)",
        autopilot.get("mode", "quick"),
        kwargs["target"],
        metric or "<default>",
    )
    project.set_target(**kwargs)
    return project


def wait_for_autopilot(project: Any, config: dict[str, Any]) -> Any:
    """Autopilot 完了まで 60 秒間隔でポーリング.

    Raises:
        TimeoutError: max_wait_minutes を超過した場合.
    """
    max_wait_minutes = int(config["autopilot"].get("max_wait_minutes", 180))
    deadline = time.monotonic() + max_wait_minutes * 60

    while time.monotonic() < deadline:
        status = project.get_status()
        stage = getattr(status, "stage", None) or status.get("stage")
        logger.info("Autopilot status: %s", stage)
        if stage == _AUTOPILOT_COMPLETED_STATUS:
            logger.info("Autopilot completed for project %s", project.id)
            return project
        time.sleep(_POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Autopilot did not complete within {max_wait_minutes} minutes "
        f"(project_id={project.id})"
    )
