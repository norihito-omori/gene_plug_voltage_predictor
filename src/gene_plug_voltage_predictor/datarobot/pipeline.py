"""DataRobot Autopilot orchestrator. Returns RunResult for downstream callers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from gene_plug_voltage_predictor.datarobot.client import setup_client
from gene_plug_voltage_predictor.datarobot.logging_setup import setup_logging
from gene_plug_voltage_predictor.datarobot.models import (
    _project_base_url,
    get_best_model,
    save_model_metrics,
)
from gene_plug_voltage_predictor.datarobot.partitioning import (
    build_partition,
    prepare_train_dataset,
)
from gene_plug_voltage_predictor.datarobot.predictions import predict_dataset
from gene_plug_voltage_predictor.datarobot.project import (
    configure_and_start,
    create_project,
    wait_for_autopilot,
)

JST = ZoneInfo("Asia/Tokyo")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    """Metadata returned by :func:`run` for experiment-log generation."""

    project_id: str
    project_url: str
    best_model_id: str
    best_blueprint: str
    metric_name: str
    metric_value: float
    test_pred_csv: Path | None


def _extract_metric_value(model: Any, metric_name: str) -> float:
    """Pull the CV (preferred) or validation score for ``metric_name`` off a model."""
    entry = (model.metrics or {}).get(metric_name) or {}
    score = entry.get("crossValidation")
    if score is None:
        score = entry.get("validation")
    if score is None:
        raise RuntimeError(
            f"No CV or validation score found for metric {metric_name!r}"
        )
    return float(score)


def run(config: dict[str, Any], *, train_path: Path) -> RunResult:
    """Run the full Autopilot flow and return structured metadata.

    This mirrors the template ``scripts/run_datarobot.py:run`` but returns a
    :class:`RunResult` for the CLI to feed into ``experiment_reporter``.
    """
    outputs_dir = Path(config["output"]["outputs_dir"])
    metrics_dir = Path(config["output"]["metrics_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Emit a per-run log file to outputs/run_YYYYMMDD_HHMMSS.log (JST),
    # mirroring the template `scripts/run_datarobot.py` convention so
    # specs/datarobot_conventions.md and the implementation stay aligned.
    ts = datetime.now(tz=JST).strftime("%Y%m%d_%H%M%S")
    setup_logging(log_file=outputs_dir / f"run_{ts}.log")

    setup_client()

    # Honor the caller-supplied train_path by overriding config.data.train_path
    # so prepare_train_dataset picks up the CSV actually staged on disk. Phase 0
    # configs often point train_path and test_path at the same CSV; in that case
    # the override must propagate to test_path too, otherwise test prediction
    # would silently run on the stale path from the JSON.
    original_data = config["data"]
    new_data = {**original_data, "train_path": str(train_path)}
    if original_data.get("test_path") == original_data.get("train_path"):
        new_data["test_path"] = str(train_path)
    config = {**config, "data": new_data}

    train_csv, fold_col = prepare_train_dataset(config)
    project = create_project(config, train_csv)
    partition = build_partition(config["partitioning"], fold_col=fold_col)
    configure_and_start(project, partition, config)
    wait_for_autopilot(project, config)

    best, metric_name = get_best_model(project, config["task"]["task_type"])

    name_prefix = config["project"]["name_prefix"]
    save_model_metrics(
        model=best,
        project=project,
        output_path=metrics_dir / f"{name_prefix}_model.json",
        selected_by_metric=metric_name,
    )

    # Only set test_pred_csv when predict_dataset actually runs; otherwise the
    # CLI would try to hash a file that was never written.
    test_pred_csv: Path | None = None
    test_path_str = config["data"].get("test_path")
    if test_path_str:
        test_pred_csv = outputs_dir / f"{name_prefix}_test_pred.csv"
        predict_dataset(
            project=project,
            model=best,
            dataset_path=Path(test_path_str),
            output_path=test_pred_csv,
            id_col=config["data"]["id_col"],
        )

    return RunResult(
        project_id=str(project.id),
        project_url=f"{_project_base_url()}/projects/{project.id}/models",
        best_model_id=str(best.id),
        best_blueprint=str(getattr(best, "model_type", "")),
        metric_name=metric_name,
        metric_value=_extract_metric_value(best, metric_name),
        test_pred_csv=test_pred_csv,
    )
