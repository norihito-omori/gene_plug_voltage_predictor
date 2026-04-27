"""Leaderboard からのモデル選択とメトリクス保存."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# (metric_name, higher_is_better)
METRIC_PRIORITY: dict[str, list[tuple[str, bool]]] = {
    "binary": [
        ("Area Under PR Curve", True),
        ("AUC", True),
        ("Max MCC", True),
        ("LogLoss", False),
    ],
    "multiclass": [("LogLoss", False), ("Accuracy", True)],
    "regression": [("RMSE", False), ("MAE", False), ("R Squared", True)],
    "timeseries": [("RMSE", False), ("MAE", False)],
}


def _extract_score(model: Any, metric: str) -> float | None:
    entry = (model.metrics or {}).get(metric)
    if entry is None:
        return None
    score = entry.get("crossValidation")
    if score is None:
        score = entry.get("validation")
    return score


def get_best_model(project: Any, task_type: str) -> tuple[Any, str]:
    """task_type 別の優先順位で最良モデルを選択する.

    CV スコア優先、無ければ validation スコア.

    Returns:
        (選ばれたモデル, 選択に使ったメトリクス名) のタプル.

    Raises:
        RuntimeError: モデルが 1 つも取得できない、または全メトリクスが null.
    """
    priorities = METRIC_PRIORITY.get(task_type)
    if priorities is None:
        raise RuntimeError(f"No metric priority defined for task_type={task_type!r}")

    models = list(project.get_models())
    if not models:
        raise RuntimeError("No models found on the project")

    for metric, higher_is_better in priorities:
        scored = [(m, _extract_score(m, metric)) for m in models]
        scored = [(m, s) for m, s in scored if s is not None]
        if not scored:
            continue
        # Primary: score (direction depends on higher_is_better).
        # Secondary (tie-break): model.id ascending, for reproducibility.
        def _sort_key(pair: tuple[Any, float]) -> tuple[float, str]:
            score = pair[1] if higher_is_better else -pair[1]
            return (-score, getattr(pair[0], "id", "") or "")

        scored.sort(key=_sort_key)
        best = scored[0]
        logger.info(
            "Selected model %s (%s) by %s=%s",
            best[0].id,
            getattr(best[0], "model_type", "?"),
            metric,
            best[1],
        )
        return best[0], metric

    raise RuntimeError(
        f"All priority metrics are null for task_type={task_type!r}. "
        f"Tried: {[m for m, _ in priorities]}"
    )


def _project_base_url() -> str:
    """Derive the DataRobot UI base URL from DATAROBOT_ENDPOINT env var.

    The API endpoint (e.g., https://app.datarobot.com/api/v2) shares a host
    with the UI. Strip the /api/v2 suffix to get the UI base URL. Falls back
    to the public SaaS URL when the env var is unset.
    """
    import os

    endpoint = os.environ.get("DATAROBOT_ENDPOINT", "")
    if endpoint:
        return endpoint.rstrip("/").removesuffix("/api/v2").rstrip("/")
    return "https://app.datarobot.com"


def save_model_metrics(
    model: Any,
    project: Any,
    output_path: Path,
    selected_by_metric: str,
) -> None:
    """モデル情報を JSON で保存する."""
    metrics = model.metrics or {}
    cv_scores = {k: v.get("crossValidation") for k, v in metrics.items()}
    val_scores = {k: v.get("validation") for k, v in metrics.items()}

    payload = {
        "project_id": project.id,
        "project_url": f"{_project_base_url()}/projects/{project.id}/models",
        "model_id": model.id,
        "model_type": getattr(model, "model_type", None),
        "selected_by_metric": selected_by_metric,
        "cv_scores": cv_scores,
        "validation_scores": val_scores,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved model metrics to %s", output_path)
