"""
Wait for DataRobot Autopilot to complete on project 69fc3a9a1b6b9876c736cfca
then fetch results and generate experiment log for exp-009.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import date
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import datarobot as dr

from gene_plug_voltage_predictor.datarobot.client import setup_client
from gene_plug_voltage_predictor.datarobot.config import load_config
from gene_plug_voltage_predictor.datarobot.experiment_reporter import (
    ExperimentContext,
    render_experiment_log,
)
from gene_plug_voltage_predictor.datarobot.models import (
    _project_base_url,
    get_best_model,
    save_model_metrics,
)
from gene_plug_voltage_predictor.utils.hashing import sha256_of_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ID = "69fc3a9a1b6b9876c736cfca"
CONFIG_PATH = Path("config/datarobot_ep370g_ts_binary_32kv_v3.json")
TRAIN_CSV = Path("outputs/dataset_ep370g_ts_binary_32kv_v3.csv")
EXPERIMENT_LOG = Path(
    "E:/projects/contact-center-toolbox/60_domains/ress/gene_plug_voltage_predictor/experiments/exp-009.md"
)
POLL_INTERVAL = 60  # seconds
MAX_WAIT_MINUTES = 360


def wait_for_autopilot(project) -> None:
    deadline = time.monotonic() + MAX_WAIT_MINUTES * 60
    while time.monotonic() < deadline:
        status = project.get_status()
        done = status.get("autopilot_done", False)
        stage = status.get("stage", "unknown")
        logger.info("Autopilot status: %s (done=%s)", stage, done)
        if done:
            logger.info("Autopilot completed for project %s", project.id)
            return
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"Autopilot did not complete within {MAX_WAIT_MINUTES} minutes"
    )


def main() -> int:
    setup_client()

    cfg = dict(load_config(CONFIG_PATH))
    task_type = cfg["task"]["task_type"]
    name_prefix = cfg["project"]["name_prefix"]
    metrics_dir = Path(cfg["output"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching project %s", PROJECT_ID)
    project = dr.Project.get(PROJECT_ID)

    wait_for_autopilot(project)

    best, metric_name = get_best_model(project, task_type)
    save_model_metrics(
        model=best,
        project=project,
        output_path=metrics_dir / f"{name_prefix}_model.json",
        selected_by_metric=metric_name,
    )

    entry = (best.metrics or {}).get(metric_name) or {}
    metric_value = entry.get("crossValidation") or entry.get("validation") or 0.0

    train_hash = sha256_of_file(TRAIN_CSV)
    project_url = f"{_project_base_url()}/projects/{project.id}/models"

    ctx = ExperimentContext(
        experiment_id="exp-009",
        run_date=date.today(),
        author="大森",
        model_type="EP370G",
        topic="DataRobot TS 二値分類 32kV 新データ v3 (exp-009)",
        train_csv=str(TRAIN_CSV).replace("\\", "/"),
        train_csv_hash=train_hash,
        related_cleaning="2026-05-07-ep370g-v3",
        related_adrs=[
            "decision-001", "decision-002", "decision-003",
            "decision-009", "decision-010", "decision-012", "decision-014"
        ],
        datarobot_project_id=str(project.id),
        datarobot_project_url=project_url,
        best_model_id=str(best.id),
        best_blueprint=str(getattr(best, "model_type", "")),
        metric_name=metric_name,
        metric_value=float(metric_value),
        test_pred_csv="",
        test_pred_hash="",
    )

    md = render_experiment_log(ctx)
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_LOG.write_text(md, encoding="utf-8")

    # Also write metrics/experiment_009.md with backtest details
    metrics_md_path = metrics_dir / "experiment_009.md"
    metrics_md = f"""# exp-009 Metrics

## Project
- DataRobot Project ID: `{project.id}`
- Project URL: {project_url}
- Best Model ID: `{best.id}`
- Blueprint: {getattr(best, 'model_type', 'N/A')}
- Selected by: {metric_name}
- {metric_name} (CV/validation): {metric_value}

## All model metrics (top models)
"""
    models = list(project.get_models())
    models_with_score = []
    for m in models:
        entry = (m.metrics or {}).get(metric_name) or {}
        score = entry.get("crossValidation") or entry.get("validation")
        if score is not None:
            models_with_score.append((m, score))
    # sort by RMSE ascending for timeseries
    models_with_score.sort(key=lambda x: x[1])
    metrics_md += f"| Rank | Model ID | Blueprint | {metric_name} |\n"
    metrics_md += "|------|----------|-----------|------|\n"
    for i, (m, s) in enumerate(models_with_score[:10], 1):
        bp = getattr(m, 'model_type', 'N/A')
        metrics_md += f"| {i} | `{m.id}` | {bp} | {s} |\n"

    metrics_md_path.write_text(metrics_md, encoding="utf-8")

    print(f"[OK] best model: {best.id}")
    print(f"[OK] experiment log written to {EXPERIMENT_LOG}")
    print(f"[OK] metrics written to {metrics_md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
