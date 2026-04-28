"""既存 DataRobot プロジェクトから結果を取得して実験ログを生成する CLI。

Autopilot が完了済みのプロジェクト ID を渡すと、ベストモデル選択・
メトリクス保存・実験ログ Markdown ドラフト生成を行う。
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as _date
from pathlib import Path

import datarobot as dr

from gene_plug_voltage_predictor.datarobot.client import setup_client
from gene_plug_voltage_predictor.datarobot.config import load_config
from gene_plug_voltage_predictor.datarobot.experiment_reporter import (
    ExperimentContext,
    render_experiment_log,
)
from gene_plug_voltage_predictor.datarobot.logging_setup import setup_logging
from gene_plug_voltage_predictor.datarobot.models import (
    _project_base_url,
    get_best_model,
    save_model_metrics,
)
from gene_plug_voltage_predictor.utils.hashing import sha256_of_file

_logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch results from a completed DataRobot project and draft an experiment log"
    )
    ap.add_argument("--project-id", required=True, help="DataRobot プロジェクト ID")
    ap.add_argument("--config", required=True, type=Path, help="datarobot_*.json")
    ap.add_argument("--train", required=True, type=Path, help="学習に使った CSV（ハッシュ計算用）")
    ap.add_argument("--experiment-log", required=True, type=Path)
    ap.add_argument("--author", default="大森")
    ap.add_argument("--model-type", required=True, choices=["EP370G", "EP400G"])
    ap.add_argument("--topic", required=True)
    ap.add_argument("--related-cleaning", required=True)
    ap.add_argument(
        "--related-adrs",
        default="decision-001,decision-002,decision-003",
        help="カンマ区切り ADR リスト",
    )
    ap.add_argument("--experiment-id", required=True)
    args = ap.parse_args()

    if not args.train.exists():
        print(f"ERROR: train CSV not found: {args.train}", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    setup_logging()
    setup_client()

    cfg = dict(load_config(args.config))

    _logger.info("Fetching project %s", args.project_id)
    project = dr.Project.get(args.project_id)

    best, metric_name = get_best_model(project, cfg["task"]["task_type"])

    metrics_dir = Path(cfg["output"]["metrics_dir"])
    name_prefix = cfg["project"]["name_prefix"]
    save_model_metrics(
        model=best,
        project=project,
        output_path=metrics_dir / f"{name_prefix}_model.json",
        selected_by_metric=metric_name,
    )

    entry = (best.metrics or {}).get(metric_name) or {}
    metric_value = entry.get("crossValidation") or entry.get("validation") or 0.0

    train_hash = sha256_of_file(args.train)
    project_url = f"{_project_base_url()}/projects/{project.id}/models"

    ctx = ExperimentContext(
        experiment_id=args.experiment_id,
        run_date=_date.today(),
        author=args.author,
        model_type=args.model_type,
        topic=args.topic,
        train_csv=str(args.train).replace("\\", "/"),
        train_csv_hash=train_hash,
        related_cleaning=args.related_cleaning,
        related_adrs=[a.strip() for a in args.related_adrs.split(",") if a.strip()],
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
    args.experiment_log.parent.mkdir(parents=True, exist_ok=True)
    args.experiment_log.write_text(md, encoding="utf-8")

    print(f"[OK] experiment log draft written to {args.experiment_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
