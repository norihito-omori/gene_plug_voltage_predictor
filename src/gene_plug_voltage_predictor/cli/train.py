"""DataRobot Autopilot 実行 CLI: Autopilot → experiment ログドラフト書き出し。"""

from __future__ import annotations

import argparse
import sys
from datetime import date as _date
from pathlib import Path

from gene_plug_voltage_predictor.datarobot import pipeline as dr_pipeline
from gene_plug_voltage_predictor.datarobot.config import load_config
from gene_plug_voltage_predictor.datarobot.experiment_reporter import (
    ExperimentContext,
    render_experiment_log,
)
from gene_plug_voltage_predictor.utils.hashing import sha256_of_file


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run DataRobot Autopilot and draft an experiment log"
    )
    ap.add_argument("--config", required=True, type=Path, help="datarobot_*.json")
    ap.add_argument("--train", required=True, type=Path, help="training CSV (from cli.clean)")
    ap.add_argument("--experiment-log", required=True, type=Path)
    ap.add_argument("--author", default="大森")
    ap.add_argument("--model-type", required=True, choices=["EP370G", "EP400G"])
    ap.add_argument("--topic", required=True, help="experiment topic slug")
    ap.add_argument("--related-cleaning", required=True)
    ap.add_argument(
        "--related-adrs",
        default="decision-001,decision-002,decision-003",
        help="comma-separated ADR list",
    )
    ap.add_argument(
        "--experiment-id",
        required=True,
        help="exp-YYYY-MM-DD-NN",
    )
    args = ap.parse_args()

    if not args.train.exists():
        print(f"ERROR: train CSV not found: {args.train}", file=sys.stderr)
        return 2

    cfg = dict(load_config(args.config))
    train_hash = sha256_of_file(args.train)

    result = dr_pipeline.run(cfg, train_path=args.train)

    # When test_path is not configured, pipeline.run() skips predict_dataset and
    # returns test_pred_csv=None. Emit empty strings so the experiment log still
    # renders cleanly (人間が後で埋める箇所になる).
    if result.test_pred_csv is None:
        test_pred_csv_str = ""
        test_pred_hash = ""
    else:
        test_pred_csv_str = str(result.test_pred_csv).replace("\\", "/")
        test_pred_hash = sha256_of_file(result.test_pred_csv)

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
        datarobot_project_id=result.project_id,
        datarobot_project_url=result.project_url,
        best_model_id=result.best_model_id,
        best_blueprint=result.best_blueprint,
        metric_name=result.metric_name,
        metric_value=result.metric_value,
        test_pred_csv=test_pred_csv_str,
        test_pred_hash=test_pred_hash,
    )
    md = render_experiment_log(ctx)
    args.experiment_log.parent.mkdir(parents=True, exist_ok=True)
    args.experiment_log.write_text(md, encoding="utf-8")

    print(f"[OK] experiment log draft written to {args.experiment_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
