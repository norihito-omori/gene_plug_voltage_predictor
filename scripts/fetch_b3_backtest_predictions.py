"""
B3-c の全 backtest 予測を取得する。

DatetimeModel.score_backtests() で各 backtest の OOF を計算し、
TrainingPredictions で取得を試みる。
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd
import datarobot as dr

from gene_plug_voltage_predictor.datarobot.client import setup_client
from gene_plug_voltage_predictor.datarobot.logging_setup import setup_logging

PROJECT_ID = "69fe7049e7c9c2dcdb4fa2ae"
MODEL_ID = "69fe71334cd6d557e0493930"
OUT = Path("outputs/b3_dr_backtest_predictions.csv")


def main():
    setup_logging()
    setup_client()
    project = dr.Project.get(PROJECT_ID)
    # DatetimeModel として取得
    model = dr.DatetimeModel.get(project.id, MODEL_ID)
    print(f"Model: {model.id} ({model.model_type})")

    # backtest スコアを計算 (まだなら)
    try:
        print("Triggering score_backtests()...")
        job = model.score_backtests()
        job.wait_for_completion(max_wait=1800)
        print("score_backtests done")
    except Exception as e:
        print(f"score_backtests info: {e}")

    # ALL_BACKTESTS で training predictions を再取得
    existing = list(dr.TrainingPredictions.list(project_id=project.id))
    target = next((t for t in existing if t.model_id == model.id), None)
    if target is None:
        print("Requesting ALL_BACKTESTS training predictions")
        job = model.request_training_predictions(
            data_subset=dr.enums.DATA_SUBSET.ALL_BACKTESTS
        )
        job.wait_for_completion(max_wait=1800)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            existing = list(dr.TrainingPredictions.list(project_id=project.id))
            target = next((t for t in existing if t.model_id == model.id), None)
            if target is not None:
                break
            time.sleep(5)
    if target is None:
        print("FAIL: predictions not found")
        sys.exit(1)
    pred_df = target.get_all_as_dataframe()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUT, index=False, encoding="utf-8-sig")
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"]).dt.tz_localize(None)
    print(f"Saved: {OUT} ({len(pred_df)} rows)")
    print(f"Time range: {pred_df['timestamp'].min().date()} .. {pred_df['timestamp'].max().date()}")
    print(f"partition_ids: {pred_df['partition_id'].value_counts().sort_index().to_dict()}")
    print(f"unique series: {pred_df['series_id'].nunique()}")


if __name__ == "__main__":
    main()
