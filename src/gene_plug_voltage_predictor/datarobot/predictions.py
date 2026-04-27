"""予測の取得と CSV 出力."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import datarobot as dr
import pandas as pd

logger = logging.getLogger(__name__)

_TRAINING_PRED_POLL_SECONDS = 5
_TRAINING_PRED_TIMEOUT_SECONDS = 300
_DATASET_UPLOAD_TIMEOUT_SECONDS = 3600


def predict_dataset(
    project: Any,
    model: Any,
    dataset_path: Path,
    output_path: Path,
    id_col: str,
) -> None:
    """データセットをアップロードして予測し、CSV に保存する（utf-8-sig）.

    予測結果に入力 CSV の id_col を結合してから保存する.
    """
    logger.info("Uploading prediction dataset: %s", dataset_path)
    dataset = project.upload_dataset(str(dataset_path))
    job = model.request_predictions(dataset.id)
    pred_df = job.get_result_when_complete(
        max_wait=_DATASET_UPLOAD_TIMEOUT_SECONDS
    )

    input_df = pd.read_csv(dataset_path, usecols=[id_col])
    if len(pred_df) != len(input_df):
        logger.warning(
            "Prediction length %d != input length %d; skipping id merge",
            len(pred_df),
            len(input_df),
        )
        merged = pred_df
    else:
        merged = pd.concat(
            [input_df.reset_index(drop=True), pred_df.reset_index(drop=True)],
            axis=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved predictions to %s (rows=%d)", output_path, len(merged))


def download_training_predictions(
    project: Any,
    model: Any,
    output_path: Path,
) -> None:
    """OOF 予測を取得して CSV に保存する.

    既存の TrainingPredictions があれば使い、無ければ作成してポーリング.
    最大 300 秒待機. 取得できなければ WARNING で継続（例外は raise しない）.
    """
    try:
        existing = list(
            dr.TrainingPredictions.list(project_id=project.id)
        )
        target = next((t for t in existing if t.model_id == model.id), None)

        if target is None:
            logger.info("Requesting training predictions for model %s", model.id)
            job = model.request_training_predictions(
                data_subset=dr.enums.DATA_SUBSET.ALL
            )
            job.wait_for_completion(max_wait=_TRAINING_PRED_TIMEOUT_SECONDS)

            deadline = time.monotonic() + _TRAINING_PRED_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                existing = list(
                    dr.TrainingPredictions.list(project_id=project.id)
                )
                target = next(
                    (t for t in existing if t.model_id == model.id), None
                )
                if target is not None:
                    break
                time.sleep(_TRAINING_PRED_POLL_SECONDS)

        if target is None:
            logger.warning(
                "Training predictions for model %s not found after polling",
                model.id,
            )
            return

        pred_df = target.get_all_as_dataframe()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Saved training predictions to %s", output_path)
    except dr.errors.ClientError as exc:
        logger.warning(
            "Failed to fetch training predictions: %s (continuing)", exc
        )
