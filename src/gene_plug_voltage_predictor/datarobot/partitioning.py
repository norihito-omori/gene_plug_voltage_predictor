"""Cross-validation 分割戦略（4 種）の DataRobot Partition 変換."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import datarobot as dr
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

FOLD_COL_NAME: str = "__fold__"


def prepare_train_dataset(config: dict[str, Any]) -> tuple[Path, str | None]:
    """DataRobot にアップロードする train CSV を準備する.

    group_cv の場合は group_col から __fold__ 列を割り当てた一時 CSV を
    outputs_dir/_tmp_train.csv に書き出し、そのパスを返す.
    それ以外は data.train_path をそのまま返す.

    Returns:
        (csv_path, fold_col_name). group_cv 以外では fold_col_name は None.

    Raises:
        FileNotFoundError: train_path が存在しない場合.
        ValueError: group_cv で group_col が train CSV に存在しない場合.
    """
    train_path = Path(config["data"]["train_path"])
    if not train_path.exists():
        raise FileNotFoundError(f"train_path not found: {train_path}")

    part = config["partitioning"]
    if part["cv_type"] != "group_cv":
        return train_path, None

    group_col = part["group_col"]
    n_folds_raw = part.get("n_folds")
    n_folds = 5 if n_folds_raw is None else int(n_folds_raw)

    df = pd.read_csv(train_path)
    if group_col not in df.columns:
        raise ValueError(
            f"group_col {group_col!r} not found in train CSV columns: {list(df.columns)}"
        )

    gkf = GroupKFold(n_splits=n_folds)
    df[FOLD_COL_NAME] = -1
    for fold_idx, (_, val_idx) in enumerate(
        gkf.split(df, groups=df[group_col])
    ):
        df.loc[val_idx, FOLD_COL_NAME] = fold_idx

    outputs_dir = Path(config["output"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = outputs_dir / "_tmp_train.csv"
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    logger.info(
        "Prepared group-CV train CSV with __fold__ column: %s (groups=%d, folds=%d)",
        tmp_path,
        df[group_col].nunique(),
        n_folds,
    )
    return tmp_path, FOLD_COL_NAME


def build_partition(
    partitioning_config: dict[str, Any],
    fold_col: str | None = None,
) -> Any:
    """partitioning_config を DataRobot の Partition オブジェクトに変換する.

    Returns:
        dr.StratifiedCV / dr.RandomCV /
        dr.DatetimePartitioningSpecification / dr.UserCV のいずれか.

    Raises:
        ValueError: cv_type 不正、または必須フィールド欠落.
    """
    cv_type = partitioning_config["cv_type"]
    n_folds_raw = partitioning_config.get("n_folds")
    n_folds = 5 if n_folds_raw is None else int(n_folds_raw)
    seed_raw = partitioning_config.get("seed")
    seed = 42 if seed_raw is None else int(seed_raw)

    if cv_type == "stratified_cv":
        return dr.StratifiedCV(holdout_pct=0, reps=n_folds, seed=seed)
    if cv_type == "random_cv":
        return dr.RandomCV(holdout_pct=0, reps=n_folds, seed=seed)
    if cv_type == "datetime_cv":
        datetime_col = partitioning_config.get("datetime_col")
        validation_duration = partitioning_config.get("validation_duration")
        use_series_id = partitioning_config.get("use_series_id")
        forecast_window_start = partitioning_config.get("forecast_window_start", 1)
        forecast_window_end = partitioning_config.get("forecast_window_end", 1)
        if datetime_col is None:
            raise ValueError("datetime_col is required for datetime_cv")
        if validation_duration is None:
            raise ValueError("validation_duration is required for datetime_cv")
        holdout_start_date_raw = partitioning_config.get("holdout_start_date")
        holdout_end_date_raw = partitioning_config.get("holdout_end_date")
        holdout_start_date = (
            datetime.strptime(holdout_start_date_raw, "%Y-%m-%d")
            if isinstance(holdout_start_date_raw, str)
            else holdout_start_date_raw
        )
        holdout_end_date = (
            datetime.strptime(holdout_end_date_raw, "%Y-%m-%d")
            if isinstance(holdout_end_date_raw, str)
            else holdout_end_date_raw
        )
        spec = dr.DatetimePartitioningSpecification(
            datetime_partition_column=datetime_col,
            number_of_backtests=n_folds,
            validation_duration=validation_duration,
            holdout_start_date=holdout_start_date,
            holdout_end_date=holdout_end_date,
            use_time_series=bool(use_series_id),
        )
        if use_series_id:
            spec.multiseries_id_columns = [use_series_id]
            spec.forecast_window_start = forecast_window_start
            spec.forecast_window_end = forecast_window_end
        return spec
    if cv_type == "group_cv":
        if fold_col is None:
            raise ValueError("fold_col is required for group_cv")
        return dr.UserCV(user_partition_col=fold_col, cv_holdout_level=None)

    raise ValueError(f"unknown cv_type: {cv_type!r}")
