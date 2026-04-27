"""Adversarial Validation（train/test 分布差検出）."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

_LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "num_threads": 1,
    "verbose": -1,
}
_NUM_BOOST_ROUND = 100
_N_SPLITS = 3
_RANDOM_STATE = 42


def _cv_auc(X: pd.DataFrame, y: np.ndarray) -> float:
    skf = StratifiedKFold(n_splits=_N_SPLITS, shuffle=True, random_state=_RANDOM_STATE)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        model = lgb.train(_LGB_PARAMS, dtrain, num_boost_round=_NUM_BOOST_ROUND)
        oof[va_idx] = model.predict(X.iloc[va_idx])
    return float(roc_auc_score(y, oof))


def run_adversarial_validation(
    train_path: Path,
    test_path: Path,
    threshold: float = 0.58,
    exclude_cols: list[str] | None = None,
) -> dict[str, Any]:
    """train/test の分布差を LightGBM で検出する.

    各特徴量を 1 つずつ使って「train=0, test=1」を分類し、AUC を計算.
    AUC >= threshold の特徴量をドリフトありと判定.
    数値列のみを対象（カテゴリ列は除外）.

    Returns:
        {
            "overall_auc": float,
            "per_feature_auc": dict[str, float],
            "drifted_features": list[str],
            "threshold": float,
        }
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    exclude_set = set(exclude_cols or [])
    common = [c for c in train.columns if c in test.columns and c not in exclude_set]
    numeric = [
        c for c in common
        if pd.api.types.is_numeric_dtype(train[c])
        and pd.api.types.is_numeric_dtype(test[c])
    ]
    if not numeric:
        raise ValueError("No shared numeric columns between train and test")

    X = pd.concat(
        [train[numeric], test[numeric]], axis=0, ignore_index=True
    )
    y = np.concatenate(
        [np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
    )

    overall_auc = _cv_auc(X, y)
    logger.info("Adversarial Validation overall AUC: %.4f", overall_auc)

    per_feature: dict[str, float] = {}
    drifted: list[str] = []
    for col in numeric:
        auc = _cv_auc(X[[col]], y)
        per_feature[col] = auc
        if auc >= threshold:
            drifted.append(col)
            logger.warning("Drift detected in %s (AUC=%.4f)", col, auc)

    return {
        "overall_auc": overall_auc,
        "per_feature_auc": per_feature,
        "drifted_features": drifted,
        "threshold": threshold,
    }
