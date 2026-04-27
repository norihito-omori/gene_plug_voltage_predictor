"""raw CSV 読み込み＋機種整合性チェック。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import InputSchema


def load_raw_csv(
    path: Path,
    *,
    expected_model_type: str,
    schema: InputSchema,
) -> pd.DataFrame:
    """CSV を読み込み、機種列との整合性を検証する。

    Raises:
        ValueError: 必須カラム欠落 or 機種列の値がディレクトリ名推定と不一致。
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in {path.name}: {missing}")

    actual = df[schema.model_type_col].unique()
    if not (len(actual) == 1 and actual[0] == expected_model_type):
        raise ValueError(
            f"model_type mismatch in {path.name}: "
            f"expected={expected_model_type}, found={list(actual)}"
        )
    return df
