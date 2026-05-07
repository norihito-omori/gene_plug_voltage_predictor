"""raw CSV 読み込み + 機種整合性チェック。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import CANONICAL_RENAMES, InputSchema


def load_raw_csv(
    path: Path,
    *,
    expected_mcnkind_id: int,
    expected_rated_kw: int,
    schema: InputSchema,
) -> pd.DataFrame:
    """CSV を読み込み、機種列・定格列との整合性を検証する。

    - encoding は utf-8-sig 固定（specs/input_schema.md §2）。
    - 必要列のみ読み込む（usecols）。生 CSV の不要列（排気温度等）は除外。
    - 読み込み直後に CANONICAL_RENAMES を適用し、EP400G 形式（`要求電圧1`..）を
      EP370G 形式（`要求電圧_1`..）へ正規化する（ADR-010）。
    - `mcnkind_id` 全行が `expected_mcnkind_id` と一致することを検証。
    - `target_output` 全行が `expected_rated_kw` と一致することを検証。

    Raises:
        ValueError: 必須カラム欠落 / mcnkind_id 不一致 / rated_output 不一致。
    """
    # usecols: rename 前後どちらの名前でも通るよう、正規化前後を合わせたセットを使用
    rename_reverse = {v: k for k, v in CANONICAL_RENAMES.items()}
    needed = set(schema.required_columns)
    needed_raw = needed | {rename_reverse.get(c, c) for c in needed}
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=lambda col: col in needed_raw,
    )
    df = df.rename(columns=CANONICAL_RENAMES)

    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in {path.name}: {missing}")

    actual_mcnkind = df[schema.mcnkind_col].dropna().unique().tolist()
    if not (len(actual_mcnkind) == 1 and int(actual_mcnkind[0]) == expected_mcnkind_id):
        raise ValueError(
            f"mcnkind_id mismatch in {path.name}: "
            f"expected={expected_mcnkind_id}, found={actual_mcnkind}"
        )

    actual_rated = df[schema.rated_output_col].dropna().unique().tolist()
    if not (len(actual_rated) == 1 and int(actual_rated[0]) == expected_rated_kw):
        raise ValueError(
            f"rated_output mismatch in {path.name}: "
            f"expected={expected_rated_kw}, found={actual_rated}"
        )
    return df
