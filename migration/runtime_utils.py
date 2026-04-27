"""累積運転時間のパック形式 → 実時間変換ヘルパー。

`graphpt_pls001` / `dailygraphpt_pls001`（本プロジェクトでは `累積運転時間` 列）は
上位 3 バイトが「時間」、下位 1 バイトが「分」のパック形式で格納されている。

- 実時間（時）= raw // 256
- 実時間（分）= raw % 256 （0〜59 の範囲）
- 実時間（float 時間）= hours + minutes / 60

仕様出典: `60_domains/ress/gene_short_data_analysis.md` §4 累積運転時間（pls001）の変換。
"""

from __future__ import annotations

import pandas as pd


def unpack_to_hours(raw: pd.Series) -> pd.Series:
    """パック形式 raw 値 → 実時間（float、単位: 時）。"""
    hours = raw // 256
    minutes = raw % 256
    return hours + minutes / 60.0


def unpack_hour_component(raw: pd.Series) -> pd.Series:
    """パック形式 raw 値 → 時成分（整数、単位: 時）。"""
    return raw // 256


def unpack_minute_component(raw: pd.Series) -> pd.Series:
    """パック形式 raw 値 → 分成分（整数、0〜59）。"""
    return raw % 256
