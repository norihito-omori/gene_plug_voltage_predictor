"""予測ターゲット生成。ADR-013: future_7d_max = t+1〜t+7 の日次最大電圧の最大。"""
from __future__ import annotations

import pandas as pd


def aggregate_daily_max_voltage(
    df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    datetime_col: str = "dailygraphpt_ptdatetime",
    voltage_col: str = "要求電圧",
    power_col: str = "発電機電力",
) -> pd.DataFrame:
    """30 分粒度 DataFrame → plug × 日付の日次最大電圧テーブルを返す。

    非運転日（その日に発電機電力 > 0 の行が 0 件）は行を生成しない。
    返却カラム: [plug_id_col, "date", "daily_max"]
    """
    running = df.loc[df[power_col] > 0].copy()
    running["date"] = pd.to_datetime(running[datetime_col]).dt.normalize()
    daily = (
        running.groupby([plug_id_col, "date"], sort=True)[voltage_col]
        .max()
        .rename("daily_max")
        .reset_index()
    )
    return daily


def add_future_7day_max_target(
    daily_df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    date_col: str = "date",
    daily_max_col: str = "daily_max",
    horizon: int = 7,
) -> pd.DataFrame:
    """日次最大電圧 DataFrame に future_7d_max 列を付与して返す（copy）。

    翌 7 カレンダー日（t+1〜t+horizon）の daily_max の最大値を計算する。
    非運転日は行なし（カレンダー補完して NaN として扱う）。
    ウィンドウ内が全て NaN → future_7d_max = NaN。
    """
    result = daily_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([plug_id_col, date_col])

    future_maxes: list[pd.Series] = []
    for plug_id, group in result.groupby(plug_id_col, sort=False):
        group = group.set_index(date_col)
        full_idx = pd.date_range(group.index.min(), group.index.max(), freq="D")
        reindexed = group[daily_max_col].reindex(full_idx)

        shifted = reindexed.shift(-1)
        fm = shifted[::-1].rolling(horizon, min_periods=1).max()[::-1]
        if len(fm) >= 1:
            last_valid_date = group.index.max()
            cutoff_date = last_valid_date - pd.Timedelta(days=horizon - 1)
            fm.loc[fm.index > cutoff_date] = float("nan")

        future_maxes.append(fm.reindex(group.index))

    all_future = pd.concat(future_maxes)
    result = result.reset_index(drop=True)
    result["future_7d_max"] = all_future.values
    return result
