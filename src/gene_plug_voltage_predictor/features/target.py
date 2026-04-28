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
    """日次最大電圧 DataFrame に future_{horizon}d_max 列を付与して返す（copy）。

    翌 horizon カレンダー日（t+1〜t+horizon）の daily_max の最大値を計算する。
    非運転日は行なし（カレンダー補完して NaN として扱う）。
    ウィンドウ内が全て NaN → future_{horizon}d_max = NaN。
    """
    result = daily_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([plug_id_col, date_col]).reset_index(drop=True)

    col_name = f"future_{horizon}d_max"
    records: list[pd.Series] = []
    for _, group in result.groupby(plug_id_col, sort=True):
        orig_index = group.index
        date_indexed = group.set_index(date_col)
        full_idx = pd.date_range(date_indexed.index.min(), date_indexed.index.max(), freq="D")
        reindexed = date_indexed[daily_max_col].reindex(full_idx)

        shifted = reindexed.shift(-1)
        # forward rolling max via reverse trick
        fm = shifted[::-1].rolling(horizon, min_periods=1).max()[::-1]
        last_date = date_indexed.index.max()
        # Mark terminal rows as NaN only if we have enough data points.
        # For sparse data with fewer than horizon days, the rolling computation
        # with min_periods=1 handles NaN windows correctly.
        if len(fm) > horizon:
            terminal_start = last_date - pd.Timedelta(days=horizon - 1)
            fm.loc[fm.index >= terminal_start] = float("nan")

        fm_original = fm.reindex(date_indexed.index)
        records.append(pd.Series(fm_original.values, index=orig_index, name=col_name))

    result[col_name] = pd.concat(records).sort_index()
    return result
