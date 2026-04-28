"""トレンド・統計特徴量生成。

voltage_trend_7d / days_since_exchange / rolling mean / rolling std。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _slope(values: np.ndarray) -> float:
    """values の線形回帰傾きを返す。2点未満の場合は NaN。

    rolling.apply(raw=True) から呼ばれ、numpy 配列を受け取る。
    """
    n = len(values)
    if n < 2:
        return float("nan")
    return float(np.polyfit(range(n), values, 1)[0])


def add_trend_features(
    daily_df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    date_col: str = "date",
    daily_max_col: str = "daily_max",
    gen_no_col: str = "gen_no",
    trend_window: int = 7,
) -> pd.DataFrame:
    """daily_df にトレンド・統計特徴量 4列を付与して返す（copy）。

    add_features() の出力（gen_no 列を含む）を入力とする前提。
    rolling は非運転日をスキップする行ベース（カレンダー日ではない）。
    .values で index alignment を回避する（lag 特徴量と同じパターン）。
    """
    result = daily_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([plug_id_col, date_col]).reset_index(drop=True)

    grouped = result.groupby(plug_id_col, sort=True)[daily_max_col]

    # voltage_trend_7d: 直近 trend_window 行の線形回帰傾き
    result["voltage_trend_7d"] = (
        grouped
        .rolling(trend_window, min_periods=2)
        .apply(_slope, raw=True)
        .reset_index(level=0, drop=True)
        .values
    )

    # daily_max_rolling_mean_7d: 直近 trend_window 行の移動平均
    result["daily_max_rolling_mean_7d"] = (
        grouped
        .rolling(trend_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .values
    )

    # daily_max_rolling_std_7d: 直近 trend_window 行の移動標準偏差
    result["daily_max_rolling_std_7d"] = (
        grouped
        .rolling(trend_window, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .values
    )

    # days_since_exchange: plug × gen_no の初日からの経過カレンダー日数
    origin = (
        result.groupby([plug_id_col, gen_no_col])[date_col]
        .min()
        .rename("_origin")
        .reset_index()
    )
    result = result.merge(origin, on=[plug_id_col, gen_no_col], how="left")
    result["days_since_exchange"] = (result[date_col] - result["_origin"]).dt.days
    result = result.drop(columns=["_origin"])

    return result
