"""30分粒度データから hours_at_threshold 特徴量を生成する。"""
from __future__ import annotations

import pandas as pd


def add_hours_at_threshold(
    daily_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    *,
    threshold: int = 31,
    plug_id_col: str = "管理No_プラグNo",
    datetime_col: str = "dailygraphpt_ptdatetime",
    voltage_col: str = "要求電圧",
    power_col: str = "発電機電力",
    date_col: str = "date",
) -> pd.DataFrame:
    """日次 DataFrame に hours_at_{threshold}kv 列を付与して返す（copy）。

    各プラグ × 日付について、稼働中（発電機電力 > 0）かつ voltage == threshold の
    30分スロット数 × 0.5 を hours_at_{threshold}kv として計算する。
    非運転日は 0.0。
    """
    col_name = f"hours_at_{threshold}kv"

    running = cleaned_df.loc[cleaned_df[power_col] > 0].copy()
    running["_date"] = pd.to_datetime(running[datetime_col]).dt.normalize()
    running["_at_thr"] = (running[voltage_col] == threshold).astype(float)

    hourly = (
        running.groupby([plug_id_col, "_date"])["_at_thr"]
        .sum()
        .mul(0.5)
        .rename(col_name)
        .reset_index()
        .rename(columns={"_date": date_col})
    )
    hourly[date_col] = pd.to_datetime(hourly[date_col])

    result = daily_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.merge(hourly, on=[plug_id_col, date_col], how="left")
    result[col_name] = result[col_name].fillna(0.0)
    return result
