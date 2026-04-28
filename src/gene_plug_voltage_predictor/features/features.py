"""説明変数生成。Phase 0 X 列: baseline/gen_no/voltage_vs_baseline/lag/稼働割合/累積運転時間。"""
from __future__ import annotations

import pandas as pd


def add_features(
    daily_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    date_col: str = "date",
    daily_max_col: str = "daily_max",
    baseline_col: str = "baseline",
    gen_no_col: str = "gen_no",
    runtime_col: str = "累積運転時間",
    power_col: str = "発電機電力",
    datetime_col: str = "dailygraphpt_ptdatetime",
    rated_kw: float = 370.0,
    lags: tuple[int, ...] = (1, 3, 7),
) -> pd.DataFrame:
    """日次最大電圧 DataFrame に説明変数列を付与して返す（copy）。

    cleaned_df（30分粒度）から baseline/gen_no/稼働割合/累積運転時間を日次集約して
    daily_df に left join し、lag 特徴量を plug 単位で付与する。
    """
    result = daily_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])

    # cleaned_df から日次集約テーブルを作成
    tmp = cleaned_df.copy()
    tmp["_date"] = pd.to_datetime(tmp[datetime_col]).dt.normalize()
    threshold = rated_kw * 0.8

    # Compute active ratio separately to avoid lambda
    active = (
        tmp.assign(_active=(tmp[power_col] >= threshold).astype(float))
        .groupby([plug_id_col, "_date"])["_active"]
        .mean()
        .rename("稼働割合")
        .reset_index()
    )
    daily_clean = (
        tmp.groupby([plug_id_col, "_date"], sort=True)
        .agg(
            **{
                baseline_col: (baseline_col, "first"),
                gen_no_col: (gen_no_col, "first"),
                runtime_col: (runtime_col, "max"),
            }
        )
        .reset_index()
        .merge(active, on=[plug_id_col, "_date"], how="left")
        .rename(columns={"_date": date_col})
    )

    # daily_df に left join
    result = result.merge(
        daily_clean[
            [plug_id_col, date_col, baseline_col, gen_no_col, "稼働割合", runtime_col]
        ],
        on=[plug_id_col, date_col],
        how="left",
    )

    # voltage_vs_baseline
    result["voltage_vs_baseline"] = result[daily_max_col] - result[baseline_col]

    # lag 特徴量（plug 単位）
    result = result.sort_values([plug_id_col, date_col]).reset_index(drop=True)
    for lag in lags:
        col = f"{daily_max_col}_lag_{lag}"
        result[col] = (
            result.groupby(plug_id_col, sort=True)[daily_max_col].shift(lag).values
        )

    return result
