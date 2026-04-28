"""tests/test_features_features.py"""
from __future__ import annotations

import pandas as pd

from gene_plug_voltage_predictor.features.features import add_features


def _make_daily_df(
    plug_id: str,
    dates: list[str],
    daily_max: list[float],
) -> pd.DataFrame:
    return pd.DataFrame({
        "管理No_プラグNo": plug_id,
        "date": pd.to_datetime(dates),
        "daily_max": daily_max,
    })


def _make_cleaned_df(
    plug_id: str,
    datetimes: list[str],
    power: list[float],
    runtime: list[float],
    baseline: float,
    gen_no: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame({
        "管理No_プラグNo": plug_id,
        "dailygraphpt_ptdatetime": pd.to_datetime(datetimes),
        "発電機電力": power,
        "累積運転時間": runtime,
        "baseline": baseline,
        "gen_no": gen_no,
    })


def test_add_features_voltage_vs_baseline() -> None:
    """voltage_vs_baseline = daily_max - baseline が正しく計算される。"""
    daily_df = _make_daily_df(
        "5630_1",
        ["2024-01-01", "2024-01-02"],
        [220.0, 225.0],
    )
    cleaned_df = _make_cleaned_df(
        "5630_1",
        ["2024-01-01 00:30", "2024-01-01 01:00",
         "2024-01-02 00:30", "2024-01-02 01:00"],
        [300.0, 300.0, 300.0, 300.0],
        [100.0, 101.0, 102.0, 103.0],
        baseline=200.0,
    )
    result = add_features(daily_df, cleaned_df)
    assert "voltage_vs_baseline" in result.columns
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-01"), "voltage_vs_baseline"
    ].iloc[0] == 20.0
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-02"), "voltage_vs_baseline"
    ].iloc[0] == 25.0


def test_add_features_baseline_nan_propagates() -> None:
    """baseline=NaN のとき voltage_vs_baseline=NaN。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    cleaned_df = _make_cleaned_df(
        "5630_1",
        ["2024-01-01 00:30"],
        [300.0],
        [100.0],
        baseline=float("nan"),
    )
    result = add_features(daily_df, cleaned_df)
    assert pd.isna(
        result.loc[
            result["date"] == pd.Timestamp("2024-01-01"), "voltage_vs_baseline"
        ].iloc[0]
    )


def test_add_features_returns_copy() -> None:
    """入力 daily_df を変更しない（copy を返す）。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    cleaned_df = _make_cleaned_df(
        "5630_1", ["2024-01-01 00:30"], [300.0], [100.0], baseline=200.0
    )
    original_cols = set(daily_df.columns)
    _ = add_features(daily_df, cleaned_df)
    assert set(daily_df.columns) == original_cols
