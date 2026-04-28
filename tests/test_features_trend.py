"""tests/test_features_trend.py"""
from __future__ import annotations

import pandas as pd

from gene_plug_voltage_predictor.features.trend_features import add_trend_features


def _make_daily_df(
    plug_id: str,
    dates: list[str],
    daily_max: list[float],
    gen_no: int | list[int] = 0,
) -> pd.DataFrame:
    n = len(dates)
    gen_nos = [gen_no] * n if isinstance(gen_no, int) else gen_no
    return pd.DataFrame({
        "管理No_プラグNo": plug_id,
        "date": pd.to_datetime(dates),
        "daily_max": daily_max,
        "gen_no": gen_nos,
    })


def test_add_trend_features_columns() -> None:
    """4列が全て追加されている。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    daily_df = _make_daily_df("5630_1", dates, [float(200 + i) for i in range(10)])
    result = add_trend_features(daily_df)
    expected_cols = {
        "voltage_trend_7d",
        "days_since_exchange",
        "daily_max_rolling_mean_7d",
        "daily_max_rolling_std_7d",
    }
    assert expected_cols.issubset(set(result.columns))


def test_add_trend_features_returns_copy() -> None:
    """入力 daily_df を変更しない（copy を返す）。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    original_cols = set(daily_df.columns)
    _ = add_trend_features(daily_df)
    assert set(daily_df.columns) == original_cols


def test_voltage_trend_7d_slope() -> None:
    """単調増加系列で傾き > 0、単調減少で傾き < 0。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 9)]

    # 単調増加: +2 V/日
    inc_values = [200.0 + i * 2.0 for i in range(8)]
    result_inc = add_trend_features(_make_daily_df("5630_1", dates, inc_values))
    slope_inc = result_inc.loc[
        result_inc["date"] == pd.Timestamp("2024-01-08"), "voltage_trend_7d"
    ].iloc[0]
    assert slope_inc > 0

    # 単調減少: -2 V/日
    dec_values = [200.0 - i * 2.0 for i in range(8)]
    result_dec = add_trend_features(_make_daily_df("5630_2", dates, dec_values))
    slope_dec = result_dec.loc[
        result_dec["date"] == pd.Timestamp("2024-01-08"), "voltage_trend_7d"
    ].iloc[0]
    assert slope_dec < 0


def test_voltage_trend_7d_first_row_nan() -> None:
    """先頭行（1点のみ）の voltage_trend_7d は NaN（min_periods=2）。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01", "2024-01-02"], [220.0, 222.0])
    result = add_trend_features(daily_df)
    assert pd.isna(
        result.loc[result["date"] == pd.Timestamp("2024-01-01"), "voltage_trend_7d"].iloc[0]
    )


def test_days_since_exchange_starts_at_zero() -> None:
    """各世代（gen_no）の初日が 0。"""
    dates = [
        "2024-01-01", "2024-01-02", "2024-01-03",
        "2024-01-04", "2024-01-05", "2024-01-06",
    ]
    gen_nos = [0, 0, 0, 1, 1, 1]
    daily_df = _make_daily_df("5630_1", dates, [220.0] * 6, gen_no=gen_nos)
    result = add_trend_features(daily_df)
    # gen_no=0 の初日
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-01"), "days_since_exchange"
    ].iloc[0] == 0
    # gen_no=1 の初日（交換直後）
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-04"), "days_since_exchange"
    ].iloc[0] == 0


def test_days_since_exchange_increments() -> None:
    """翌日が 1、その翌日が 2 と増加する。"""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    daily_df = _make_daily_df("5630_1", dates, [220.0, 221.0, 222.0])
    result = add_trend_features(daily_df)
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-01"), "days_since_exchange"
    ].iloc[0] == 0
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-02"), "days_since_exchange"
    ].iloc[0] == 1
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-03"), "days_since_exchange"
    ].iloc[0] == 2


def test_rolling_mean_7d() -> None:
    """既知値で rolling mean が正しく計算される。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 8)]
    values = [210.0, 212.0, 208.0, 215.0, 211.0, 213.0, 209.0]
    daily_df = _make_daily_df("5630_1", dates, values)
    result = add_trend_features(daily_df)
    # day 7 の rolling mean（7日分の平均）
    expected = sum(values) / 7
    actual = result.loc[
        result["date"] == pd.Timestamp("2024-01-07"), "daily_max_rolling_mean_7d"
    ].iloc[0]
    assert abs(actual - expected) < 1e-9
    # day 1 は min_periods=1 なので values[0] そのまま
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-01"), "daily_max_rolling_mean_7d"
    ].iloc[0] == 210.0


def test_rolling_std_7d_nan_for_single_row() -> None:
    """1行のみの場合 rolling_std は NaN（min_periods=2）。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    result = add_trend_features(daily_df)
    assert pd.isna(
        result.loc[
            result["date"] == pd.Timestamp("2024-01-01"), "daily_max_rolling_std_7d"
        ].iloc[0]
    )


def test_no_cross_plug_leakage() -> None:
    """plug A の rolling が plug B の結果に影響しない。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 8)]
    daily_a = _make_daily_df("A_1", dates, [999.0] * 7)
    daily_b = _make_daily_df("B_1", dates, [100.0] * 7)
    daily_df = pd.concat([daily_a, daily_b], ignore_index=True)
    result = add_trend_features(daily_df)
    b_mean = result.loc[
        (result["管理No_プラグNo"] == "B_1")
        & (result["date"] == pd.Timestamp("2024-01-07")),
        "daily_max_rolling_mean_7d",
    ].iloc[0]
    assert b_mean == 100.0
