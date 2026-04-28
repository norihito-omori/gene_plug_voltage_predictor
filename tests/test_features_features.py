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


def test_add_features_lags() -> None:
    """daily_max_lag_1/3/7 が plug 単位で正しくシフトされる。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 9)]  # 8 日
    values = [float(200 + i) for i in range(8)]  # 200..207
    daily_df = _make_daily_df("5630_1", dates, values)
    cleaned_df = _make_cleaned_df(
        "5630_1",
        [f"2024-01-{d:02d} 00:30" for d in range(1, 9)],
        [300.0] * 8,
        [float(100 + i) for i in range(8)],
        baseline=200.0,
    )
    result = add_features(daily_df, cleaned_df)
    # lag_1: day2 の lag_1 = day1 の daily_max = 200
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-02"), "daily_max_lag_1"
    ].iloc[0] == 200.0
    # lag_3: day4 の lag_3 = day1 の daily_max = 200
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-04"), "daily_max_lag_3"
    ].iloc[0] == 200.0
    # lag_7: day8 の lag_7 = day1 の daily_max = 200
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-08"), "daily_max_lag_7"
    ].iloc[0] == 200.0
    # lag_1: day1 の lag_1 = NaN（先頭）
    assert pd.isna(
        result.loc[
            result["date"] == pd.Timestamp("2024-01-01"), "daily_max_lag_1"
        ].iloc[0]
    )


def test_add_features_lag_no_cross_plug() -> None:
    """plug A の lag が plug B に漏れない。"""
    daily_a = _make_daily_df("A_1", ["2024-01-01", "2024-01-02"], [999.0, 999.0])
    daily_b = _make_daily_df("B_1", ["2024-01-01", "2024-01-02"], [100.0, 100.0])
    daily_df = pd.concat([daily_a, daily_b], ignore_index=True)
    cleaned_a = _make_cleaned_df(
        "A_1",
        ["2024-01-01 00:30", "2024-01-02 00:30"],
        [300.0, 300.0],
        [100.0, 101.0],
        baseline=200.0,
    )
    cleaned_b = _make_cleaned_df(
        "B_1",
        ["2024-01-01 00:30", "2024-01-02 00:30"],
        [300.0, 300.0],
        [100.0, 101.0],
        baseline=200.0,
    )
    cleaned_df = pd.concat([cleaned_a, cleaned_b], ignore_index=True)
    result = add_features(daily_df, cleaned_df)
    # B_1 の day2 の lag_1 は B_1 の day1 の値 (100.0)
    b_lag = result.loc[
        (result["管理No_プラグNo"] == "B_1")
        & (result["date"] == pd.Timestamp("2024-01-02")),
        "daily_max_lag_1",
    ].iloc[0]
    assert b_lag == 100.0


def test_add_features_operating_ratio() -> None:
    """発電機電力 >= rated_kw * 0.8 の行数 ÷ 全行数が正しく計算される。"""
    # 4 行中 3 行が定格 80% 以上 → 稼働割合 = 0.75
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    cleaned_df = _make_cleaned_df(
        "5630_1",
        [
            "2024-01-01 00:00",
            "2024-01-01 00:30",
            "2024-01-01 01:00",
            "2024-01-01 01:30",
        ],
        [296.0, 296.0, 296.0, 200.0],  # 370 * 0.8 = 296; 最後の1行は閾値未満
        [100.0, 101.0, 102.0, 103.0],
        baseline=200.0,
        gen_no=0,
    )
    result = add_features(daily_df, cleaned_df, rated_kw=370.0)
    val = result.loc[
        result["date"] == pd.Timestamp("2024-01-01"), "稼働割合"
    ].iloc[0]
    assert abs(val - 0.75) < 1e-9


def test_add_features_cumulative_runtime() -> None:
    """plug × 日付の累積運転時間最大値が返る。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    cleaned_df = _make_cleaned_df(
        "5630_1",
        ["2024-01-01 00:30", "2024-01-01 01:00", "2024-01-01 01:30"],
        [300.0, 300.0, 300.0],
        [100.0, 101.0, 102.5],  # 最大値は 102.5
        baseline=200.0,
    )
    result = add_features(daily_df, cleaned_df)
    val = result.loc[
        result["date"] == pd.Timestamp("2024-01-01"), "累積運転時間"
    ].iloc[0]
    assert val == 102.5
