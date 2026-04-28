"""tests/test_features_target.py"""
from __future__ import annotations

import pandas as pd

from gene_plug_voltage_predictor.features.target import (
    add_future_7day_max_target,
    aggregate_daily_max_voltage,
)


def _make_30min_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["dailygraphpt_ptdatetime"] = pd.to_datetime(df["dailygraphpt_ptdatetime"])
    return df


def test_aggregate_daily_max_voltage_basic() -> None:
    """2 plug × 2 日で daily_max が正しく集約される。"""
    df = _make_30min_df([
        {
            "管理No_プラグNo": "5630_1",
            "dailygraphpt_ptdatetime": "2024-01-01 00:30",
            "要求電圧": 220.0,
            "発電機電力": 300.0,
        },
        {
            "管理No_プラグNo": "5630_1",
            "dailygraphpt_ptdatetime": "2024-01-01 01:00",
            "要求電圧": 225.0,
            "発電機電力": 300.0,
        },
        {
            "管理No_プラグNo": "5630_1",
            "dailygraphpt_ptdatetime": "2024-01-02 00:30",
            "要求電圧": 230.0,
            "発電機電力": 300.0,
        },
        {
            "管理No_プラグNo": "5630_2",
            "dailygraphpt_ptdatetime": "2024-01-01 00:30",
            "要求電圧": 210.0,
            "発電機電力": 300.0,
        },
    ])
    result = aggregate_daily_max_voltage(df)
    assert set(result.columns) == {"管理No_プラグNo", "date", "daily_max"}
    mask_a1 = (result["管理No_プラグNo"] == "5630_1") & (
        result["date"] == pd.Timestamp("2024-01-01")
    )
    a1 = result[mask_a1]
    assert a1["daily_max"].iloc[0] == 225.0
    mask_a2 = (result["管理No_プラグNo"] == "5630_1") & (
        result["date"] == pd.Timestamp("2024-01-02")
    )
    a2 = result[mask_a2]
    assert a2["daily_max"].iloc[0] == 230.0
    mask_b1 = (result["管理No_プラグNo"] == "5630_2") & (
        result["date"] == pd.Timestamp("2024-01-01")
    )
    b1 = result[mask_b1]
    assert b1["daily_max"].iloc[0] == 210.0


def test_aggregate_daily_max_voltage_excludes_non_running() -> None:
    """発電機電力 <= 0 の行は集約対象外。"""
    df = _make_30min_df([
        {
            "管理No_プラグNo": "5630_1",
            "dailygraphpt_ptdatetime": "2024-01-01 00:30",
            "要求電圧": 250.0,
            "発電機電力": 0.0,
        },
        {
            "管理No_プラグNo": "5630_1",
            "dailygraphpt_ptdatetime": "2024-01-01 01:00",
            "要求電圧": 220.0,
            "発電機電力": 300.0,
        },
    ])
    result = aggregate_daily_max_voltage(df)
    assert result["daily_max"].iloc[0] == 220.0


def test_aggregate_daily_max_voltage_no_running_day_omitted() -> None:
    """全非運転日（発電機電力 <= 0 のみ）は日次テーブルに行が存在しない。"""
    df = _make_30min_df([
        {
            "管理No_プラグNo": "5630_1",
            "dailygraphpt_ptdatetime": "2024-01-01 00:30",
            "要求電圧": 220.0,
            "発電機電力": 300.0,
        },
        {
            "管理No_プラグNo": "5630_1",
            "dailygraphpt_ptdatetime": "2024-01-02 00:30",
            "要求電圧": 999.0,
            "発電機電力": 0.0,
        },
        {
            "管理No_プラグNo": "5630_1",
            "dailygraphpt_ptdatetime": "2024-01-03 00:30",
            "要求電圧": 225.0,
            "発電機電力": 300.0,
        },
    ])
    result = aggregate_daily_max_voltage(df)
    dates = result["date"].tolist()
    assert pd.Timestamp("2024-01-02") not in dates
    assert pd.Timestamp("2024-01-01") in dates
    assert pd.Timestamp("2024-01-03") in dates


def _make_daily_df(plug_id: str, dates: list[str], values: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame({
        "管理No_プラグNo": plug_id,
        "date": pd.to_datetime(dates),
        "daily_max": [float("nan") if v is None else v for v in values],
    })


def test_add_future_7day_max_target_basic() -> None:
    """10 日データで future_7d_max が正しく計算される。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    values = [float(200 + i) for i in range(10)]  # 200, 201, ..., 209
    df = _make_daily_df("5630_1", dates, values)
    result = add_future_7day_max_target(df)
    # day 1 (index 0): t+1〜t+7 = days 2〜8 → max(201..207) = 207
    val_day1 = result.loc[result["date"] == pd.Timestamp("2024-01-01"), "future_7d_max"].iloc[0]
    assert val_day1 == 207.0
    # day 3 (index 2): t+1〜t+7 = days 4〜10 → max(203..209) = 209
    val_day3 = result.loc[result["date"] == pd.Timestamp("2024-01-03"), "future_7d_max"].iloc[0]
    assert val_day3 == 209.0


def test_add_future_7day_max_target_terminal_nan() -> None:
    """最後の horizon 行は future_7d_max = NaN。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]  # 10 日
    values = [220.0] * 10
    df = _make_daily_df("5630_1", dates, values)
    result = add_future_7day_max_target(df, horizon=7)
    # 最後の 7 日（day 4〜10）は t+1〜t+7 が範囲外 → NaN
    terminal = result[result["date"] >= pd.Timestamp("2024-01-04")]
    assert terminal["future_7d_max"].isna().all()


def test_add_future_7day_max_target_plug_isolation() -> None:
    """plug A の値が plug B の future_7d_max に影響しない。"""
    df_a = _make_daily_df("A_1", ["2024-01-01", "2024-01-02", "2024-01-03"], [999.0, 999.0, 999.0])
    df_b = _make_daily_df("B_1", ["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 100.0, 100.0])
    df = pd.concat([df_a, df_b], ignore_index=True)
    result = add_future_7day_max_target(df, horizon=7)
    b_vals = result[result["管理No_プラグNo"] == "B_1"]["future_7d_max"].dropna()
    assert (b_vals <= 100.0).all()


def test_add_future_7day_max_target_partial_nan() -> None:
    """一部 NaN（非運転日）はスキップして残りの最大を返す。"""
    df = pd.DataFrame({
        "管理No_プラグNo": ["5630_1", "5630_1", "5630_1"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
        "daily_max": [210.0, 220.0, 230.0],
    })
    result = add_future_7day_max_target(df, horizon=7)
    # day 1 の future_7d_max: t+1〜t+7 = 2〜8日。稼働は 2日(220)と4日(230) → max=230
    val = result.loc[result["date"] == pd.Timestamp("2024-01-01"), "future_7d_max"].iloc[0]
    assert val == 230.0


def test_add_future_7day_max_target_calendar_gap() -> None:
    """非運転日（行なし）を挟んでもカレンダー 7 日で正しく計算される。"""
    # day 1 の t+1〜t+7 (day2〜8) に day8(=999) が入る → future_7d_max=999
    df = pd.DataFrame({
        "管理No_プラグNo": ["5630_1", "5630_1", "5630_1"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-08"]),
        "daily_max": [220.0, 200.0, 999.0],
    })
    result = add_future_7day_max_target(df, horizon=7)
    val = result.loc[result["date"] == pd.Timestamp("2024-01-01"), "future_7d_max"].iloc[0]
    assert val == 999.0
    # day9 は t+8 なのでウィンドウ外 → future_7d_max は day2(200) のみ
    df2 = pd.DataFrame({
        "管理No_プラグNo": ["5630_1", "5630_1", "5630_1"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-09"]),
        "daily_max": [220.0, 200.0, 999.0],
    })
    result2 = add_future_7day_max_target(df2, horizon=7)
    val2 = result2.loc[result2["date"] == pd.Timestamp("2024-01-01"), "future_7d_max"].iloc[0]
    assert val2 == 200.0
