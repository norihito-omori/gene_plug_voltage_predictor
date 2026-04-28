"""tests/test_features_target.py"""
from __future__ import annotations

import pandas as pd
import pytest

from gene_plug_voltage_predictor.features.target import aggregate_daily_max_voltage


def _make_30min_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["dailygraphpt_ptdatetime"] = pd.to_datetime(df["dailygraphpt_ptdatetime"])
    return df


def test_aggregate_daily_max_voltage_basic() -> None:
    """2 plug × 2 日で daily_max が正しく集約される。"""
    df = _make_30min_df([
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-01 00:30", "要求電圧": 220.0, "発電機電力": 300.0},
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-01 01:00", "要求電圧": 225.0, "発電機電力": 300.0},
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-02 00:30", "要求電圧": 230.0, "発電機電力": 300.0},
        {"管理No_プラグNo": "5630_2", "dailygraphpt_ptdatetime": "2024-01-01 00:30", "要求電圧": 210.0, "発電機電力": 300.0},
    ])
    result = aggregate_daily_max_voltage(df)
    assert set(result.columns) == {"管理No_プラグNo", "date", "daily_max"}
    a1 = result[(result["管理No_プラグNo"] == "5630_1") & (result["date"] == pd.Timestamp("2024-01-01"))]
    assert a1["daily_max"].iloc[0] == 225.0
    a2 = result[(result["管理No_プラグNo"] == "5630_1") & (result["date"] == pd.Timestamp("2024-01-02"))]
    assert a2["daily_max"].iloc[0] == 230.0
    b1 = result[(result["管理No_プラグNo"] == "5630_2") & (result["date"] == pd.Timestamp("2024-01-01"))]
    assert b1["daily_max"].iloc[0] == 210.0


def test_aggregate_daily_max_voltage_excludes_non_running() -> None:
    """発電機電力 <= 0 の行は集約対象外。"""
    df = _make_30min_df([
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-01 00:30", "要求電圧": 250.0, "発電機電力": 0.0},
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-01 01:00", "要求電圧": 220.0, "発電機電力": 300.0},
    ])
    result = aggregate_daily_max_voltage(df)
    assert result["daily_max"].iloc[0] == 220.0


def test_aggregate_daily_max_voltage_no_running_day_omitted() -> None:
    """全非運転日（発電機電力 <= 0 のみ）は日次テーブルに行が存在しない。"""
    df = _make_30min_df([
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-01 00:30", "要求電圧": 220.0, "発電機電力": 300.0},
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-02 00:30", "要求電圧": 999.0, "発電機電力": 0.0},
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-03 00:30", "要求電圧": 225.0, "発電機電力": 300.0},
    ])
    result = aggregate_daily_max_voltage(df)
    dates = result["date"].tolist()
    assert pd.Timestamp("2024-01-02") not in dates
    assert pd.Timestamp("2024-01-01") in dates
    assert pd.Timestamp("2024-01-03") in dates
