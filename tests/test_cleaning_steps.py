from __future__ import annotations

import pandas as pd
import pytest

from gene_plug_voltage_predictor.cleaning.steps import (
    StepResult,
    exclude_location_plug,
    exclude_locations,
    filter_by_rated_power_ratio,
    filter_cumulative_runtime,
    melt_voltage_columns,
)


def test_exclude_locations_removes_listed_locations() -> None:
    df = pd.DataFrame({"location": [5630, 8950, 9221], "v": [1.0, 2.0, 3.0]})
    result = exclude_locations(df, location_col="location", excluded=(8950,))
    assert list(result.df["location"]) == [5630, 9221]
    assert result.excluded_rows == 1
    assert "8950" in result.note


def test_filter_cumulative_runtime_removes_below_threshold() -> None:
    df = pd.DataFrame({"cum_runtime_h": [100, 999, 1000, 5000]})
    result = filter_cumulative_runtime(df, runtime_col="cum_runtime_h", min_hours=1000)
    assert list(result.df["cum_runtime_h"]) == [1000, 5000]
    assert result.excluded_rows == 2


def test_filter_by_rated_power_ratio_keeps_only_80pct_and_above() -> None:
    df = pd.DataFrame({"power_kw": [100, 200, 296, 300]})
    # rated=370 → 80% = 296
    result = filter_by_rated_power_ratio(
        df, power_col="power_kw", rated_kw=370, ratio=0.80
    )
    assert list(result.df["power_kw"]) == [296, 300]
    assert result.excluded_rows == 2


def test_step_result_is_dataclass_like() -> None:
    df = pd.DataFrame({"location": [5630], "v": [1.0]})
    r = exclude_locations(df, location_col="location", excluded=())
    assert isinstance(r, StepResult)
    assert r.excluded_rows == 0


def test_melt_voltage_columns_expands_to_long_format() -> None:
    """ADR-009: 6 列の要求電圧を縦持ちに展開し、管理No_プラグNo を生成する。"""
    df = pd.DataFrame(
        {
            "target_no": [563, 576],
            "dailygraphpt_ptdatetime": ["2018-03-27 00:30:00", "2018-03-27 00:30:00"],
            "要求電圧_1": [221, 321],
            "要求電圧_2": [222, 322],
            "要求電圧_3": [223, 323],
            "要求電圧_4": [224, 324],
            "要求電圧_5": [225, 325],
            "要求電圧_6": [226, 326],
        }
    )
    voltage_cols = (
        "要求電圧_1",
        "要求電圧_2",
        "要求電圧_3",
        "要求電圧_4",
        "要求電圧_5",
        "要求電圧_6",
    )
    result = melt_voltage_columns(
        df, location_no_col="target_no", voltage_cols=voltage_cols
    )
    assert result.excluded_rows == 0
    # 2 input rows × 6 voltage cols = 12 output rows
    assert len(result.df) == 12
    assert set(result.df.columns) >= {"要求電圧", "プラグNo", "管理No_プラグNo"}
    # voltage column and plug number column should be populated
    row_first = result.df[
        (result.df["target_no"] == 563) & (result.df["プラグNo"] == 1)
    ]
    assert len(row_first) == 1
    assert row_first["要求電圧"].iloc[0] == 221
    assert row_first["管理No_プラグNo"].iloc[0] == "563_1"
    row_last = result.df[
        (result.df["target_no"] == 576) & (result.df["プラグNo"] == 6)
    ]
    assert row_last["要求電圧"].iloc[0] == 326
    assert row_last["管理No_プラグNo"].iloc[0] == "576_6"
    # original voltage columns are gone after melt
    for c in voltage_cols:
        assert c not in result.df.columns
    assert "melted" in result.note


def test_melt_voltage_columns_rejects_empty_voltage_cols() -> None:
    """空の voltage_cols は ValueError となる。"""
    df = pd.DataFrame({"target_no": [563], "要求電圧_1": [221]})
    with pytest.raises(ValueError, match="non-empty"):
        melt_voltage_columns(df, location_no_col="target_no", voltage_cols=())


def test_melt_voltage_columns_rejects_non_digit_suffix() -> None:
    """末尾が数字でない列名は ValueError となる。"""
    df = pd.DataFrame(
        {"target_no": [563], "要求電圧_1": [221], "unknown_col": [0]}
    )
    with pytest.raises(ValueError, match="must end in a digit"):
        melt_voltage_columns(
            df,
            location_no_col="target_no",
            voltage_cols=("要求電圧_1", "unknown_col"),
        )


def test_exclude_location_plug_removes_by_composite_id() -> None:
    """ADR-001 L-03: 管理No_プラグNo=9221_4 を除外する。"""
    df = pd.DataFrame(
        {
            "管理No_プラグNo": ["5630_1", "9221_4", "9221_5", "5630_6"],
            "要求電圧": [220, 230, 231, 221],
        }
    )
    result = exclude_location_plug(
        df, id_col="管理No_プラグNo", excluded=["9221_4"]
    )
    assert list(result.df["管理No_プラグNo"]) == ["5630_1", "9221_5", "5630_6"]
    assert result.excluded_rows == 1
    assert "9221_4" in result.note


def test_exclude_location_plug_with_empty_excluded_returns_no_change() -> None:
    """空の excluded では行数も内容も変わらず、note は 'no ids excluded'。"""
    df = pd.DataFrame(
        {
            "管理No_プラグNo": ["5630_1", "9221_4", "5630_6"],
            "要求電圧": [220, 230, 221],
        }
    )
    result = exclude_location_plug(df, id_col="管理No_プラグNo", excluded=())
    assert list(result.df["管理No_プラグNo"]) == ["5630_1", "9221_4", "5630_6"]
    assert list(result.df["要求電圧"]) == [220, 230, 221]
    assert result.excluded_rows == 0
    assert result.note == "no ids excluded"
