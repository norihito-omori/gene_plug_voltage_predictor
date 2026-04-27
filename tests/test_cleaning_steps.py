from __future__ import annotations

import pandas as pd
import pytest

from gene_plug_voltage_predictor.cleaning.steps import (
    StepResult,
    assign_generation,
    compute_baseline,
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


def test_assign_generation_basic_boundaries() -> None:
    df = pd.DataFrame({
        "target_no": ["5630"] * 5,
        "dailygraphpt_ptdatetime": pd.to_datetime([
            "2022-07-01", "2023-05-14", "2023-05-15", "2024-08-22", "2025-01-01",
        ]),
    })
    events = {
        "5630": [pd.Timestamp("2023-05-15"), pd.Timestamp("2024-08-22")],
    }
    result = assign_generation(
        df,
        id_col="target_no",
        datetime_col="dailygraphpt_ptdatetime",
        events_by_location=events,
    )
    assert list(result.df["gen_no"]) == [0, 0, 1, 2, 2]
    assert result.excluded_rows == 0


def test_assign_generation_same_day_as_event_gets_new_gen() -> None:
    """交換日当日 00:00 / 14:30 / 23:59 のいずれも新世代側。"""
    df = pd.DataFrame({
        "target_no": ["5630"] * 3,
        "dailygraphpt_ptdatetime": pd.to_datetime([
            "2023-05-15 00:00:00",
            "2023-05-15 14:30:00",
            "2023-05-15 23:59:59",
        ]),
    })
    events = {"5630": [pd.Timestamp("2023-05-15")]}
    result = assign_generation(
        df, id_col="target_no", datetime_col="dailygraphpt_ptdatetime",
        events_by_location=events,
    )
    assert list(result.df["gen_no"]) == [1, 1, 1]


def test_assign_generation_no_events_assigns_all_zero() -> None:
    df = pd.DataFrame({
        "target_no": ["9140", "9140"],
        "dailygraphpt_ptdatetime": pd.to_datetime(["2024-01-01", "2025-01-01"]),
    })
    events: dict[str, list[pd.Timestamp]] = {"9140": []}
    result = assign_generation(
        df, id_col="target_no", datetime_col="dailygraphpt_ptdatetime",
        events_by_location=events,
    )
    assert list(result.df["gen_no"]) == [0, 0]


def test_assign_generation_missing_location_assigns_zero() -> None:
    df = pd.DataFrame({
        "target_no": ["5630", "9999"],
        "dailygraphpt_ptdatetime": pd.to_datetime(["2024-01-01", "2024-01-01"]),
    })
    events = {"5630": [pd.Timestamp("2023-06-01")]}
    result = assign_generation(
        df, id_col="target_no", datetime_col="dailygraphpt_ptdatetime",
        events_by_location=events,
    )
    # 9999 は dict に無い → gen=0、5630 は 2024-01-01 が events[0]=2023-06-01 より後 → gen=1
    assert list(result.df["gen_no"]) == [1, 0]
    assert "1 locations had no events" in result.note


def test_assign_generation_requires_existing_columns() -> None:
    df = pd.DataFrame({"target_no": ["5630"]})
    with pytest.raises(ValueError, match="missing required column"):
        assign_generation(
            df, id_col="target_no", datetime_col="missing_col",
            events_by_location={},
        )


def test_compute_baseline_happy_path() -> None:
    """30 日分運転（6 プラグ × 30 日 = 180 rows）で baseline 中央値が broadcast される。"""
    ts = pd.date_range("2023-06-01 00:00", periods=30, freq="D")
    rows = []
    for day in ts:
        for plug in range(1, 7):
            rows.append({
                "target_no": "5630",
                "dailygraphpt_ptdatetime": day,
                "発電機電力": 300.0,
                "要求電圧": 25.0 + plug,  # プラグごとに 26..31
                "管理No_プラグNo": f"5630_{plug}",
                "プラグNo": plug,
                "gen_no": 0,
            })
    df = pd.DataFrame(rows)
    result = compute_baseline(
        df,
        id_col="target_no",
        gen_col="gen_no",
        datetime_col="dailygraphpt_ptdatetime",
        voltage_col="要求電圧",
        power_col="発電機電力",
    )
    # 各日の max = 31（プラグ6）、30 日分の median = 31
    assert result.excluded_rows == 0
    assert set(result.df["baseline"].unique()) == {31.0}
    assert "1 組" in result.note or "1 groups" in result.note or "有効" in result.note


def test_compute_baseline_nan_when_active_days_below_threshold() -> None:
    """運転日 < min_active_days なら baseline = NaN。"""
    ts = pd.date_range("2023-06-01", periods=6, freq="D")
    rows = [{
        "target_no": "5630",
        "dailygraphpt_ptdatetime": day,
        "発電機電力": 300.0,
        "要求電圧": 30.0,
        "gen_no": 0,
    } for day in ts]
    df = pd.DataFrame(rows)
    result = compute_baseline(
        df, id_col="target_no", gen_col="gen_no",
        datetime_col="dailygraphpt_ptdatetime", voltage_col="要求電圧",
        power_col="発電機電力",
    )
    assert result.df["baseline"].isna().all()
    assert "NaN=1" in result.note


def test_compute_baseline_respects_gen_boundaries() -> None:
    """2 世代分の行で、それぞれ独立した baseline が broadcast される。"""
    rows: list[dict] = []
    for day in pd.date_range("2023-01-01", periods=10, freq="D"):
        rows.append({
            "target_no": "5630", "dailygraphpt_ptdatetime": day,
            "発電機電力": 300.0, "要求電圧": 30.0, "gen_no": 0,
        })
    for day in pd.date_range("2023-06-01", periods=10, freq="D"):
        rows.append({
            "target_no": "5630", "dailygraphpt_ptdatetime": day,
            "発電機電力": 300.0, "要求電圧": 22.0, "gen_no": 1,
        })
    df = pd.DataFrame(rows)
    result = compute_baseline(
        df, id_col="target_no", gen_col="gen_no",
        datetime_col="dailygraphpt_ptdatetime", voltage_col="要求電圧",
        power_col="発電機電力",
    )
    gen0_baselines = result.df.loc[result.df["gen_no"] == 0, "baseline"].unique()
    gen1_baselines = result.df.loc[result.df["gen_no"] == 1, "baseline"].unique()
    assert list(gen0_baselines) == [30.0]
    assert list(gen1_baselines) == [22.0]


def test_compute_baseline_requires_gen_col_and_voltage_col() -> None:
    df_no_gen = pd.DataFrame({
        "target_no": ["5630"],
        "dailygraphpt_ptdatetime": pd.to_datetime(["2023-01-01"]),
        "発電機電力": [300.0],
        "要求電圧": [30.0],
    })
    with pytest.raises(ValueError, match="run assign_generation first"):
        compute_baseline(
            df_no_gen, id_col="target_no", gen_col="gen_no",
            datetime_col="dailygraphpt_ptdatetime", voltage_col="要求電圧",
            power_col="発電機電力",
        )

    df_no_volt = pd.DataFrame({
        "target_no": ["5630"],
        "dailygraphpt_ptdatetime": pd.to_datetime(["2023-01-01"]),
        "発電機電力": [300.0],
        "gen_no": [0],
    })
    with pytest.raises(ValueError, match="run melt_voltage_columns first"):
        compute_baseline(
            df_no_volt, id_col="target_no", gen_col="gen_no",
            datetime_col="dailygraphpt_ptdatetime", voltage_col="要求電圧",
            power_col="発電機電力",
        )
