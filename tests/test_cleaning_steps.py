from __future__ import annotations

import pandas as pd

from gene_plug_voltage_predictor.cleaning.steps import (
    StepResult,
    exclude_locations,
    filter_by_rated_power_ratio,
    filter_cumulative_runtime,
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
