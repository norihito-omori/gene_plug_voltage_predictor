from __future__ import annotations

import pandas as pd

from gene_plug_voltage_predictor.cleaning.pipeline import (
    CleaningHistory,
    CleaningPipeline,
    StepSpec,
)


def test_pipeline_applies_steps_in_order_and_records_history() -> None:
    df = pd.DataFrame(
        {
            "location": [5630, 8950, 8950, 9221],
            "cum_runtime_h": [1500, 500, 2000, 1800],
        }
    )
    specs = [
        StepSpec(
            name="exclude_locations",
            adr="decision-004",
            params={"location_col": "location", "excluded": [8950]},
        ),
        StepSpec(
            name="filter_cumulative_runtime",
            adr="decision-005",
            params={"runtime_col": "cum_runtime_h", "min_hours": 1000},
        ),
    ]
    pipeline = CleaningPipeline(specs)
    out_df, history = pipeline.run(df)

    assert list(out_df["location"]) == [5630, 9221]
    assert isinstance(history, CleaningHistory)
    assert len(history.steps) == 2
    assert history.steps[0].name == "exclude_locations"
    assert history.steps[0].excluded_rows == 2
    assert history.steps[1].excluded_rows == 0
    assert history.input_rows == 4
    assert history.output_rows == 2


def test_pipeline_raises_on_unknown_step() -> None:
    specs = [StepSpec(name="no_such_step", adr="decision-999", params={})]
    pipeline = CleaningPipeline(specs)
    df = pd.DataFrame({"location": [1]})
    try:
        pipeline.run(df)
    except ValueError as e:
        assert "unknown step" in str(e)
    else:
        raise AssertionError("should have raised")


def test_runtime_params_injects_events_dict() -> None:
    """runtime_params で assign_generation に events_by_location を注入できる。"""
    df = pd.DataFrame({
        "target_no": ["5630", "5630", "5630"],
        "dailygraphpt_ptdatetime": pd.to_datetime([
            "2023-01-01", "2023-06-01", "2024-01-01",
        ]),
    })
    specs = [
        StepSpec(
            name="assign_generation",
            adr="decision-014",
            params={"id_col": "target_no", "datetime_col": "dailygraphpt_ptdatetime"},
        ),
    ]
    events = {"5630": [pd.Timestamp("2023-05-01")]}
    pipeline = CleaningPipeline(specs)
    out_df, history = pipeline.run(
        df,
        runtime_params={"assign_generation": {"events_by_location": events}},
    )
    assert list(out_df["gen_no"]) == [0, 1, 1]
    assert len(history.steps) == 1


def test_runtime_params_overrides_static_params() -> None:
    df = pd.DataFrame({
        "target_no": ["5630"],
        "dailygraphpt_ptdatetime": pd.to_datetime(["2024-01-01"]),
    })
    specs = [
        StepSpec(
            name="assign_generation",
            adr="decision-014",
            params={
                "id_col": "target_no",
                "datetime_col": "dailygraphpt_ptdatetime",
                "events_by_location": {"5630": [pd.Timestamp("2025-01-01")]},  # 先の日、gen=0
            },
        ),
    ]
    pipeline = CleaningPipeline(specs)
    # runtime で events を上書き → gen=1 になる
    out_df, history = pipeline.run(
        df,
        runtime_params={
            "assign_generation": {
                "events_by_location": {"5630": [pd.Timestamp("2023-01-01")]},
            },
        },
    )
    assert list(out_df["gen_no"]) == [1]
    assert "runtime-overridden keys" in history.steps[0].note
