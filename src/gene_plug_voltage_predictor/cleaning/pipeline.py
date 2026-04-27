"""クリーニングステップを順番に適用し、履歴を残すオーケストレータ。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .steps import (
    StepResult,
    assign_generation,
    compute_baseline,
    exclude_location_plug,
    exclude_locations,
    filter_by_rated_power_ratio,
    filter_cumulative_runtime,
    melt_voltage_columns,
)

_logger = logging.getLogger(__name__)

StepFn = Callable[..., StepResult]

_STEP_REGISTRY: dict[str, StepFn] = {
    "exclude_locations": exclude_locations,
    "filter_cumulative_runtime": filter_cumulative_runtime,
    "filter_by_rated_power_ratio": filter_by_rated_power_ratio,
    "melt_voltage_columns": melt_voltage_columns,
    "exclude_location_plug": exclude_location_plug,
    "assign_generation": assign_generation,
    "compute_baseline": compute_baseline,
}


@dataclass(frozen=True)
class StepSpec:
    name: str
    adr: str
    params: dict[str, Any]


@dataclass(frozen=True)
class StepHistory:
    name: str
    adr: str
    excluded_rows: int
    rows_before: int
    rows_after: int
    note: str


@dataclass(frozen=True)
class CleaningHistory:
    input_rows: int
    output_rows: int
    steps: list[StepHistory] = field(default_factory=list)


class CleaningPipeline:
    def __init__(self, specs: list[StepSpec]) -> None:
        self._specs = specs

    def run(
        self,
        df: pd.DataFrame,
        *,
        runtime_params: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[pd.DataFrame, CleaningHistory]:
        input_rows = len(df)
        history: list[StepHistory] = []
        current = df
        runtime_params = runtime_params or {}

        # typo 検出: 未知ステップ名は warning
        known_steps = {s.name for s in self._specs}
        for name in runtime_params:
            if name not in known_steps:
                _logger.warning(
                    "runtime_params has unknown step name: %s (ignored)", name
                )

        for spec in self._specs:
            fn = _STEP_REGISTRY.get(spec.name)
            if fn is None:
                raise ValueError(f"unknown step: {spec.name}")
            params = dict(spec.params)
            overridden: list[str] = []
            if spec.name in runtime_params:
                for k, v in runtime_params[spec.name].items():
                    if k in params:
                        overridden.append(k)
                    params[k] = v
            before = len(current)
            result = fn(current, **params)
            note = result.note
            if overridden:
                note = f"{note}; runtime-overridden keys: {sorted(overridden)}"
            history.append(
                StepHistory(
                    name=spec.name,
                    adr=spec.adr,
                    excluded_rows=result.excluded_rows,
                    rows_before=before,
                    rows_after=len(result.df),
                    note=note,
                )
            )
            current = result.df
        return current, CleaningHistory(
            input_rows=input_rows,
            output_rows=len(current),
            steps=history,
        )
