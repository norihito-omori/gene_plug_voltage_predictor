"""クリーニングステップ(純粋関数)。各ステップは DataFrame -> StepResult を返す。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StepResult:
    df: pd.DataFrame
    excluded_rows: int
    note: str


def exclude_locations(
    df: pd.DataFrame,
    *,
    location_col: str,
    excluded: Iterable[int | str],
) -> StepResult:
    """指定機場を除外する。ADR-001 L-01 (8950 除外) 等で使用。"""
    excluded_set = set(excluded)
    mask = ~df[location_col].isin(excluded_set)
    out = df.loc[mask].reset_index(drop=True)
    note = f"excluded={sorted(excluded_set)}" if excluded_set else "no locations excluded"
    return StepResult(df=out, excluded_rows=len(df) - len(out), note=note)


def filter_cumulative_runtime(
    df: pd.DataFrame,
    *,
    runtime_col: str,
    min_hours: float,
) -> StepResult:
    """累積運転時間が閾値未満の行を除外。ADR-001 L-02 (500h 点検影響) で使用。"""
    mask = df[runtime_col] >= min_hours
    out = df.loc[mask].reset_index(drop=True)
    return StepResult(
        df=out,
        excluded_rows=len(df) - len(out),
        note=f"keep rows where {runtime_col} >= {min_hours}",
    )


def filter_by_rated_power_ratio(
    df: pd.DataFrame,
    *,
    power_col: str,
    rated_kw: int,
    ratio: float,
) -> StepResult:
    """定格電力比率以上の行のみ残す。ADR-001 L-04 (80% 以上) で使用。"""
    threshold = rated_kw * ratio
    mask = df[power_col] >= threshold
    out = df.loc[mask].reset_index(drop=True)
    return StepResult(
        df=out,
        excluded_rows=len(df) - len(out),
        note=f"keep rows where {power_col} >= {threshold} (rated {rated_kw}kW x {ratio})",
    )
