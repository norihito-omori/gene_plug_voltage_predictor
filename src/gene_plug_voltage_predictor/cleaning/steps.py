"""クリーニングステップ(純粋関数)。各ステップは DataFrame -> StepResult を返す。"""

from __future__ import annotations

import re
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
    """指定機場を除外する。ADR-001 L-01 (8950 除外) 等で使用。

    See exclude_location_plug for composite location×plug ID exclusion (ADR-001 L-03).
    """
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


def melt_voltage_columns(
    df: pd.DataFrame,
    *,
    location_no_col: str,
    voltage_cols: Iterable[str],
) -> StepResult:
    """要求電圧の 6 列を縦持ち（長形式）に展開する。ADR-009。

    出力は元の非電圧列 + 以下を持つ：
      - `要求電圧`: 値
      - `プラグNo`: 列名末尾の数字 1..6
      - `管理No_プラグNo`: f"{location_no}_{plug_no}"（文字列）

    変換ステップのため excluded_rows は常に 0。
    出力行数 = 入力行数 × len(voltage_cols)。
    """
    voltage_cols_list = list(voltage_cols)
    if not voltage_cols_list:
        raise ValueError("voltage_cols must be non-empty")
    bad = [c for c in voltage_cols_list if not re.search(r"\d+$", c)]
    if bad:
        raise ValueError(f"voltage_cols must end in a digit: {bad}")
    id_vars = [c for c in df.columns if c not in voltage_cols_list]
    melted = df.melt(
        id_vars=id_vars,
        value_vars=voltage_cols_list,
        var_name="_voltage_col",
        value_name="要求電圧",
    )
    # 列名末尾の数字をプラグ番号として抽出（"要求電圧_3" → 3, "要求電圧3" → 3）
    plug_no = (
        melted["_voltage_col"].str.extract(r"(\d+)$", expand=False).astype(int)
    )
    melted["プラグNo"] = plug_no
    melted["管理No_プラグNo"] = (
        melted[location_no_col].astype(str) + "_" + plug_no.astype(str)
    )
    out = melted.drop(columns=["_voltage_col"]).reset_index(drop=True)
    return StepResult(
        df=out,
        excluded_rows=0,
        note=f"melted {len(voltage_cols_list)} voltage columns into long format",
    )


def exclude_location_plug(
    df: pd.DataFrame,
    *,
    id_col: str,
    excluded: Iterable[str],
) -> StepResult:
    """指定の `管理No_プラグNo` を除外する。ADR-001 L-03（9221 プラグ 4 除外）等で使用。

    See exclude_locations for whole-location exclusion (ADR-001 L-01).
    """
    excluded_set = set(excluded)
    mask = ~df[id_col].isin(excluded_set)
    out = df.loc[mask].reset_index(drop=True)
    note = (
        f"excluded id={sorted(excluded_set)}" if excluded_set else "no ids excluded"
    )
    return StepResult(df=out, excluded_rows=len(df) - len(out), note=note)
