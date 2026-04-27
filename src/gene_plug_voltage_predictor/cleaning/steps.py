"""クリーニングステップ(純粋関数)。各ステップは DataFrame -> StepResult を返す。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
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


def assign_generation(
    df: pd.DataFrame,
    *,
    id_col: str,
    datetime_col: str,
    events_by_location: Mapping[str, Sequence[pd.Timestamp]],
    gen_col: str = "gen_no",
) -> StepResult:
    """各行に gen_no (int, 0-origin) を付与する（ADR-014 §C-5 基本形）。

    比較は d = 行の datetime_col を normalize した日付で行う:
      d < events[0]                   → gen_no = 0
      events[k-1] <= d < events[k]    → gen_no = k
      d >= events[last]               → gen_no = len(events)

    events_by_location に無い機場は全行 gen_no = 0。
    events は内部で normalize + sort + unique 化する。
    """
    for c in (id_col, datetime_col):
        if c not in df.columns:
            raise ValueError(f"missing required column: {c}")

    dt = pd.to_datetime(df[datetime_col], errors="coerce").dt.normalize()
    ids = df[id_col].astype(str)

    cleaned_events: dict[str, list[pd.Timestamp]] = {}
    for loc, evs in events_by_location.items():
        norm = sorted({pd.Timestamp(e).normalize() for e in evs})
        cleaned_events[str(loc)] = norm

    gen = pd.Series(0, index=df.index, dtype="int64")
    known_ids = set(cleaned_events.keys())
    missing_ids = sorted(set(ids.unique()) - known_ids)

    for loc, evs in cleaned_events.items():
        if not evs:
            continue
        loc_mask = ids == loc
        if not loc_mask.any():
            continue
        d_sub = dt[loc_mask]
        values = pd.Series(evs).values
        # Boundary rule (ADR-014 §C-5): d.normalize() >= event_date → new generation.
        # (values <= d).sum() counts the event day itself in the new gen; using > /
        # side="right" would shift the boundary by one day and misclassify event-day rows.
        gens = pd.Series(
            [int((values <= d).sum()) for d in d_sub.values],
            index=d_sub.index,
        )
        gen.loc[loc_mask] = gens

    out = df.copy()
    out[gen_col] = gen.values
    note_parts = [f"assigned gen_no for {len(cleaned_events)} locations"]
    if missing_ids:
        note_parts.append(
            f"{len(missing_ids)} locations had no events (gen=0 全行)"
        )
    return StepResult(df=out, excluded_rows=0, note="; ".join(note_parts))
