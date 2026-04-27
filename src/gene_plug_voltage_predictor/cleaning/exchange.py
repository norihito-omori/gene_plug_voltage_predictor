"""プラグ交換検出（ADR-014、median-diff 方式）。pipeline 外の純粋関数。

migration/evaluate_exchange_precision.py からの移植版。
アルゴリズム: 前半窓 median vs 後半窓 median 差 > threshold のプラグが quorum 以上
で検出日とし、merge_window_days 以内の連続検出は 1 イベントにまとめる。
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

_logger = logging.getLogger(__name__)


def _daily_max_running(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    power_col: str,
    voltage_cols: Sequence[str],
) -> pd.DataFrame:
    """発電機電力 > 0 の行に絞り、日別プラグ別の max voltage を返す。

    戻り値: columns = [date, plug_1, ..., plug_N]（N = len(voltage_cols)）。
    """
    missing = [c for c in (datetime_col, power_col, *voltage_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column: {missing}")

    work = df[[datetime_col, power_col, *voltage_cols]].copy()
    work[datetime_col] = pd.to_datetime(work[datetime_col], errors="coerce")
    work = work.dropna(subset=[datetime_col])
    running = work[work[power_col] > 0]
    if running.empty:
        plug_names = [f"plug_{i}" for i in range(1, len(voltage_cols) + 1)]
        return pd.DataFrame(columns=["date", *plug_names])

    running = running.copy()
    running["date"] = running[datetime_col].dt.normalize()
    out = pd.DataFrame({"date": sorted(running["date"].unique())})
    for i, col in enumerate(voltage_cols, 1):
        valid = running[col] > 0
        daily_max = running.loc[valid].groupby("date")[col].max().rename(f"plug_{i}")
        out = out.merge(daily_max, on="date", how="left")
    return out.reset_index(drop=True)


def _scan_detect_days(
    daily_max: pd.DataFrame,
    *,
    threshold: float,
    quorum: int,
    window_days: int,
    min_days_each_side: int,
    n_plugs: int,
) -> list[pd.Timestamp]:
    if daily_max.empty:
        return []
    plug_cols = [f"plug_{i}" for i in range(1, n_plugs + 1)]
    detected: list[pd.Timestamp] = []
    dates = daily_max["date"].to_numpy()
    for d in dates:
        event_date = pd.Timestamp(d)
        before = daily_max[
            (daily_max["date"] >= event_date - pd.Timedelta(days=window_days))
            & (daily_max["date"] <= event_date - pd.Timedelta(days=1))
        ]
        after = daily_max[
            (daily_max["date"] >= event_date + pd.Timedelta(days=1))
            & (daily_max["date"] <= event_date + pd.Timedelta(days=window_days))
        ]
        drop_count = 0
        valid_plugs = 0
        for col in plug_cols:
            b = before[col].dropna()
            a = after[col].dropna()
            if len(b) < min_days_each_side or len(a) < min_days_each_side:
                continue
            valid_plugs += 1
            if abs(float(a.median()) - float(b.median())) > threshold:
                drop_count += 1
        if valid_plugs == 0:
            continue
        if drop_count >= quorum:
            detected.append(event_date)
    return detected


def _merge_consecutive(
    detected: list[pd.Timestamp], merge_window: int
) -> list[pd.Timestamp]:
    if not detected:
        return []
    sorted_days = sorted(detected)
    groups: list[list[pd.Timestamp]] = [[sorted_days[0]]]
    for d in sorted_days[1:]:
        if (d - groups[-1][-1]).days <= merge_window:
            groups[-1].append(d)
        else:
            groups.append([d])
    return [g[len(g) // 2] for g in groups]


def detect_exchange_events(
    df: pd.DataFrame,
    *,
    datetime_col: str = "dailygraphpt_ptdatetime",
    power_col: str = "発電機電力",
    voltage_cols: Sequence[str],
    cutoff: pd.Timestamp | None = None,
    voltage_drop_threshold: float = 5.0,
    plug_quorum: int = 3,
    window_days: int = 10,
    min_days_each_side: int = 3,
    merge_window_days: int = 7,
) -> list[pd.Timestamp]:
    """1 機場の生データ（横持ち）から交換日時系列を検出する（ADR-014 §C-3）。

    戻り値: ソート済み交換代表日の list（normalize 済み）。
    running 行がゼロ、または cutoff 後に運転日が無い場合は空 list。
    """
    vcols = list(voltage_cols)
    if not vcols:
        raise ValueError("voltage_cols must be non-empty")
    if plug_quorum < 1:
        raise ValueError(f"plug_quorum must be >= 1 (got {plug_quorum})")
    if plug_quorum > len(vcols):
        raise ValueError(
            f"plug_quorum ({plug_quorum}) exceeds voltage column count ({len(vcols)})"
        )
    if min_days_each_side < 1:
        raise ValueError(f"min_days_each_side must be >= 1 (got {min_days_each_side})")

    daily = _daily_max_running(
        df,
        datetime_col=datetime_col,
        power_col=power_col,
        voltage_cols=vcols,
    )
    if cutoff is not None:
        cutoff_norm = pd.Timestamp(cutoff).normalize()
        daily = daily[daily["date"] >= cutoff_norm].reset_index(drop=True)
    detected = _scan_detect_days(
        daily,
        threshold=voltage_drop_threshold,
        quorum=plug_quorum,
        window_days=window_days,
        min_days_each_side=min_days_each_side,
        n_plugs=len(vcols),
    )
    return _merge_consecutive(detected, merge_window_days)
