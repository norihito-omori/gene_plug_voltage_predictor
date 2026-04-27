"""9580 / 9581 の要求電圧 > 50 kV の期間と分布を調べる。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_INPUT_DIR = Path("E:/gene/input/EP370G_orig")
_DT_COL = "dailygraphpt_ptdatetime"
_POWER_COL = "発電機電力"
_VOLTAGE_COLS_U = tuple(f"要求電圧_{i}" for i in range(1, 7))
_VOLTAGE_COLS_N = tuple(f"要求電圧{i}" for i in range(1, 7))


def _resolve_voltage_cols(path: Path) -> tuple[str, ...]:
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    avail = set(head.columns)
    if set(_VOLTAGE_COLS_N).issubset(avail):
        return _VOLTAGE_COLS_N
    if set(_VOLTAGE_COLS_U).issubset(avail):
        return _VOLTAGE_COLS_U
    raise ValueError(f"No complete 要求電圧 column set in {path.name}")


def summarize(target_no: str) -> None:
    path = _INPUT_DIR / f"{target_no}.csv"
    voltage_cols = _resolve_voltage_cols(path)
    df = pd.read_csv(path, encoding="utf-8-sig",
                     usecols=[_DT_COL, _POWER_COL, *voltage_cols])
    df[_DT_COL] = pd.to_datetime(df[_DT_COL], errors="coerce")
    df = df.dropna(subset=[_DT_COL])
    print(f"\n=== {target_no} ===")
    print(f"raw rows: {len(df)}")
    print(f"min datetime: {df[_DT_COL].min()}")
    print(f"max datetime: {df[_DT_COL].max()}")

    running = df[df[_POWER_COL] > 0]
    print(f"running rows: {len(running)}")

    for col in voltage_cols:
        mask = running[col] > 50
        if not mask.any():
            continue
        sub = running[mask]
        print(f"\n  [{col}] n={len(sub)}  "
              f"range=[{sub[col].min():.1f}, {sub[col].max():.1f}]")
        print(f"    period: {sub[_DT_COL].min()} → {sub[_DT_COL].max()}")
        print(f"    distinct values (top 10):")
        print(sub[col].value_counts().head(10).to_string())
        print(f"    yearly counts:")
        print(sub[_DT_COL].dt.year.value_counts().sort_index().to_string())


if __name__ == "__main__":
    for t in ("9580", "9581"):
        summarize(t)
