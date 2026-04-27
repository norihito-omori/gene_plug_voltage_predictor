"""ADR-012 カットオフ適用後、要求電圧 > 50 kV の行が残っていないかをチェック。

スパイク行（要求電圧が 50 を大幅に超える異常値）は ADR-011 相当の既知
センサー異常か、未発見の外れ値。カットオフで取り切れているかを確認する。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd

_DT_COL: Final[str] = "dailygraphpt_ptdatetime"
_POWER_COL: Final[str] = "発電機電力"
_VOLTAGE_COLS_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧_{i}" for i in range(1, 7))
_VOLTAGE_COLS_NO_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧{i}" for i in range(1, 7))

_VOLTAGE_LIMIT: Final[float] = 50.0

_EP370G_START_DATETIMES: Final[dict[str, datetime]] = {
    "5630": datetime(2022, 6, 26, 0, 30),
    "8950": datetime(2021, 6, 12, 20, 30),
    "9290": datetime(2022, 7, 21, 10, 0),
    "9380": datetime(2023, 5, 31, 0, 30),
    "9381": datetime(2023, 5, 31, 14, 0),
    "9580": datetime(2023, 7, 21, 0, 30),
    "9581": datetime(2023, 7, 21, 0, 30),
    "9610": datetime(2024, 4, 2, 19, 30),
    "9611": datetime(2024, 4, 2, 20, 0),
    "9690": datetime(2024, 3, 19, 8, 30),
}

_INPUT_DIR: Final[Path] = Path("E:/gene/input/EP370G_orig")


def _resolve_voltage_cols(path: Path) -> tuple[str, ...]:
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    available = set(head.columns)
    if set(_VOLTAGE_COLS_NO_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_NO_UNDERSCORE
    if set(_VOLTAGE_COLS_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_UNDERSCORE
    raise ValueError(f"No complete 要求電圧 column set found in {path.name}")


def _check_location(path: Path) -> dict:
    target_no = path.stem
    cutoff = _EP370G_START_DATETIMES.get(target_no)
    voltage_cols = _resolve_voltage_cols(path)
    df = pd.read_csv(
        path, encoding="utf-8-sig",
        usecols=[_DT_COL, _POWER_COL, *voltage_cols],
    )
    df[_DT_COL] = pd.to_datetime(df[_DT_COL], errors="coerce")
    df = df.dropna(subset=[_DT_COL])
    total_before_cutoff = len(df)

    if cutoff is not None:
        df = df[df[_DT_COL] >= pd.Timestamp(cutoff)].reset_index(drop=True)
    total_after_cutoff = len(df)

    running = df[df[_POWER_COL] > 0]
    total_running = len(running)

    over = running[(running[list(voltage_cols)] > _VOLTAGE_LIMIT).any(axis=1)]
    n_over = len(over)

    per_plug_counts: dict[str, int] = {}
    per_plug_max: dict[str, float] = {}
    for i, col in enumerate(voltage_cols, 1):
        mask = running[col] > _VOLTAGE_LIMIT
        per_plug_counts[f"plug_{i}"] = int(mask.sum())
        per_plug_max[f"plug_{i}"] = float(running.loc[mask, col].max()) if mask.any() else 0.0

    first_dt = over[_DT_COL].min() if n_over else None
    last_dt = over[_DT_COL].max() if n_over else None

    return {
        "target_no": target_no,
        "cutoff": cutoff.isoformat() if cutoff else "none",
        "rows_raw": total_before_cutoff,
        "rows_after_cutoff": total_after_cutoff,
        "rows_running": total_running,
        "rows_voltage_over_50": n_over,
        "over_first_datetime": first_dt,
        "over_last_datetime": last_dt,
        "per_plug_counts": per_plug_counts,
        "per_plug_max": per_plug_max,
    }


def main() -> int:
    if not _INPUT_DIR.exists():
        print(f"input dir not found: {_INPUT_DIR}", file=sys.stderr)
        return 1

    csvs = sorted(_INPUT_DIR.glob("*.csv"))
    print(f"Checking {len(csvs)} EP370G CSVs for 要求電圧 > {_VOLTAGE_LIMIT} kV "
          f"after ADR-012 cutoff", file=sys.stderr)
    print()
    results: list[dict] = []
    for p in csvs:
        try:
            r = _check_location(p)
        except Exception as exc:
            print(f"[{p.stem}] ERROR: {exc}", file=sys.stderr)
            continue
        results.append(r)
        flag = "!" if r["rows_voltage_over_50"] else " "
        print(f"{flag} [{r['target_no']}] cutoff={r['cutoff'][:10] if r['cutoff']!='none' else 'none':10s}  "
              f"running={r['rows_running']:>7d}  over50={r['rows_voltage_over_50']:>5d}  "
              f"first={str(r['over_first_datetime'])[:19] if r['over_first_datetime'] else '-':19s}  "
              f"last={str(r['over_last_datetime'])[:19] if r['over_last_datetime'] else '-':19s}")
        if r["rows_voltage_over_50"]:
            for k, v in r["per_plug_counts"].items():
                if v:
                    print(f"        {k}: n={v}  max={r['per_plug_max'][k]:.1f} kV")

    total_over = sum(r["rows_voltage_over_50"] for r in results)
    total_running = sum(r["rows_running"] for r in results)
    print()
    print(f"TOTAL: {total_over} rows with voltage > {_VOLTAGE_LIMIT} kV "
          f"across {total_running} running rows "
          f"({total_over/total_running*100:.3f}%)" if total_running else "TOTAL: 0 running rows")

    # CSV 出力
    out_rows: list[dict] = []
    for r in results:
        base = {k: r[k] for k in ["target_no", "cutoff", "rows_raw",
                                   "rows_after_cutoff", "rows_running",
                                   "rows_voltage_over_50",
                                   "over_first_datetime", "over_last_datetime"]}
        for k, v in r["per_plug_counts"].items():
            base[f"over50_{k}"] = v
            base[f"max_over50_{k}_kV"] = r["per_plug_max"][k]
        out_rows.append(base)
    df_out = pd.DataFrame(out_rows)
    out_path = Path(__file__).parent / "voltage_over_50_summary.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nWrote: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
