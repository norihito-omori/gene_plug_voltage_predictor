"""A-5c で記録漏れ疑いとされた 5 件の詳細数値を抽出する。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_exchange_precision import (
    _load_location_daily_max, _EP370G_START_DATETIMES,
)

_INPUT_DIR = Path("E:/gene/input/EP370G_orig")
_LEGACY_CSV = Path(
    "D:/code_commit/EP-VoltPredictor/ep_voltpredictor/exchange_timings_summary4.csv"
)

_FP_SUSPECTS: list[tuple[str, str]] = [
    ("5630", "2025-12-22"),
    ("9110", "2025-05-31"),
    ("9110", "2025-09-04"),
    ("9240", "2022-09-22"),
    ("9720", "2025-05-19"),
]


def main() -> None:
    legacy = pd.read_csv(_LEGACY_CSV, encoding="utf-8-sig")
    legacy["dailygraphpt_ptdatetime"] = pd.to_datetime(legacy["dailygraphpt_ptdatetime"])
    legacy["target_no"] = legacy["file_name"].str.replace(".csv", "", regex=False)

    for target_no, fp_str in _FP_SUSPECTS:
        fp = pd.Timestamp(fp_str)
        daily = _load_location_daily_max(_INPUT_DIR / f"{target_no}.csv")
        cutoff = _EP370G_START_DATETIMES.get(target_no)
        if cutoff is not None:
            daily = daily[daily["date"] >= pd.Timestamp(cutoff).normalize()]

        before = daily[(daily["date"] >= fp - pd.Timedelta(days=10))
                       & (daily["date"] <= fp - pd.Timedelta(days=1))]
        after = daily[(daily["date"] >= fp + pd.Timedelta(days=1))
                      & (daily["date"] <= fp + pd.Timedelta(days=10))]

        plug_cols = [f"plug_{i}" for i in range(1, 7)]
        shifts: list[float] = []
        b_meds: list[str] = []
        a_meds: list[str] = []
        for col in plug_cols:
            b = before[col].dropna()
            a = after[col].dropna()
            if len(b) < 3 or len(a) < 3:
                b_meds.append("-")
                a_meds.append("-")
                shifts.append(float("nan"))
                continue
            bm = float(b.median())
            am = float(a.median())
            b_meds.append(f"{bm:.1f}")
            a_meds.append(f"{am:.1f}")
            shifts.append(am - bm)

        legacy_rows = legacy[(legacy["target_no"] == target_no)
                             & (abs(legacy["dailygraphpt_ptdatetime"] - fp).dt.days <= 60)]
        legacy_dates = legacy_rows["dailygraphpt_ptdatetime"].dt.strftime("%Y-%m-%d").tolist()

        max_shift = max((abs(s) for s in shifts if pd.notna(s)), default=0.0)
        print(f"\n== {target_no}  FP {fp_str} ==")
        print(f"  before median: {', '.join(b_meds)} kV")
        print(f"  after  median: {', '.join(a_meds)} kV")
        shift_strs = [f"{s:+.1f}" if pd.notna(s) else "nan" for s in shifts]
        print(f"  shift         : {', '.join(shift_strs)} kV (max|Δ|={max_shift:.1f})")
        print(f"  legacy ±60d   : {legacy_dates if legacy_dates else '(none)'}")


if __name__ == "__main__":
    main()
