"""F1 回帰確認スクリプト。

cleaning.exchange.detect_exchange_events を使って全 EP370G 機場を評価し、
F1=0.924 (summary5, tolerance=7) を再現する。

Usage:
    cd E:/projects/gene_plug_voltage_predictor
    python migration/evaluate_with_new_detect.py
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

# プロジェクト src を sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gene_plug_voltage_predictor.cleaning.exchange import detect_exchange_events
from gene_plug_voltage_predictor.constants import (
    EP370G_EXCLUDED_LOCATIONS,
    EP370G_START_DATETIMES,
    EXCHANGE_DETECTION_DEFAULTS,
)

VOLTAGE_COLS = [f"要求電圧_{i}" for i in range(1, 7)]
INPUT_DIR = Path("E:/gene/input/EP370G_orig")
SUMMARY5_PATH = Path("D:/code_commit/EP-VoltPredictor/ep_voltpredictor/exchange_timings_summary5.csv")
MATCH_TOLERANCE_DAYS = 7


def _load_legacy(path: Path) -> dict[str, list[pd.Timestamp]]:
    df = pd.read_csv(path)
    result: dict[str, list[pd.Timestamp]] = {}
    for _, row in df.iterrows():
        loc = str(row["file_name"]).replace(".csv", "").strip()
        dt = pd.Timestamp(str(row["dailygraphpt_ptdatetime"]).strip())
        result.setdefault(loc, []).append(dt)
    return result


def _match(detected: list[pd.Timestamp], legacy: list[pd.Timestamp], tol: int) -> tuple[int, int, int]:
    """(TP, FP, FN) を返す。貪欲マッチング。"""
    tol_td = timedelta(days=tol)
    used_legacy = set()
    tp = 0
    for det in detected:
        for i, leg in enumerate(legacy):
            if i not in used_legacy and abs(det - leg) <= tol_td:
                tp += 1
                used_legacy.add(i)
                break
    fp = len(detected) - tp
    fn = len(legacy) - len(used_legacy)
    return tp, fp, fn


def main() -> None:
    legacy_by_loc = _load_legacy(SUMMARY5_PATH)
    csvs = sorted(INPUT_DIR.glob("*.csv"))
    excluded = set(EP370G_EXCLUDED_LOCATIONS)

    total_tp = total_fp = total_fn = 0
    rows = []

    for csv in csvs:
        loc = csv.stem
        if loc in excluded:
            continue
        df = pd.read_csv(csv, encoding="utf-8-sig")
        # CANONICAL_RENAMES (要求電圧N → 要求電圧_N)
        df = df.rename(columns={f"要求電圧{i}": f"要求電圧_{i}" for i in range(1, 7)})

        cutoff_dt = EP370G_START_DATETIMES.get(loc)
        cutoff_ts = pd.Timestamp(cutoff_dt) if cutoff_dt else None

        try:
            detected = detect_exchange_events(
                df,
                voltage_cols=VOLTAGE_COLS,
                cutoff=cutoff_ts,
                **EXCHANGE_DETECTION_DEFAULTS,
            )
        except ValueError as e:
            print(f"WARN {loc}: {e}")
            detected = []

        legacy = legacy_by_loc.get(loc)
        # summary5 に登録のない機場は ground truth がないため FP/FN カウント対象外
        if legacy is None:
            rows.append({"loc": loc, "det": len(detected), "leg": "-", "TP": "-", "FP": "-", "FN": "-",
                         "P": "-", "R": "-", "note": "no legacy"})
            continue
        tp, fp, fn = _match(detected, legacy, MATCH_TOLERANCE_DAYS)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        rows.append({"loc": loc, "det": len(detected), "leg": len(legacy), "TP": tp, "FP": fp, "FN": fn,
                     "P": round(p, 3), "R": round(r, 3), "note": ""})

    p_total = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r_total = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p_total * r_total / (p_total + r_total) if (p_total + r_total) > 0 else 0

    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"Precision={p_total:.3f}  Recall={r_total:.3f}  F1={f1:.3f}")
    print(f"(tolerance={MATCH_TOLERANCE_DAYS}d, summary5)")


if __name__ == "__main__":
    main()
