"""summary5 基準で v5 残 FP のみを ±30 日 zoom プロット化する。

summary4 ベースの `plot_fp_zoom_only.py` から legacy CSV を summary5 に差し替え、
出力先を `plots/fp_zoom_v5/` に分離。キャッシュ衝突を避ける。

対象: ADR-012 カットオフ済、t=5, q=3, w=10, merge=7 で検出し、summary5 legacy に
tolerance=10 でマッチしない検出イベント（= 真の FP 候補）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_exchange_precision import (
    _load_location_daily_max, _scan_detect_days, _merge_consecutive,
    _EP370G_START_DATETIMES, _MERGE_WINDOW_DAYS,
)
from plot_fp_analysis import _classify_detections

_INPUT_DIR = Path("E:/gene/input/EP370G_orig")
_LEGACY_CSV = Path(
    "D:/code_commit/EP-VoltPredictor/ep_voltpredictor/exchange_timings_summary5.csv"
)
_OUT_DIR = Path(__file__).parent / "plots" / "fp_zoom_v5"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")

_THRESHOLD = 5.0
_QUORUM = 3
_WINDOW = 10
_MATCH_TOLERANCE = 10  # A-5c 運用値


def plot_fp(target_no: str, fp_date: pd.Timestamp, legacy: list[pd.Timestamp],
            daily_max: pd.DataFrame, zoom_days: int = 30) -> Path:
    start = fp_date - pd.Timedelta(days=zoom_days)
    end = fp_date + pd.Timedelta(days=zoom_days)
    w = daily_max[(daily_max["date"] >= start) & (daily_max["date"] <= end)]
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(1, 7):
        s = w[f"plug_{i}"].dropna()
        if s.empty:
            continue
        ax.plot(w.loc[s.index, "date"], s.values, marker="o", markersize=4,
                linewidth=1.2, color=_COLORS[i-1], label=f"P{i}")
    ax.axvline(fp_date, color="tab:red", linewidth=2.0,
               label=f"FP {fp_date:%Y-%m-%d}")
    for ev in legacy:
        if abs((ev - fp_date).days) <= zoom_days:
            ax.axvline(ev, color="black", linestyle="--", linewidth=1.4,
                       label=f"legacy {ev:%Y-%m-%d}")
    ax.set_title(f"target_no={target_no}  FP {fp_date:%Y-%m-%d}  "
                 f"±{zoom_days}d zoom  (v5 / tol={_MATCH_TOLERANCE})")
    ax.set_ylabel("daily max voltage [kV]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out = _OUT_DIR / f"zoom_{target_no}_{fp_date:%Y%m%d}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(target_nos: list[str]) -> int:
    legacy = pd.read_csv(_LEGACY_CSV, encoding="utf-8-sig")
    legacy["dailygraphpt_ptdatetime"] = pd.to_datetime(
        legacy["dailygraphpt_ptdatetime"]
    )
    legacy["target_no"] = legacy["file_name"].str.replace(".csv", "", regex=False)

    total_fp = 0
    for t in target_nos:
        csv_path = _INPUT_DIR / f"{t}.csv"
        if not csv_path.exists():
            print(f"[{t}] SKIP: no CSV", file=sys.stderr)
            continue
        daily = _load_location_daily_max(csv_path)
        cutoff = _EP370G_START_DATETIMES.get(t)
        if cutoff is not None:
            daily = daily[daily["date"] >= pd.Timestamp(cutoff).normalize()].reset_index(drop=True)
        ev_legacy = [pd.Timestamp(x).normalize()
                     for x in legacy[legacy["target_no"] == t]["dailygraphpt_ptdatetime"]]
        det_days = _scan_detect_days(
            daily, threshold=_THRESHOLD, quorum=_QUORUM, window_days=_WINDOW,
        )
        det_events = _merge_consecutive(det_days, _MERGE_WINDOW_DAYS)
        _, fp_events = _classify_detections(det_events, ev_legacy, _MATCH_TOLERANCE)
        for fp in fp_events:
            out = plot_fp(t, fp, ev_legacy, daily)
            print(f"[{t}] FP {fp:%Y-%m-%d} -> {out}", file=sys.stderr)
            total_fp += 1
    print(f"\nTotal FP plotted: {total_fp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    # v5 FP at tol=10: 9110, 9111, 9520 （9610 2024-08-25 は tol=10 で TP 化済）
    target_nos = sys.argv[1:] or ["9110", "9111", "9520", "9610"]
    sys.exit(main(target_nos))
