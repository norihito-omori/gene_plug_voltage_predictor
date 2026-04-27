"""Plan A-5c: FP の真偽を目視判別するプロット。

`evaluate_exchange_precision.py` の F1 最大パラメータで検出した日を、機場ごとの
時系列プロット上に重ねて表示する。

- 黒破線: legacy イベント（ground truth）
- 緑実線: 検出イベントのうち legacy とマッチ（TP）
- 赤実線: 検出イベントのうち legacy と非マッチ（FP、精査対象）

目視で「FP が本当に電圧急落を起こしているか」「legacy 記録漏れの可能性があるか」
を判断する。FP 付近のプラグ波形を確認し、以下を判別:

1. プラグ交換レベルの level shift が確認できる → legacy 記録漏れ（FP ではなく TP）
2. 小さい変動しかない or センサーノイズ → 真の FP（アルゴリズムの偽陽性）
3. level shift 有り + 1 プラグのみ → 単独プラグ故障イベント（legacy も落とした可能性）

出力: `migration/plots/fp_analysis/<target_no>.png`
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd

from evaluate_exchange_precision import (
    _load_location_daily_max,
    _match_to_legacy,
    _merge_consecutive,
    _scan_detect_days,
    _MATCH_TOLERANCE_DAYS,
    _MERGE_WINDOW_DAYS,
    _EP370G_START_DATETIMES,
)

# F1 最大パラメータ（Plan A-4 結果、`precision-grid-results.md`）
_THRESHOLD: Final[float] = 5.0
_QUORUM: Final[int] = 3
_WINDOW_DAYS: Final[int] = 10

_DT_COL: Final[str] = "dailygraphpt_ptdatetime"

_LEGACY_CSV: Final[Path] = Path(
    "D:/code_commit/EP-VoltPredictor/ep_voltpredictor/exchange_timings_summary4.csv"
)
_INPUT_DIR: Final[Path] = Path("E:/gene/input/EP370G_orig")
_OUT_DIR: Final[Path] = Path(__file__).parent / "plots" / "fp_analysis_v3"

_PLUG_COLORS: Final[tuple[str, ...]] = (
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
)


def _cutoff_status(target_no: str, event_dt: datetime) -> str:
    cutoff = _EP370G_START_DATETIMES.get(target_no)
    if cutoff is None:
        return "no_cutoff"
    return "after_cutoff" if event_dt >= cutoff else "before_cutoff"


def _classify_detections(
    detected: list[pd.Timestamp],
    legacy: list[pd.Timestamp],
    tolerance: int,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """検出イベントを (TP, FP) に分類。同じマッチング方式だが結果セットを返す版。"""
    legacy_sorted = sorted(legacy)
    matched_legacy: set[int] = set()
    tp: list[pd.Timestamp] = []
    fp: list[pd.Timestamp] = []
    for d in sorted(detected):
        best_idx = -1
        best_dist = tolerance + 1
        for i, e in enumerate(legacy_sorted):
            if i in matched_legacy:
                continue
            dist = abs((d - e).days)
            if dist <= tolerance and dist < best_dist:
                best_idx = i
                best_dist = dist
        if best_idx >= 0:
            matched_legacy.add(best_idx)
            tp.append(d)
        else:
            fp.append(d)
    return tp, fp


def _plot_full_timeline(ax: plt.Axes, daily_max: pd.DataFrame,
                        legacy_events: list[pd.Timestamp],
                        tp_events: list[pd.Timestamp],
                        fp_events: list[pd.Timestamp]) -> None:
    plug_cols = [f"plug_{i}" for i in range(1, 7)]
    for i, col in enumerate(plug_cols):
        s = daily_max[col].dropna()
        if s.empty:
            continue
        plug_dates = daily_max.loc[s.index, "date"]
        ax.plot(plug_dates, s.values, marker=".", markersize=2, linewidth=0.6,
                color=_PLUG_COLORS[i], label=f"Plug {i+1}", alpha=0.85)
    for ev in legacy_events:
        ax.axvline(ev, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    for ev in tp_events:
        ax.axvline(ev, color="tab:green", linestyle="-", linewidth=1.2, alpha=0.6)
    for ev in fp_events:
        ax.axvline(ev, color="tab:red", linestyle="-", linewidth=1.6, alpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("daily max voltage [kV]")


def _plot_fp_zoom(ax: plt.Axes, daily_max: pd.DataFrame, fp_date: pd.Timestamp,
                  legacy_events: list[pd.Timestamp], zoom_days: int = 30) -> None:
    start = fp_date - pd.Timedelta(days=zoom_days)
    end = fp_date + pd.Timedelta(days=zoom_days)
    window = daily_max[(daily_max["date"] >= start) & (daily_max["date"] <= end)]
    plug_cols = [f"plug_{i}" for i in range(1, 7)]
    for i, col in enumerate(plug_cols):
        s = window[col].dropna()
        if s.empty:
            continue
        plug_dates = window.loc[s.index, "date"]
        ax.plot(plug_dates, s.values, marker="o", markersize=3, linewidth=1.0,
                color=_PLUG_COLORS[i], label=f"P{i+1}")
    ax.axvline(fp_date, color="tab:red", linestyle="-", linewidth=1.6, alpha=0.85)
    # 近傍の legacy があれば黒破線で
    for ev in legacy_events:
        if abs((ev - fp_date).days) <= zoom_days:
            ax.axvline(ev, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_title(f"FP {fp_date:%Y-%m-%d}", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=7)


def _plot_location(target_no: str, legacy_events: list[pd.Timestamp], location_path: Path) -> Path:
    daily_max = _load_location_daily_max(location_path)
    cutoff = _EP370G_START_DATETIMES.get(target_no)
    if cutoff is not None:
        cutoff_ts = pd.Timestamp(cutoff).normalize()
        daily_max = daily_max[daily_max["date"] >= cutoff_ts].reset_index(drop=True)

    detected_days = _scan_detect_days(
        daily_max, threshold=_THRESHOLD, quorum=_QUORUM, window_days=_WINDOW_DAYS,
    )
    detected_events = _merge_consecutive(detected_days, _MERGE_WINDOW_DAYS)
    tp_events, fp_events = _classify_detections(
        detected_events, legacy_events, _MATCH_TOLERANCE_DAYS,
    )

    # レイアウト: 上段 1 行 = 全期間、下段 = FP 近傍の zoom（2 列）
    n_fp = len(fp_events)
    n_zoom_rows = (n_fp + 1) // 2 if n_fp else 0
    total_rows = 1 + n_zoom_rows
    fig = plt.figure(figsize=(18, 4 + 3.5 * n_zoom_rows))
    gs = fig.add_gridspec(total_rows, 2)

    ax_full = fig.add_subplot(gs[0, :])
    _plot_full_timeline(ax_full, daily_max, legacy_events, tp_events, fp_events)

    # 全期間プロットの凡例
    from matplotlib.lines import Line2D
    h_plugs, l_plugs = ax_full.get_legend_handles_labels()
    custom = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
               label=f"legacy ({len(legacy_events)})"),
        Line2D([0], [0], color="tab:green", linestyle="-", linewidth=1.2,
               label=f"TP ({len(tp_events)})"),
        Line2D([0], [0], color="tab:red", linestyle="-", linewidth=1.6,
               label=f"FP ({len(fp_events)})"),
    ]
    ax_full.legend(h_plugs + custom, l_plugs + [h.get_label() for h in custom],
                   loc="upper right", fontsize=8, ncol=3)

    recall = len(tp_events) / len(legacy_events) if legacy_events else 0.0
    precision = len(tp_events) / len(detected_events) if detected_events else 0.0
    ax_full.set_title(
        f"target_no={target_no}  threshold={_THRESHOLD} quorum={_QUORUM} window={_WINDOW_DAYS}d  "
        f"TP={len(tp_events)} FP={len(fp_events)} legacy={len(legacy_events)}  "
        f"R={recall:.2f} P={precision:.2f}",
        fontsize=11,
    )

    # FP zoom サブプロット
    for i, fp_date in enumerate(fp_events):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        _plot_fp_zoom(ax, daily_max, fp_date, legacy_events)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=3)

    fig.tight_layout()
    out_path = _OUT_DIR / f"fp_{target_no}_t{int(_THRESHOLD)}_q{_QUORUM}_w{_WINDOW_DAYS}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # FP の日付を標準出力に（目視確認用リスト）
    if fp_events:
        fp_list = ", ".join(ev.strftime("%Y-%m-%d") for ev in fp_events)
        print(f"  [{target_no}] FP dates: {fp_list}", file=sys.stderr)

    return out_path


def main(target_nos: list[str]) -> int:
    if not _LEGACY_CSV.exists():
        print(f"legacy CSV not found: {_LEGACY_CSV}", file=sys.stderr)
        return 1
    if not _INPUT_DIR.exists():
        print(f"input dir not found: {_INPUT_DIR}", file=sys.stderr)
        return 1

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    legacy = pd.read_csv(_LEGACY_CSV, encoding="utf-8-sig")
    legacy[_DT_COL] = pd.to_datetime(legacy[_DT_COL])
    legacy["target_no"] = legacy["file_name"].str.replace(".csv", "", regex=False)

    # ADR-012 カットオフ前は除外
    def _is_after(row: pd.Series) -> bool:
        return _cutoff_status(row["target_no"], row[_DT_COL].to_pydatetime()) != "before_cutoff"
    legacy = legacy[legacy.apply(_is_after, axis=1)].reset_index(drop=True)

    if not target_nos:
        target_nos = sorted(legacy["target_no"].unique().tolist())

    print(f"Plotting FP analysis for {len(target_nos)} locations "
          f"(params: t={_THRESHOLD} q={_QUORUM} w={_WINDOW_DAYS})",
          file=sys.stderr)
    for target_no in target_nos:
        events = legacy[legacy["target_no"] == target_no]
        csv_path = _INPUT_DIR / f"{target_no}.csv"
        if not csv_path.exists():
            print(f"[{target_no}] no input CSV", file=sys.stderr)
            continue
        legacy_evts = [pd.Timestamp(dt).normalize() for dt in events[_DT_COL]]
        try:
            out = _plot_location(target_no, legacy_evts, csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[{target_no}] ERROR: {exc}", file=sys.stderr)
            continue
        print(f"[{target_no}] -> {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
