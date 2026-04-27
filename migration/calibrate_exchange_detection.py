"""ADR-014 検出パラメータ校正: 前半 median vs 後半 median 方式でグリッドサーチ。

Phase 0.5 の目視確認（5630 / 9110 / 9111）で分かった事実：

1. legacy ラベルは信用できる（3 機場 25 イベント全てで level shift が見える）
2. `inspect_legacy_exchange_labels.py` の「隣接 2 日 diff」方式は弱い
   - 非運転日を挟むと diff が取れない
   - 交換直後の慣らし運転で徐々に電圧安定するパターンに弱い
3. 機場ごとに level shift の幅が違う（5630: 10〜15 kV / 9110: 3〜10 kV）

本スクリプトは **前半窓 median vs 後半窓 median の差** を交換検出指標とし、
しきい値・quorum・窓幅のグリッドサーチで confirmed 率を最大化する組合せを探る。

検出ロジック:
- イベント日時 D に対し、窓 ± W 日:
  - before = [D - W, D - 1]（D 当日を除く）
  - after  = [D + 1, D + W]（D 当日を除く）
- 各プラグについて、before / after の日次最大電圧 median を計算
  - 片側でも運転日数 < min_days_each_side ならそのプラグは除外（data_insufficient として扱う）
- |median_before − median_after| > threshold のプラグ数を数える
  - count >= quorum → confirmed
  - 1 <= count < quorum → suspect
  - count == 0 → not_observed
  - 全プラグ除外なら data_missing

出力:
- `experiments/exchange-detection-calibration/grid-search-results.csv`: 全グリッド結果
- `experiments/exchange-detection-calibration/grid-search-results.md`: 人間レビュー向け
- `experiments/exchange-detection-calibration/best-params-per-location.md`: ベスト組合せの機場別 breakdown

ADR-012 カットオフ前のイベントは学習対象外として分母から除外（`inspect_legacy_exchange_labels.py` と同じ）。
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd

_DT_COL: Final[str] = "dailygraphpt_ptdatetime"
_POWER_COL: Final[str] = "発電機電力"
_VOLTAGE_COLS_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧_{i}" for i in range(1, 7))
_VOLTAGE_COLS_NO_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧{i}" for i in range(1, 7))

# グリッドサーチ範囲（目視所見ベース）
_THRESHOLDS: Final[tuple[float, ...]] = (3.0, 5.0, 7.0, 10.0, 12.0, 15.0)
_QUORUMS: Final[tuple[int, ...]] = (2, 3, 4, 5)
_WINDOWS: Final[tuple[int, ...]] = (7, 10, 14)
_MIN_DAYS_EACH_SIDE: Final[int] = 3  # 片側 median の最小運転日数

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

_LEGACY_CSV: Final[Path] = Path(
    "D:/code_commit/EP-VoltPredictor/ep_voltpredictor/exchange_timings_summary4.csv"
)
_INPUT_DIR: Final[Path] = Path("E:/gene/input/EP370G_orig")
_OUT_DIR: Final[Path] = Path(
    "E:/projects/contact-center-toolbox/60_domains/ress/gene_plug_voltage_predictor/"
    "experiments/exchange-detection-calibration"
)


@dataclass
class GridResult:
    threshold: float
    quorum: int
    window_days: int
    n_events: int
    confirmed: int
    suspect: int
    not_observed: int
    data_missing: int

    @property
    def confirmed_rate(self) -> float:
        return self.confirmed / self.n_events if self.n_events else 0.0

    @property
    def recall_confirmed_or_suspect(self) -> float:
        """confirmed + suspect を「少なくとも何か検出した」として再現率化。"""
        if self.n_events == 0:
            return 0.0
        return (self.confirmed + self.suspect) / self.n_events


def _resolve_voltage_cols(path: Path) -> tuple[str, ...]:
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    available = set(head.columns)
    if set(_VOLTAGE_COLS_NO_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_NO_UNDERSCORE
    if set(_VOLTAGE_COLS_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_UNDERSCORE
    raise ValueError(f"No complete 要求電圧 column set found in {path.name}")


def _load_location_daily_max(path: Path) -> pd.DataFrame:
    voltage_cols = _resolve_voltage_cols(path)
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=[_DT_COL, _POWER_COL, *voltage_cols],
    )
    df[_DT_COL] = pd.to_datetime(df[_DT_COL], errors="coerce")
    df = df.dropna(subset=[_DT_COL])
    running_mask = df[_POWER_COL] > 0
    if not running_mask.any():
        return pd.DataFrame(columns=["date"] + [f"plug_{i}" for i in range(1, 7)])
    df = df.loc[running_mask].copy()
    df["date"] = df[_DT_COL].dt.normalize()

    out = pd.DataFrame({"date": sorted(df["date"].unique())})
    for i, col in enumerate(voltage_cols, 1):
        valid = df[col] > 0
        daily_max = (
            df.loc[valid].groupby("date")[col].max().rename(f"plug_{i}")
        )
        out = out.merge(daily_max, on="date", how="left")
    return out.reset_index(drop=True)


def _classify_event_median(
    daily_max: pd.DataFrame,
    event_dt: datetime,
    *,
    window_days: int,
    threshold: float,
    quorum: int,
    min_days_each_side: int = _MIN_DAYS_EACH_SIDE,
) -> tuple[str, int, float]:
    """前半/後半 median 差でイベントを分類。

    戻り値: (status, drop_plug_count, max_shift_voltage)
    """
    event_date = pd.Timestamp(event_dt).normalize()
    before_start = event_date - pd.Timedelta(days=window_days)
    before_end = event_date - pd.Timedelta(days=1)
    after_start = event_date + pd.Timedelta(days=1)
    after_end = event_date + pd.Timedelta(days=window_days)

    before = daily_max[(daily_max["date"] >= before_start) & (daily_max["date"] <= before_end)]
    after = daily_max[(daily_max["date"] >= after_start) & (daily_max["date"] <= after_end)]

    plug_cols = [f"plug_{i}" for i in range(1, 7)]
    drop_count = 0
    max_shift = 0.0
    valid_plugs = 0
    for col in plug_cols:
        b = before[col].dropna()
        a = after[col].dropna()
        if len(b) < min_days_each_side or len(a) < min_days_each_side:
            continue
        valid_plugs += 1
        shift = abs(float(a.median()) - float(b.median()))
        if shift > max_shift:
            max_shift = shift
        if shift > threshold:
            drop_count += 1

    if valid_plugs == 0:
        return "data_missing", 0, 0.0
    if drop_count >= quorum:
        return "confirmed", drop_count, max_shift
    if drop_count > 0:
        return "suspect", drop_count, max_shift
    return "not_observed", drop_count, max_shift


def _cutoff_status(target_no: str, event_dt: datetime) -> str:
    cutoff = _EP370G_START_DATETIMES.get(target_no)
    if cutoff is None:
        return "no_cutoff"
    return "after_cutoff" if event_dt >= cutoff else "before_cutoff"


def _run_grid(
    legacy_events_per_loc: dict[str, list[pd.Timestamp]],
    daily_max_per_loc: dict[str, pd.DataFrame],
) -> list[GridResult]:
    results: list[GridResult] = []
    for threshold, quorum, window in itertools.product(_THRESHOLDS, _QUORUMS, _WINDOWS):
        counts = Counter()
        n_events = 0
        for target_no, events in legacy_events_per_loc.items():
            daily_max = daily_max_per_loc[target_no]
            for ev in events:
                status, _, _ = _classify_event_median(
                    daily_max, ev.to_pydatetime(),
                    window_days=window, threshold=threshold, quorum=quorum,
                )
                counts[status] += 1
                n_events += 1
        results.append(GridResult(
            threshold=threshold,
            quorum=quorum,
            window_days=window,
            n_events=n_events,
            confirmed=counts.get("confirmed", 0),
            suspect=counts.get("suspect", 0),
            not_observed=counts.get("not_observed", 0),
            data_missing=counts.get("data_missing", 0),
        ))
    return results


def _per_location_breakdown(
    best: GridResult,
    legacy_events_per_loc: dict[str, list[pd.Timestamp]],
    daily_max_per_loc: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict] = []
    for target_no, events in sorted(legacy_events_per_loc.items()):
        daily_max = daily_max_per_loc[target_no]
        counts = Counter()
        max_shifts: list[float] = []
        for ev in events:
            status, _, shift = _classify_event_median(
                daily_max, ev.to_pydatetime(),
                window_days=best.window_days,
                threshold=best.threshold, quorum=best.quorum,
            )
            counts[status] += 1
            max_shifts.append(shift)
        rows.append({
            "target_no": target_no,
            "events_total": len(events),
            "confirmed": counts.get("confirmed", 0),
            "suspect": counts.get("suspect", 0),
            "not_observed": counts.get("not_observed", 0),
            "data_missing": counts.get("data_missing", 0),
            "median_max_shift_kV": round(float(pd.Series(max_shifts).median()), 2) if max_shifts else 0.0,
            "min_max_shift_kV": round(min(max_shifts), 2) if max_shifts else 0.0,
        })
    return pd.DataFrame(rows)


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(
    results: list[GridResult],
    best: GridResult,
    per_loc: pd.DataFrame,
    n_before_cutoff_excluded: int,
    out_md: Path,
) -> None:
    # サマリーテーブル
    results_df = pd.DataFrame([
        {
            "threshold_kV": r.threshold,
            "quorum": r.quorum,
            "window_days": r.window_days,
            "confirmed": r.confirmed,
            "suspect": r.suspect,
            "not_observed": r.not_observed,
            "data_missing": r.data_missing,
            "confirmed_rate": r.confirmed_rate,
            "recall_c_or_s": r.recall_confirmed_or_suspect,
        }
        for r in results
    ])

    lines: list[str] = []
    lines.append("# legacy exchange 検出パラメータ校正（前半/後半 median 方式）\n")
    lines.append(
        "- 生成: `migration/calibrate_exchange_detection.py`\n"
        "- 検出ロジック: 前半窓 median vs 後半窓 median の差がしきい値超のプラグ数 >= quorum → confirmed\n"
        "- 片側 median の最小運転日数: "
        f"{_MIN_DAYS_EACH_SIDE} 日未満のプラグは除外\n"
        f"- 分母: legacy CSV 全イベント − before_cutoff（{n_before_cutoff_excluded} 件）\n"
        "- ADR-012 カットオフ: 5630/8950/9290/9380/9381/9690（他機場は no_cutoff）\n\n"
    )

    lines.append("## ベスト組合せ（confirmed_rate が最大）\n")
    lines.append(
        f"- threshold = **{best.threshold:.1f} kV**\n"
        f"- quorum    = **{best.quorum}**\n"
        f"- window    = **{best.window_days} 日**\n"
        f"- confirmed = {best.confirmed}/{best.n_events} ({best.confirmed_rate:.1%})\n"
        f"- suspect   = {best.suspect}\n"
        f"- not_observed = {best.not_observed}\n"
        f"- data_missing = {best.data_missing}\n"
        f"- recall (confirmed + suspect) = {best.recall_confirmed_or_suspect:.1%}\n\n"
    )

    lines.append("## 全グリッド結果（confirmed_rate 降順）\n")
    top = results_df.sort_values(["confirmed_rate", "recall_c_or_s"], ascending=False)
    lines.append(_to_markdown_table(top))
    lines.append("")

    lines.append("\n## ベスト組合せでの機場別 breakdown\n")
    lines.append(_to_markdown_table(per_loc))
    lines.append("")

    # 解釈ガイド
    lines.append("\n## 解釈のヒント\n")
    lines.append(
        "- `confirmed_rate` が 70%+ を実用ラインとみなす。それ未満は閾値を緩めるか判定方式を見直す必要\n"
        "- `not_observed` が多い機場は「その機場の交換時 level shift が小さい」か「legacy ラベル誤記」の\n"
        "  どちらか。目視確認（`plot_exchange_window.py`）で判別する\n"
        "- `min_max_shift_kV` が閾値を大幅に下回る機場は、そもそもその機場で検出困難。\n"
        "  機場別閾値（C-4 候補、ADR-014 で議論）か、別特徴量（moving average、回帰 slope 等）検討\n"
        "- quorum を下げるほど recall は上がるが precision は下がる（legacy CSV に含まれない偽陽性が\n"
        "  増える可能性）。**真の precision は未評価** であり、本校正は legacy CSV に対する recall 寄り\n"
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
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

    # ADR-012 カットオフ前のイベントは除外
    def _is_after(row: pd.Series) -> bool:
        return _cutoff_status(row["target_no"], row[_DT_COL].to_pydatetime()) != "before_cutoff"

    n_total = len(legacy)
    legacy_learn = legacy[legacy.apply(_is_after, axis=1)].reset_index(drop=True)
    n_excluded = n_total - len(legacy_learn)
    print(f"legacy: {n_total} total, {n_excluded} before_cutoff, {len(legacy_learn)} learning", file=sys.stderr)

    legacy_events_per_loc: dict[str, list[pd.Timestamp]] = {}
    daily_max_per_loc: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for target_no, events in legacy_learn.groupby("target_no"):
        csv_path = _INPUT_DIR / f"{target_no}.csv"
        if not csv_path.exists():
            missing.append(target_no)
            print(f"  [{target_no}] SKIP: no input CSV", file=sys.stderr)
            continue
        try:
            daily_max_per_loc[target_no] = _load_location_daily_max(csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{target_no}] ERROR loading: {exc}", file=sys.stderr)
            continue
        legacy_events_per_loc[target_no] = list(events[_DT_COL])
        print(f"  [{target_no}] {len(events)} events loaded", file=sys.stderr)

    print(
        f"Grid search: {len(_THRESHOLDS)} × {len(_QUORUMS)} × {len(_WINDOWS)} "
        f"= {len(_THRESHOLDS) * len(_QUORUMS) * len(_WINDOWS)} combos",
        file=sys.stderr,
    )
    results = _run_grid(legacy_events_per_loc, daily_max_per_loc)

    # CSV 出力
    results_df = pd.DataFrame([asdict(r) for r in results])
    results_df["confirmed_rate"] = [r.confirmed_rate for r in results]
    results_df["recall_confirmed_or_suspect"] = [r.recall_confirmed_or_suspect for r in results]
    out_csv = _OUT_DIR / "grid-search-results.csv"
    results_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ベスト組合せ
    best = max(results, key=lambda r: (r.confirmed_rate, r.recall_confirmed_or_suspect))
    per_loc = _per_location_breakdown(best, legacy_events_per_loc, daily_max_per_loc)

    out_md = _OUT_DIR / "grid-search-results.md"
    _write_report(results, best, per_loc, n_excluded, out_md)

    print(f"Best: threshold={best.threshold} quorum={best.quorum} window={best.window_days}  "
          f"confirmed={best.confirmed}/{best.n_events} ({best.confirmed_rate:.1%})", file=sys.stderr)
    print(f"Wrote: {out_csv}", file=sys.stderr)
    print(f"Wrote: {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
