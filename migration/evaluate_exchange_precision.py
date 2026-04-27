"""ADR-014 検出パラメータ校正: precision / F1 評価（Plan A-4）。

`calibrate_exchange_detection.py` は legacy CSV のイベント日にだけ検出ロジックを
適用し recall を測った。本スクリプトは **全運転日を走査** し、検出ロジックが
legacy CSV に無い日に偽陽性を出していないかを評価する。

評価方法:
- 各機場の運転日 D を全て走査し、median shift 方式で `is_detected(D)` を判定
- legacy CSV のイベント日 E に対し、`|D - E| <= merge_window_days` の D は真陽性
  （同じイベントに複数日ヒットすることがあるので、イベント単位で最大 1 本としてまとめる）
- legacy に対応しない D は偽陽性
- 評価指標:
  - recall = detected legacy events / total legacy events
  - precision = detected legacy events / (detected legacy events + detected non-legacy days)
  - F1 = 2 * P * R / (P + R)

連続する検出日は `merge_window_days` 以内なら 1 イベントとしてマージ（run-length encoding）。
これにより交換直後の「level shift 後の安定期間」が連続で検出されて precision が
不当に低くなる問題を回避。

出力:
- `experiments/exchange-detection-calibration/precision-grid-results.csv`
- `experiments/exchange-detection-calibration/precision-grid-results.md`
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd

_DT_COL: Final[str] = "dailygraphpt_ptdatetime"
_POWER_COL: Final[str] = "発電機電力"
_VOLTAGE_COLS_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧_{i}" for i in range(1, 7))
_VOLTAGE_COLS_NO_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧{i}" for i in range(1, 7))

# グリッドサーチ範囲（recall 高得点帯に集中）
_THRESHOLDS: Final[tuple[float, ...]] = (3.0, 5.0, 7.0, 10.0, 12.0, 15.0)
_QUORUMS: Final[tuple[int, ...]] = (2, 3, 4)
_WINDOWS: Final[tuple[int, ...]] = (7, 10, 14)
_MIN_DAYS_EACH_SIDE: Final[int] = 3

# イベントマージ窓: この日数以内の連続検出は同一イベントとして 1 本に
_MERGE_WINDOW_DAYS: Final[int] = 7

# 再現率評価: 検出日 D と legacy 日 E が |D - E| <= _MATCH_TOLERANCE_DAYS なら同一イベント扱い
_MATCH_TOLERANCE_DAYS: Final[int] = 7

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
class EvalResult:
    threshold: float
    quorum: int
    window_days: int
    total_legacy_events: int
    detected_legacy_events: int  # TP (event-level)
    false_positive_events: int  # FP (event-level, non-legacy)
    total_operating_days: int

    @property
    def recall(self) -> float:
        return self.detected_legacy_events / self.total_legacy_events if self.total_legacy_events else 0.0

    @property
    def precision(self) -> float:
        total_detected = self.detected_legacy_events + self.false_positive_events
        return self.detected_legacy_events / total_detected if total_detected else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


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


def _scan_detect_days(
    daily_max: pd.DataFrame,
    *,
    threshold: float,
    quorum: int,
    window_days: int,
    min_days_each_side: int = _MIN_DAYS_EACH_SIDE,
) -> list[pd.Timestamp]:
    """全運転日を走査し、検出ロジックが正の日を返す。"""
    if daily_max.empty:
        return []
    plug_cols = [f"plug_{i}" for i in range(1, 7)]
    dates = daily_max["date"].to_numpy()
    # 日付 → index の辞書（前半/後半スライス高速化）
    detected: list[pd.Timestamp] = []
    for i, d in enumerate(dates):
        event_date = pd.Timestamp(d)
        before_start = event_date - pd.Timedelta(days=window_days)
        before_end = event_date - pd.Timedelta(days=1)
        after_start = event_date + pd.Timedelta(days=1)
        after_end = event_date + pd.Timedelta(days=window_days)

        before = daily_max[(daily_max["date"] >= before_start) & (daily_max["date"] <= before_end)]
        after = daily_max[(daily_max["date"] >= after_start) & (daily_max["date"] <= after_end)]

        drop_count = 0
        valid_plugs = 0
        for col in plug_cols:
            b = before[col].dropna()
            a = after[col].dropna()
            if len(b) < min_days_each_side or len(a) < min_days_each_side:
                continue
            valid_plugs += 1
            shift = abs(float(a.median()) - float(b.median()))
            if shift > threshold:
                drop_count += 1
        if valid_plugs == 0:
            continue
        if drop_count >= quorum:
            detected.append(event_date)
    return detected


def _merge_consecutive(detected_days: list[pd.Timestamp], merge_window: int) -> list[pd.Timestamp]:
    """連続検出日を merge_window 以内でまとめて 1 イベントに。各グループの中央日を返す。"""
    if not detected_days:
        return []
    sorted_days = sorted(detected_days)
    groups: list[list[pd.Timestamp]] = [[sorted_days[0]]]
    for d in sorted_days[1:]:
        if (d - groups[-1][-1]).days <= merge_window:
            groups[-1].append(d)
        else:
            groups.append([d])
    # 各グループの中央日を代表に（偶数個なら後半の先頭）
    return [g[len(g) // 2] for g in groups]


def _match_to_legacy(
    detected_events: list[pd.Timestamp],
    legacy_events: list[pd.Timestamp],
    tolerance_days: int,
) -> tuple[int, int]:
    """検出イベントを legacy イベントに最大マッチング。戻り値: (TP, FP)。

    各 legacy event は最大 1 回マッチ。各検出イベントは最も近い未マッチ legacy event に
    tolerance 以内で対応付け。
    """
    legacy_sorted = sorted(legacy_events)
    detected_sorted = sorted(detected_events)
    matched_legacy: set[int] = set()
    tp = 0
    fp = 0
    for d in detected_sorted:
        best_idx = -1
        best_dist = tolerance_days + 1
        for i, e in enumerate(legacy_sorted):
            if i in matched_legacy:
                continue
            dist = abs((d - e).days)
            if dist <= tolerance_days and dist < best_dist:
                best_idx = i
                best_dist = dist
        if best_idx >= 0:
            matched_legacy.add(best_idx)
            tp += 1
        else:
            fp += 1
    return tp, fp


def _cutoff_status(target_no: str, event_dt: datetime) -> str:
    cutoff = _EP370G_START_DATETIMES.get(target_no)
    if cutoff is None:
        return "no_cutoff"
    return "after_cutoff" if event_dt >= cutoff else "before_cutoff"


def _evaluate_combo(
    threshold: float,
    quorum: int,
    window: int,
    legacy_events_per_loc: dict[str, list[pd.Timestamp]],
    daily_max_per_loc: dict[str, pd.DataFrame],
    cutoff_per_loc: dict[str, pd.Timestamp | None],
) -> EvalResult:
    total_tp = 0
    total_fp = 0
    total_legacy = 0
    total_opdays = 0
    for target_no, legacy_events in legacy_events_per_loc.items():
        daily_max = daily_max_per_loc[target_no]
        # カットオフ前の検出は学習対象外として分母から除外
        cutoff = cutoff_per_loc.get(target_no)
        if cutoff is not None:
            daily_max = daily_max[daily_max["date"] >= cutoff]

        detected_days = _scan_detect_days(
            daily_max, threshold=threshold, quorum=quorum, window_days=window,
        )
        detected_events = _merge_consecutive(detected_days, _MERGE_WINDOW_DAYS)
        tp, fp = _match_to_legacy(detected_events, legacy_events, _MATCH_TOLERANCE_DAYS)
        total_tp += tp
        total_fp += fp
        total_legacy += len(legacy_events)
        total_opdays += len(daily_max)
    return EvalResult(
        threshold=threshold,
        quorum=quorum,
        window_days=window,
        total_legacy_events=total_legacy,
        detected_legacy_events=total_tp,
        false_positive_events=total_fp,
        total_operating_days=total_opdays,
    )


def _per_location_breakdown(
    best: EvalResult,
    legacy_events_per_loc: dict[str, list[pd.Timestamp]],
    daily_max_per_loc: dict[str, pd.DataFrame],
    cutoff_per_loc: dict[str, pd.Timestamp | None],
) -> pd.DataFrame:
    rows: list[dict] = []
    for target_no, legacy_events in sorted(legacy_events_per_loc.items()):
        daily_max = daily_max_per_loc[target_no]
        cutoff = cutoff_per_loc.get(target_no)
        if cutoff is not None:
            daily_max = daily_max[daily_max["date"] >= cutoff]
        detected_days = _scan_detect_days(
            daily_max, threshold=best.threshold, quorum=best.quorum, window_days=best.window_days,
        )
        detected_events = _merge_consecutive(detected_days, _MERGE_WINDOW_DAYS)
        tp, fp = _match_to_legacy(detected_events, legacy_events, _MATCH_TOLERANCE_DAYS)
        rows.append({
            "target_no": target_no,
            "operating_days": len(daily_max),
            "legacy_events": len(legacy_events),
            "detected_events": len(detected_events),
            "tp": tp,
            "fp": fp,
            "recall": tp / len(legacy_events) if legacy_events else 0.0,
            "precision": tp / len(detected_events) if detected_events else 0.0,
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
    results: list[EvalResult],
    best_f1: EvalResult,
    best_precision_at_90_recall: EvalResult | None,
    per_loc: pd.DataFrame,
    out_md: Path,
) -> None:
    results_df = pd.DataFrame([
        {
            "threshold_kV": r.threshold,
            "quorum": r.quorum,
            "window_days": r.window_days,
            "tp": r.detected_legacy_events,
            "fp": r.false_positive_events,
            "recall": r.recall,
            "precision": r.precision,
            "f1": r.f1,
        }
        for r in results
    ])

    lines: list[str] = []
    lines.append("# legacy exchange 検出パラメータ校正（precision/F1 評価）\n")
    lines.append(
        "- 生成: `migration/evaluate_exchange_precision.py`\n"
        "- 方式: 前半窓 median vs 後半窓 median、全運転日走査 + 連続日マージ\n"
        f"- マージ窓: {_MERGE_WINDOW_DAYS} 日以内の連続検出を同一イベント扱い\n"
        f"- マッチ許容: |検出日 − legacy 日| <= {_MATCH_TOLERANCE_DAYS} 日で TP\n"
        f"- 片側 median の最小運転日数: {_MIN_DAYS_EACH_SIDE}\n"
        "- ADR-012 カットオフ後の日だけを走査対象とする（カットオフ前の legacy 1 件は分母外）\n\n"
    )

    lines.append("## F1 最大の組合せ\n")
    lines.append(
        f"- threshold = **{best_f1.threshold:.1f} kV**\n"
        f"- quorum    = **{best_f1.quorum}**\n"
        f"- window    = **{best_f1.window_days} 日**\n"
        f"- TP = {best_f1.detected_legacy_events}, FP = {best_f1.false_positive_events}\n"
        f"- recall    = {best_f1.recall:.1%}\n"
        f"- precision = {best_f1.precision:.1%}\n"
        f"- F1        = {best_f1.f1:.3f}\n\n"
    )

    if best_precision_at_90_recall is not None:
        b = best_precision_at_90_recall
        lines.append("## recall >= 90% の条件下で precision 最大\n")
        lines.append(
            f"- threshold = **{b.threshold:.1f} kV**\n"
            f"- quorum    = **{b.quorum}**\n"
            f"- window    = **{b.window_days} 日**\n"
            f"- TP = {b.detected_legacy_events}, FP = {b.false_positive_events}\n"
            f"- recall    = {b.recall:.1%}\n"
            f"- precision = {b.precision:.1%}\n"
            f"- F1        = {b.f1:.3f}\n\n"
        )
    else:
        lines.append("## recall >= 90% の条件下で precision 最大\n")
        lines.append("- 該当組合せ無し（recall 90% を満たす組合せが存在しない）\n\n")

    lines.append("## 全グリッド結果（F1 降順）\n")
    top = results_df.sort_values(["f1", "recall"], ascending=False)
    lines.append(_to_markdown_table(top))
    lines.append("")

    lines.append("\n## F1 最大組合せでの機場別 breakdown\n")
    lines.append(_to_markdown_table(per_loc))
    lines.append("")

    lines.append("\n## 解釈のヒント\n")
    lines.append(
        "- FP は「legacy CSV に無いが検出ロジックが拾った日」。legacy CSV が完全な正解と\n"
        "  仮定した場合の偽陽性だが、legacy CSV 自体の記録漏れもあり得る（真の FP とは限らない）\n"
        "- 実運用では FP が学習データに「交換したことにされる」ため、世代境界の誤挿入で\n"
        "  baseline 計算を壊すリスクがある。precision は recall と同等以上に重要\n"
        "- 機場別 breakdown で `fp >> tp` の機場は、その機場の電圧変動が大きくしきい値で\n"
        "  拾いきれない。機場別閾値 or 別特徴量を検討\n"
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

    def _is_after(row: pd.Series) -> bool:
        return _cutoff_status(row["target_no"], row[_DT_COL].to_pydatetime()) != "before_cutoff"

    n_total = len(legacy)
    legacy_learn = legacy[legacy.apply(_is_after, axis=1)].reset_index(drop=True)
    print(f"legacy: {n_total} total, {n_total - len(legacy_learn)} before_cutoff, "
          f"{len(legacy_learn)} learning", file=sys.stderr)

    legacy_events_per_loc: dict[str, list[pd.Timestamp]] = {}
    daily_max_per_loc: dict[str, pd.DataFrame] = {}
    cutoff_per_loc: dict[str, pd.Timestamp | None] = {}
    for target_no, events in legacy_learn.groupby("target_no"):
        csv_path = _INPUT_DIR / f"{target_no}.csv"
        if not csv_path.exists():
            continue
        try:
            daily_max_per_loc[target_no] = _load_location_daily_max(csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{target_no}] ERROR loading: {exc}", file=sys.stderr)
            continue
        legacy_events_per_loc[target_no] = [
            pd.Timestamp(dt).normalize() for dt in events[_DT_COL]
        ]
        cutoff = _EP370G_START_DATETIMES.get(target_no)
        cutoff_per_loc[target_no] = pd.Timestamp(cutoff).normalize() if cutoff else None
        print(f"  [{target_no}] legacy={len(events)} opdays={len(daily_max_per_loc[target_no])}",
              file=sys.stderr)

    combos = list(itertools.product(_THRESHOLDS, _QUORUMS, _WINDOWS))
    print(f"Eval: {len(combos)} combos (full-day scan, may take a minute)...", file=sys.stderr)
    results: list[EvalResult] = []
    for i, (threshold, quorum, window) in enumerate(combos, 1):
        r = _evaluate_combo(threshold, quorum, window,
                            legacy_events_per_loc, daily_max_per_loc, cutoff_per_loc)
        results.append(r)
        print(f"  [{i}/{len(combos)}] t={threshold} q={quorum} w={window}  "
              f"P={r.precision:.2f} R={r.recall:.2f} F1={r.f1:.3f}  "
              f"TP={r.detected_legacy_events} FP={r.false_positive_events}", file=sys.stderr)

    # 保存
    results_df = pd.DataFrame([asdict(r) for r in results])
    results_df["recall"] = [r.recall for r in results]
    results_df["precision"] = [r.precision for r in results]
    results_df["f1"] = [r.f1 for r in results]
    out_csv = _OUT_DIR / "precision-grid-results.csv"
    results_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    best_f1 = max(results, key=lambda r: (r.f1, r.recall))
    high_recall = [r for r in results if r.recall >= 0.90]
    best_prec_90r = max(high_recall, key=lambda r: r.precision) if high_recall else None

    per_loc = _per_location_breakdown(
        best_f1, legacy_events_per_loc, daily_max_per_loc, cutoff_per_loc,
    )

    out_md = _OUT_DIR / "precision-grid-results.md"
    _write_report(results, best_f1, best_prec_90r, per_loc, out_md)

    print(f"Best F1: t={best_f1.threshold} q={best_f1.quorum} w={best_f1.window_days}  "
          f"F1={best_f1.f1:.3f} P={best_f1.precision:.2f} R={best_f1.recall:.2f}", file=sys.stderr)
    print(f"Wrote: {out_csv}", file=sys.stderr)
    print(f"Wrote: {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
