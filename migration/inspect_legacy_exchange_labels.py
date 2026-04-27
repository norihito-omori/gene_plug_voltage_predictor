"""legacy `exchange_timings_summary4.csv` の品質を調査する一度きりのヘルパー。

ADR-014 の校正で legacy CSV を「正解ラベル」として扱う前に、ラベルが信頼できるか
目視・機械両面で確認する（Phase 0.5）。

検証観点:
- 機場ごとの交換回数・交換間隔の分布
- 各イベント日時の前後（± `_WINDOW_DAYS` 日）に実際の電圧急落があったか照合
- ADR-012 の機場別カットオフ前に記録されたイベント（学習対象外）を分離

照合方法:
- legacy CSV の各イベント (機場, 日時) について、現行データ `E:/gene/input/EP370G_orig/`
  から該当機場の CSV を読み、`dailygraphpt_ptdatetime` が event_date ± 7 日の範囲で
  運転中日次最大電圧を集約し、`abs(diff) > _DROP_THRESHOLD` となるプラグ数を数える。
- 同時に急落したプラグが `_QUORUM` 本以上あれば `confirmed`、それ未満なら `suspect`、
  0 本なら `not_observed` と分類。

出力:
- `experiments/exchange-detection-calibration/legacy-label-quality.csv`: 機械処理向け
- `experiments/exchange-detection-calibration/legacy-label-quality.md`  : 人間レビュー向け

ガードレール:
- 判定結果は「ラベル品質調査」であり、アルゴリズムの最終 F1 ではない。人間レビュー用。
- 運転中判定は `発電機電力 > 0`（ADR-013 C-1）。
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import pandas as pd

from runtime_utils import unpack_to_hours

# 照合ウィンドウ: イベント日時 ± _WINDOW_DAYS 日の間に急落があるか調べる
_WINDOW_DAYS: Final[int] = 7
# プラグ別日次最大電圧の 1 日あたり絶対差分がこの値を超えたら「急落候補日」
_DROP_THRESHOLD: Final[float] = 20.0
# 同一日に何本のプラグが同時急落していれば「機場交換」とみなすか
_QUORUM: Final[int] = 4

_DT_COL: Final[str] = "dailygraphpt_ptdatetime"
_POWER_COL: Final[str] = "発電機電力"
_VOLTAGE_COLS_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧_{i}" for i in range(1, 7))
_VOLTAGE_COLS_NO_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧{i}" for i in range(1, 7))

# ADR-012: EP370G 機場別の開始日時カットオフ
_EP370G_START_DATETIMES: Final[dict[str, datetime]] = {
    "5630": datetime(2022, 6, 26, 0, 30),
    "8950": datetime(2021, 6, 12, 20, 30),
    "9290": datetime(2022, 7, 21, 10, 0),
    "9380": datetime(2023, 5, 31, 0, 30),
    "9381": datetime(2023, 5, 31, 14, 0),
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
class EventQuality:
    target_no: str
    event_datetime: str
    event_date: str
    cutoff_status: str  # "before_cutoff" | "after_cutoff" | "no_cutoff"
    confirmation: str  # "confirmed" | "suspect" | "not_observed" | "data_missing"
    drop_plugs_in_window: int
    max_drop_voltage: float
    days_from_prev_event: int | None
    rt_at_event_hours: float


def _resolve_voltage_cols(path: Path) -> tuple[str, ...]:
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    available = set(head.columns)
    if set(_VOLTAGE_COLS_NO_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_NO_UNDERSCORE
    if set(_VOLTAGE_COLS_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_UNDERSCORE
    raise ValueError(f"No complete 要求電圧 column set found in {path.name}")


def _load_location_daily_max(path: Path) -> pd.DataFrame:
    """機場 CSV を読み、プラグ別日次最大電圧（運転中のみ）の wide DataFrame を返す。

    列: date, plug_1..plug_6。運転日が 0 の日は行自体が存在しない。
    """
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
        return pd.DataFrame(
            columns=["date"] + [f"plug_{i}" for i in range(1, 7)]
        )
    df = df.loc[running_mask].copy()
    df["date"] = df[_DT_COL].dt.normalize()

    out = pd.DataFrame({"date": df["date"].unique()}).sort_values("date")
    for i, col in enumerate(voltage_cols, 1):
        # 運転中行かつプラグ値 > 0 の行だけ採用（保持値ではなく有効センサー値）
        valid = df[col] > 0
        daily_max = (
            df.loc[valid].groupby("date")[col].max().rename(f"plug_{i}")
        )
        out = out.merge(daily_max, on="date", how="left")
    return out.reset_index(drop=True)


def _cutoff_status(target_no: str, event_dt: datetime) -> str:
    cutoff = _EP370G_START_DATETIMES.get(target_no)
    if cutoff is None:
        return "no_cutoff"
    return "after_cutoff" if event_dt >= cutoff else "before_cutoff"


def _classify_event(
    daily_max: pd.DataFrame, event_dt: datetime
) -> tuple[str, int, float]:
    """event_dt ± _WINDOW_DAYS の daily_max に対し、プラグ別の最大絶対差分を計算。

    急落プラグ数（diff > _DROP_THRESHOLD）が _QUORUM 以上 → confirmed、
    1 以上 _QUORUM 未満 → suspect、0 → not_observed。
    窓内にデータが無ければ data_missing。
    """
    event_date = pd.Timestamp(event_dt).normalize()
    start = event_date - pd.Timedelta(days=_WINDOW_DAYS)
    end = event_date + pd.Timedelta(days=_WINDOW_DAYS)
    window = daily_max[(daily_max["date"] >= start) & (daily_max["date"] <= end)]
    if window.empty:
        return "data_missing", 0, 0.0
    plug_cols = [f"plug_{i}" for i in range(1, 7)]
    drop_count = 0
    max_drop = 0.0
    for col in plug_cols:
        s = window[col].dropna()
        if len(s) < 2:
            continue
        diffs = s.diff().abs().dropna()
        if diffs.empty:
            continue
        local_max = float(diffs.max())
        if local_max > max_drop:
            max_drop = local_max
        if local_max > _DROP_THRESHOLD:
            drop_count += 1
    if drop_count >= _QUORUM:
        status = "confirmed"
    elif drop_count > 0:
        status = "suspect"
    else:
        status = "not_observed"
    return status, drop_count, max_drop


def _inspect_location(
    target_no: str, events: pd.DataFrame, location_path: Path
) -> list[EventQuality]:
    daily_max = _load_location_daily_max(location_path)
    events = events.sort_values(_DT_COL).reset_index(drop=True)
    results: list[EventQuality] = []
    prev_dt: pd.Timestamp | None = None
    for _, row in events.iterrows():
        event_dt: pd.Timestamp = row[_DT_COL]
        rt_hours = float(row["累積運転時間(Hour)"])
        status, drop_plugs, max_drop = _classify_event(daily_max, event_dt.to_pydatetime())
        cutoff = _cutoff_status(target_no, event_dt.to_pydatetime())
        gap_days = (
            int((event_dt - prev_dt).days) if prev_dt is not None else None
        )
        results.append(
            EventQuality(
                target_no=target_no,
                event_datetime=event_dt.strftime("%Y-%m-%d %H:%M"),
                event_date=event_dt.strftime("%Y-%m-%d"),
                cutoff_status=cutoff,
                confirmation=status,
                drop_plugs_in_window=drop_plugs,
                max_drop_voltage=max_drop,
                days_from_prev_event=gap_days,
                rt_at_event_hours=rt_hours,
            )
        )
        prev_dt = event_dt
    return results


def _summarize_per_location(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("target_no")
    summary = pd.DataFrame(
        {
            "events_total": grp.size(),
            "events_before_cutoff": grp.apply(
                lambda g: int((g["cutoff_status"] == "before_cutoff").sum())
            ),
            "events_after_cutoff_effective": grp.apply(
                lambda g: int((g["cutoff_status"] != "before_cutoff").sum())
            ),
            "confirmed": grp.apply(
                lambda g: int((g["confirmation"] == "confirmed").sum())
            ),
            "suspect": grp.apply(
                lambda g: int((g["confirmation"] == "suspect").sum())
            ),
            "not_observed": grp.apply(
                lambda g: int((g["confirmation"] == "not_observed").sum())
            ),
            "data_missing": grp.apply(
                lambda g: int((g["confirmation"] == "data_missing").sum())
            ),
            "min_gap_days": grp["days_from_prev_event"].min(),
            "median_gap_days": grp["days_from_prev_event"].median(),
            "max_gap_days": grp["days_from_prev_event"].max(),
        }
    )
    return summary.reset_index()


def _format_cell(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if pd.isna(v):
            return "-"
        return f"{v:.2f}"
    if pd.isna(v):
        return "-"
    return str(v)


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _write_report(
    events_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    missing_locations: list[str],
    out_md: Path,
) -> None:
    status_counts = Counter(events_df["confirmation"])
    cutoff_counts = Counter(events_df["cutoff_status"])
    total = len(events_df)

    # after_cutoff のみを分母にしたラベル品質
    after_df = events_df[events_df["cutoff_status"] != "before_cutoff"]
    after_total = len(after_df)
    after_counts = Counter(after_df["confirmation"])

    lines: list[str] = []
    lines.append("# legacy `exchange_timings_summary4.csv` ラベル品質調査\n")
    lines.append(
        "- 生成: `migration/inspect_legacy_exchange_labels.py`\n"
        "- 入力: `D:/code_commit/EP-VoltPredictor/ep_voltpredictor/exchange_timings_summary4.csv`\n"
        f"- 照合データ: `{_INPUT_DIR}`\n"
        f"- 照合ウィンドウ: イベント日時 ± {_WINDOW_DAYS} 日\n"
        f"- 急落しきい値: プラグ別日次最大の `abs(diff) > {_DROP_THRESHOLD}`\n"
        f"- quorum: 窓内同時急落プラグ数 >= {_QUORUM} → confirmed\n"
        f"- 運転中判定: `発電機電力 > 0`（ADR-013 C-1）\n"
        f"- ADR-012 カットオフ前のイベントは `before_cutoff` として分離\n\n"
    )

    lines.append("## 全体サマリー\n")
    lines.append(f"- 総イベント数: {total}")
    lines.append(f"- `before_cutoff`（学習対象外）: {cutoff_counts.get('before_cutoff', 0)}")
    lines.append(f"- `after_cutoff` + `no_cutoff`（学習対象）: {after_total}")
    lines.append("")
    lines.append("### 学習対象イベントの分類（ラベル品質の主要指標）\n")
    lines.append(
        f"- confirmed (quorum >= {_QUORUM}): {after_counts.get('confirmed', 0)} "
        f"({after_counts.get('confirmed', 0) / after_total:.1%})"
        if after_total
        else "- 学習対象イベント無し"
    )
    if after_total:
        lines.append(
            f"- suspect (1 <= drop plugs < {_QUORUM}): {after_counts.get('suspect', 0)} "
            f"({after_counts.get('suspect', 0) / after_total:.1%})"
        )
        lines.append(
            f"- not_observed (drop plugs = 0): {after_counts.get('not_observed', 0)} "
            f"({after_counts.get('not_observed', 0) / after_total:.1%})"
        )
        lines.append(
            f"- data_missing (窓内にデータ無し): {after_counts.get('data_missing', 0)}"
        )
    lines.append("")

    lines.append("## 機場別サマリー\n")
    lines.append(_to_markdown_table(summary_df))
    lines.append("")

    suspects_df = events_df[
        (events_df["cutoff_status"] != "before_cutoff")
        & (events_df["confirmation"].isin(["suspect", "not_observed"]))
    ].sort_values(["target_no", "event_datetime"])
    lines.append(f"\n## 要レビュー: suspect / not_observed ({len(suspects_df)} 件)\n")
    lines.append(
        "目視で「本当に交換か」「ラベル誤り／ノイズか」を判断してください。"
        "`suspect` はプラグ数が少ないので単独プラグの故障・センサー異常の疑い。"
        "`not_observed` は照合窓内に全く急落が見えない＝誤記録の可能性。\n"
    )
    if len(suspects_df):
        lines.append(_to_markdown_table(suspects_df))
    lines.append("")

    short_gap_df = events_df[
        (events_df["cutoff_status"] != "before_cutoff")
        & events_df["days_from_prev_event"].notna()
        & (events_df["days_from_prev_event"] < 60)
    ].sort_values(["target_no", "event_datetime"])
    lines.append(
        f"\n## 要レビュー: 交換間隔が短い（60 日未満、{len(short_gap_df)} 件）\n"
    )
    lines.append("プラグ寿命は通常数ヶ月〜数年想定。60 日未満は再調整/センサー入れ替え等の可能性。\n")
    if len(short_gap_df):
        lines.append(_to_markdown_table(short_gap_df))
    lines.append("")

    if missing_locations:
        lines.append(
            f"\n## 対応する現行 CSV が存在しない機場 ({len(missing_locations)} 件)\n"
        )
        for m in missing_locations:
            lines.append(f"- {m}")
        lines.append("")

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
    print(
        f"legacy CSV: {len(legacy)} events across {legacy['target_no'].nunique()} locations",
        file=sys.stderr,
    )

    all_results: list[EventQuality] = []
    missing: list[str] = []
    for target_no, events in legacy.groupby("target_no"):
        csv_path = _INPUT_DIR / f"{target_no}.csv"
        if not csv_path.exists():
            missing.append(target_no)
            print(f"  [{target_no}] SKIP: no input CSV", file=sys.stderr)
            continue
        try:
            results = _inspect_location(target_no, events, csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{target_no}] ERROR: {exc}", file=sys.stderr)
            continue
        all_results.extend(results)
        print(f"  [{target_no}] {len(results)} events inspected", file=sys.stderr)

    events_df = pd.DataFrame([asdict(r) for r in all_results])
    out_csv = _OUT_DIR / "legacy-label-quality.csv"
    events_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_df = _summarize_per_location(events_df)
    out_md = _OUT_DIR / "legacy-label-quality.md"
    _write_report(events_df, summary_df, missing, out_md)

    print(f"Wrote: {out_csv}", file=sys.stderr)
    print(f"Wrote: {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
