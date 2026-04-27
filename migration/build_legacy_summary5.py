"""legacy 交換記録 `exchange_timings_summary4.csv` に対し、2026-04-28 大森氏の
目視確認結果を反映して `exchange_timings_summary5.csv` を生成する。

**反映内容（ソース: `issues/2026-04-28-legacy-exchange-record-*.md`）**

新規追加（missing-candidates、大森氏確認で交換確定）:
- 5630: 2025-12-22 を追加
- 9110: 2025-05-31 を追加
- 9110: 2025-09-04 を追加（ただし 2025-10-06 の既存 legacy を削除、同一イベントの誤記）
- 9240: 2022-09-22 を追加
- 9720: 2025-05-19 を追加

日付訂正（corrections、FP 側が正解日）:
- 9110: 2025-10-06 を削除（上記で 2025-09-04 に置換された）
- 9111: 2023-08-16 → 2023-08-08 に変更
- 9720: 2024-11-12 → 2024-11-01 に変更
- 9720: 2025-06-02 を削除（誤記、交換実績なし）

**各新規エントリの 累積運転時間(Hour) 値の求め方**

検出日を normalized date として、元 CSV の当日最後の running 行の
`累積運転時間` raw 値を `runtime_utils.unpack_to_hours` でアンパックした値を採用。
datetime はその行の `dailygraphpt_ptdatetime` をそのまま採用する。

出力は `data/exchange_timings_summary5.csv`。ADR-014 再校正時の input として使う。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from runtime_utils import unpack_to_hours  # noqa: E402

_DT_COL: Final[str] = "dailygraphpt_ptdatetime"
_POWER_COL: Final[str] = "発電機電力"
_RUNTIME_COL: Final[str] = "累積運転時間"
_RUNTIME_OUT_COL: Final[str] = "累積運転時間(Hour)"

_LEGACY_IN: Final[Path] = Path(
    "D:/code_commit/EP-VoltPredictor/ep_voltpredictor/exchange_timings_summary4.csv"
)
_LEGACY_OUT: Final[Path] = Path(
    "D:/code_commit/EP-VoltPredictor/ep_voltpredictor/exchange_timings_summary5.csv"
)
_INPUT_DIR: Final[Path] = Path("E:/gene/input/EP370G_orig")


@dataclass(frozen=True)
class AddEntry:
    target_no: str
    event_date: date
    note: str


@dataclass(frozen=True)
class RemoveEntry:
    target_no: str
    legacy_date: date
    note: str


@dataclass(frozen=True)
class ChangeEntry:
    target_no: str
    from_date: date
    to_date: date
    note: str


# 2026-04-28 大森氏確認結果に基づく legacy 更新定義
_ADDITIONS: Final[tuple[AddEntry, ...]] = (
    AddEntry("5630", date(2025, 12, 22), "missing-candidates #1"),
    AddEntry("9110", date(2025, 5, 31), "missing-candidates #2"),
    AddEntry("9240", date(2022, 9, 22), "missing-candidates #4"),
    AddEntry("9720", date(2025, 5, 19), "missing-candidates #5"),
)

# 9110 の 2025-09-04 追加と 2025-10-06 削除は「同一イベントの誤記」なので ChangeEntry に集約
_CHANGES: Final[tuple[ChangeEntry, ...]] = (
    ChangeEntry("9110", date(2025, 10, 6), date(2025, 9, 4),
                "corrections: 10-06 → 09-04 (真の交換日)"),
    ChangeEntry("9111", date(2023, 8, 16), date(2023, 8, 8),
                "corrections: 08-16 → 08-08 (真の交換日)"),
    ChangeEntry("9720", date(2024, 11, 12), date(2024, 11, 1),
                "corrections: 11-12 → 11-01 (真の交換日)"),
)

_REMOVALS: Final[tuple[RemoveEntry, ...]] = (
    RemoveEntry("9720", date(2025, 6, 2),
                "corrections: 誤記、交換実績なし（#5 の 05-19 と別枠判定）"),
)


def _resolve_voltage_cols_and_runtime(path: Path) -> str:
    """runtime column を解決。本 CSV は累積運転時間 raw 値列。"""
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    if _RUNTIME_COL not in head.columns:
        raise ValueError(f"{_RUNTIME_COL} 列が見つからない: {path.name}")
    return _RUNTIME_COL


def _lookup_event_details(
    target_no: str, event_date: date
) -> tuple[str, float] | None:
    """指定 target の指定日における最終 running 行の datetime と runtime(hours) を返す。

    当日 running 行が無ければ None。
    """
    path = _INPUT_DIR / f"{target_no}.csv"
    runtime_col = _resolve_voltage_cols_and_runtime(path)
    df = pd.read_csv(
        path, encoding="utf-8-sig",
        usecols=[_DT_COL, _POWER_COL, runtime_col],
    )
    df[_DT_COL] = pd.to_datetime(df[_DT_COL], errors="coerce")
    df = df.dropna(subset=[_DT_COL])
    running = df[df[_POWER_COL] > 0]
    day_mask = running[_DT_COL].dt.normalize() == pd.Timestamp(event_date)
    day = running[day_mask]
    if day.empty:
        return None
    last = day.sort_values(_DT_COL).iloc[-1]
    dt_str = last[_DT_COL].strftime("%Y/%-m/%-d %-H:%M") if sys.platform != "win32" \
        else f"{last[_DT_COL].year}/{last[_DT_COL].month}/{last[_DT_COL].day} " \
             f"{last[_DT_COL].hour}:{last[_DT_COL].minute:02d}"
    runtime_hours = float(unpack_to_hours(pd.Series([last[runtime_col]])).iloc[0])
    return dt_str, runtime_hours


def _find_nearest_running_day(
    target_no: str, event_date: date, max_search_days: int = 7
) -> tuple[str, float, date] | None:
    """当日に running 行が無い場合、前後 max_search_days 以内で最も近い running 日を探す。"""
    for delta in range(max_search_days + 1):
        for sign in (0, +1, -1) if delta > 0 else (0,):
            try_date = pd.Timestamp(event_date) + pd.Timedelta(days=sign * delta)
            r = _lookup_event_details(target_no, try_date.date())
            if r is not None:
                return r[0], r[1], try_date.date()
    return None


def main() -> int:
    if not _LEGACY_IN.exists():
        print(f"legacy 入力 CSV が無い: {_LEGACY_IN}", file=sys.stderr)
        return 1
    if not _INPUT_DIR.exists():
        print(f"入力 CSV ディレクトリが無い: {_INPUT_DIR}", file=sys.stderr)
        return 1

    df = pd.read_csv(_LEGACY_IN, encoding="utf-8-sig")
    # 列名の絶対参照（実体は 累積運転時間(Hour)）
    runtime_out_col_actual = df.columns[1]  # 2 列目
    df.rename(columns={runtime_out_col_actual: _RUNTIME_OUT_COL}, inplace=True)
    df["__dt"] = pd.to_datetime(df[_DT_COL], errors="coerce")
    df["target_no"] = df["file_name"].str.replace(".csv", "", regex=False)
    print(f"read legacy: {len(df)} rows")

    change_log: list[str] = []
    rows_to_drop: list[int] = []

    # 削除
    for rem in _REMOVALS:
        mask = (df["target_no"] == rem.target_no) & \
               (df["__dt"].dt.normalize() == pd.Timestamp(rem.legacy_date))
        idx = df.index[mask].tolist()
        if not idx:
            print(f"WARN: 削除対象行なし {rem.target_no} {rem.legacy_date}", file=sys.stderr)
            continue
        rows_to_drop.extend(idx)
        change_log.append(f"REMOVE {rem.target_no} {rem.legacy_date}  # {rem.note}")

    # 変更（既存行を削除 → 新日で追加扱い）
    for chg in _CHANGES:
        mask = (df["target_no"] == chg.target_no) & \
               (df["__dt"].dt.normalize() == pd.Timestamp(chg.from_date))
        idx = df.index[mask].tolist()
        if not idx:
            print(f"WARN: 変更対象行なし {chg.target_no} {chg.from_date}", file=sys.stderr)
            continue
        rows_to_drop.extend(idx)
        change_log.append(
            f"CHANGE {chg.target_no} {chg.from_date} → {chg.to_date}  # {chg.note}"
        )

    kept = df.drop(index=rows_to_drop).copy()

    # 追加分と変更先を集約
    new_rows: list[dict] = []
    add_targets: list[tuple[str, date, str]] = [
        (a.target_no, a.event_date, a.note) for a in _ADDITIONS
    ] + [
        (c.target_no, c.to_date, c.note) for c in _CHANGES
    ]

    for target_no, ev_date, note in add_targets:
        r = _lookup_event_details(target_no, ev_date)
        if r is None:
            near = _find_nearest_running_day(target_no, ev_date)
            if near is None:
                print(f"ERROR: {target_no} {ev_date} の running 行が見つからない",
                      file=sys.stderr)
                return 2
            dt_str, runtime_hours, actual_date = near
            print(f"  NOTE: {target_no} {ev_date} は running なし → {actual_date} を採用",
                  file=sys.stderr)
        else:
            dt_str, runtime_hours = r
        new_rows.append({
            _DT_COL: dt_str,
            _RUNTIME_OUT_COL: runtime_hours,
            "file_name": f"{target_no}.csv",
        })
        change_log.append(
            f"ADD    {target_no} {ev_date}  dt={dt_str}  runtime={runtime_hours:.6f}  # {note}"
        )

    # 出力用 DataFrame
    out_cols = [_DT_COL, _RUNTIME_OUT_COL, "file_name"]
    kept_out = kept[out_cols].copy()
    new_df = pd.DataFrame(new_rows, columns=out_cols)
    final = pd.concat([kept_out, new_df], ignore_index=True)

    # ソート: file_name → datetime（読みやすさのため）
    final["__sort_dt"] = pd.to_datetime(final[_DT_COL], errors="coerce")
    final = final.sort_values(["file_name", "__sort_dt"]).reset_index(drop=True)
    final = final.drop(columns="__sort_dt")

    final.to_csv(_LEGACY_OUT, index=False, encoding="utf-8-sig")
    print(f"\nwrote: {_LEGACY_OUT}  ({len(final)} rows)")

    print("\n==== change log ====")
    for line in change_log:
        print(f"  {line}")

    # 機場別 counts
    print("\n==== 機場別件数 (summary4 → summary5) ====")
    before = df.groupby("target_no").size()
    after = final["file_name"].str.replace(".csv", "", regex=False).value_counts()
    for t in sorted(set(before.index) | set(after.index)):
        b = int(before.get(t, 0))
        a = int(after.get(t, 0))
        diff = a - b
        mark = "" if diff == 0 else f"  ({diff:+d})"
        print(f"  {t}: {b} → {a}{mark}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
