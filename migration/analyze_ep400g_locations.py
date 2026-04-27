"""EP400G 98 機場の選定材料を集計する一度きりのヘルパー（ADR-003 用）。

選定基準（decision-003-phase0-target-locations.md）:
- 累積運転時間が十分
- プラグ交換履歴が無い（もしくは少ない）
- 欠測率が低い

出力:
- migration/ep400g_location_stats.csv  : 機械処理向け
- migration/ep400g_location_stats.md   : ADR 記入・目視向け（rt_span 降順）

注:
- プラグ交換履歴は legacy `exchange_timings_summary4.csv` が EP370G しかカバー
  していないため、EP400G はヒューリスティクス（要求電圧の急変検出）で候補数のみ推定。
- 絶対的な運転時間の単位は不明（file 間で桁が揃っていれば比較材料として十分）。
- 人間判断のための一覧化であり、推薦や機場選定は行わない（ガードレール 5.6）。
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd

_INPUT_DIR: Final[Path] = Path("E:/gene/input/EP400G_orig")
_OUT_DIR: Final[Path] = Path(__file__).parent
_OUT_CSV: Final[Path] = _OUT_DIR / "ep400g_location_stats.csv"
_OUT_MD: Final[Path] = _OUT_DIR / "ep400g_location_stats.md"

_DT_COL: Final[str] = "dailygraphpt_ptdatetime"
_MGMT_COL: Final[str] = "管理No"
_RT_COL: Final[str] = "累積運転時間"
_VOLTAGE_COLS_NO_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧{i}" for i in range(1, 7))
_VOLTAGE_COLS_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧_{i}" for i in range(1, 7))

# 要求電圧の急変しきい値（1 ステップで絶対値がこれ以上動いたら交換候補とみなす）
_EXCHANGE_JUMP_THRESHOLD: Final[int] = 20


@dataclass
class LocationStats:
    target_no: str
    mgmt_no: str
    rows: int
    dt_start: str
    dt_end: str
    span_days: int
    rt_min: float
    rt_max: float
    rt_span: float
    rt_all_zero: bool
    voltage_all_zero_rate: float
    voltage_na_rate: float
    exchange_candidates_total: int
    exchange_candidates_per_plug: str


def _resolve_voltage_cols(path: Path) -> tuple[str, ...]:
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    available = set(head.columns)
    if set(_VOLTAGE_COLS_NO_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_NO_UNDERSCORE
    if set(_VOLTAGE_COLS_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_UNDERSCORE
    raise ValueError(f"No complete 要求電圧 column set found in {path.name}")


def _analyze_one(path: Path) -> LocationStats:
    voltage_cols = _resolve_voltage_cols(path)
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=[_MGMT_COL, _DT_COL, _RT_COL, *voltage_cols],
        dtype={_MGMT_COL: str},
    )
    df[_DT_COL] = pd.to_datetime(df[_DT_COL], errors="coerce")
    rows = len(df)
    mgmt_no = str(df[_MGMT_COL].iloc[0]) if rows else ""

    dt_series = df[_DT_COL].dropna()
    dt_start = dt_series.min().strftime("%Y-%m-%d") if len(dt_series) else ""
    dt_end = dt_series.max().strftime("%Y-%m-%d") if len(dt_series) else ""
    span_days = (dt_series.max() - dt_series.min()).days if len(dt_series) else 0

    rt = df[_RT_COL]
    rt_min = float(rt.min()) if rows else 0.0
    rt_max = float(rt.max()) if rows else 0.0
    rt_span = rt_max - rt_min
    rt_all_zero = bool((rt == 0).all())

    voltage_df = df[list(voltage_cols)]
    if rows:
        all_zero_mask = (voltage_df.fillna(0) == 0).all(axis=1)
        voltage_all_zero_rate = float(all_zero_mask.mean())
        voltage_na_rate = float(voltage_df.isna().any(axis=1).mean())
    else:
        voltage_all_zero_rate = 0.0
        voltage_na_rate = 0.0

    per_plug: list[int] = []
    for col in voltage_cols:
        diff = voltage_df[col].diff().abs()
        count = int((diff > _EXCHANGE_JUMP_THRESHOLD).sum())
        per_plug.append(count)
    exchange_total = sum(per_plug)
    exchange_per_plug = ",".join(str(x) for x in per_plug)

    return LocationStats(
        target_no=path.stem,
        mgmt_no=mgmt_no,
        rows=rows,
        dt_start=dt_start,
        dt_end=dt_end,
        span_days=span_days,
        rt_min=rt_min,
        rt_max=rt_max,
        rt_span=rt_span,
        rt_all_zero=rt_all_zero,
        voltage_all_zero_rate=voltage_all_zero_rate,
        voltage_na_rate=voltage_na_rate,
        exchange_candidates_total=exchange_total,
        exchange_candidates_per_plug=exchange_per_plug,
    )


def _format_cell(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    header = (
        "# EP400G 98 機場 選定材料（ADR-003 用）\n\n"
        f"- 生成: `migration/analyze_ep400g_locations.py`\n"
        f"- 入力: `{_INPUT_DIR}`\n"
        f"- 件数: {len(df)} 機場\n"
        f"- ソート: `rt_span` 降順（累積運転時間の伸びが大きい順）\n"
        f"- 交換候補しきい値: 要求電圧の 1 ステップ絶対値変化 > {_EXCHANGE_JUMP_THRESHOLD}\n"
        f"- 欠測率定義: `voltage_all_zero_rate` = 6 プラグの要求電圧が全て 0 の行率\n\n"
    )
    out_path.write_text(header + _to_markdown_table(df) + "\n", encoding="utf-8")


def main() -> int:
    csvs = sorted(_INPUT_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSV files under {_INPUT_DIR}", file=sys.stderr)
        return 1

    print(f"Analyzing {len(csvs)} EP400G files...", file=sys.stderr)
    rows: list[dict[str, object]] = []
    for i, path in enumerate(csvs, 1):
        try:
            stats = _analyze_one(path)
            rows.append(asdict(stats))
        except Exception as exc:
            print(f"  [{i}/{len(csvs)}] {path.name}: ERROR {exc}", file=sys.stderr)
            continue
        if i % 10 == 0 or i == len(csvs):
            print(f"  [{i}/{len(csvs)}] {path.name}", file=sys.stderr)

    df = pd.DataFrame(rows)
    df = df.sort_values("rt_span", ascending=False).reset_index(drop=True)
    df.to_csv(_OUT_CSV, index=False, encoding="utf-8-sig")
    _write_markdown(df, _OUT_MD)

    print(f"Wrote: {_OUT_CSV}", file=sys.stderr)
    print(f"Wrote: {_OUT_MD}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
