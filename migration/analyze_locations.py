"""機場選定材料を集計する一度きりのヘルパー（ADR-003 用）。

EP370G / EP400G の両方に対応。第 1 引数で切り替える（デフォルト EP400G）。

選定基準（decision-003-phase0-target-locations.md）:
- 累積運転時間が十分
- プラグ交換履歴が無い（もしくは少ない）
- 欠測率が低い

**要求電圧 <= 10 の行はアイドル扱いで無視**（発電機停止時のノイズ除外）。
全ての電圧系統計（`voltage_active_rate`, `exchange_candidates_*`）はアクティブ行のみで計算。

出力:
- migration/{model}_location_stats.csv  : 機械処理向け
- migration/{model}_location_stats.md   : ADR 記入・目視向け（rt_span 降順）

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

_OUT_DIR: Final[Path] = Path(__file__).parent

_DT_COL: Final[str] = "dailygraphpt_ptdatetime"
_MGMT_COL: Final[str] = "管理No"
_RT_COL: Final[str] = "累積運転時間"
_VOLTAGE_COLS_NO_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧{i}" for i in range(1, 7))
_VOLTAGE_COLS_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧_{i}" for i in range(1, 7))

# 要求電圧がこれ以下の行はアイドル扱いで無視
_IDLE_VOLTAGE_THRESHOLD: Final[float] = 10.0

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
    voltage_active_rate: float
    voltage_na_rate: float
    active_rows_per_plug: str
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
        any_active_mask = (voltage_df > _IDLE_VOLTAGE_THRESHOLD).any(axis=1)
        voltage_active_rate = float(any_active_mask.mean())
        voltage_na_rate = float(voltage_df.isna().any(axis=1).mean())
    else:
        voltage_active_rate = 0.0
        voltage_na_rate = 0.0

    active_rows: list[int] = []
    per_plug: list[int] = []
    for col in voltage_cols:
        col_series = voltage_df[col]
        active_mask = col_series > _IDLE_VOLTAGE_THRESHOLD
        active_rows.append(int(active_mask.sum()))
        # 急変検出はアクティブ行だけで計算（アイドル→動作の立ち上がりで誤検出しないため）
        active_only = col_series.where(active_mask)
        diff = active_only.diff().abs()
        count = int((diff > _EXCHANGE_JUMP_THRESHOLD).sum())
        per_plug.append(count)
    exchange_total = sum(per_plug)

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
        voltage_active_rate=voltage_active_rate,
        voltage_na_rate=voltage_na_rate,
        active_rows_per_plug=",".join(str(x) for x in active_rows),
        exchange_candidates_total=exchange_total,
        exchange_candidates_per_plug=",".join(str(x) for x in per_plug),
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


def _write_markdown(df: pd.DataFrame, out_path: Path, model_label: str, input_dir: Path) -> None:
    header = (
        f"# {model_label} {len(df)} 機場 選定材料（ADR-003 用）\n\n"
        f"- 生成: `migration/analyze_locations.py`\n"
        f"- 入力: `{input_dir}`\n"
        f"- 件数: {len(df)} 機場\n"
        f"- ソート: `rt_span` 降順（累積運転時間の伸びが大きい順）\n"
        f"- アイドル除外: 要求電圧 <= {_IDLE_VOLTAGE_THRESHOLD:.0f} の行は無視\n"
        f"- 交換候補しきい値: 要求電圧の 1 ステップ絶対値変化 > {_EXCHANGE_JUMP_THRESHOLD}\n"
        f"- `voltage_active_rate`: いずれかのプラグで要求電圧 > {_IDLE_VOLTAGE_THRESHOLD:.0f} の行率\n\n"
    )
    out_path.write_text(header + _to_markdown_table(df) + "\n", encoding="utf-8")


def main(model: str) -> int:
    model_key = model.lower()
    if model_key == "ep370g":
        input_dir = Path("E:/gene/input/EP370G_orig")
        model_label = "EP370G"
    elif model_key == "ep400g":
        input_dir = Path("E:/gene/input/EP400G_orig")
        model_label = "EP400G"
    else:
        print(f"Unknown model: {model}. Expected EP370G or EP400G.", file=sys.stderr)
        return 2

    out_csv = _OUT_DIR / f"{model_key}_location_stats.csv"
    out_md = _OUT_DIR / f"{model_key}_location_stats.md"

    csvs = sorted(input_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files under {input_dir}", file=sys.stderr)
        return 1

    print(f"Analyzing {len(csvs)} {model_label} files...", file=sys.stderr)
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
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    _write_markdown(df, out_md, model_label, input_dir)

    print(f"Wrote: {out_csv}", file=sys.stderr)
    print(f"Wrote: {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "EP400G"
    sys.exit(main(model_arg))
