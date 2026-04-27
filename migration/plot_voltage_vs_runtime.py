"""「要求電圧 vs 累積運転時間」散布図を機場ごとに PNG 生成（ADR-003 判断材料）。

- EP370G / EP400G の両モデルを CLI 引数で切り替え可能。
- 1 機場 = 1 PNG、6 プラグを 2x3 サブプロットで並べる。
- 横軸: 累積運転時間、縦軸: 要求電圧、点のサイズと透明度で密度を視認。
- 列名差（EP370G 形式の `要求電圧_1..6` / EP400G 形式の `要求電圧1..6`）は自動判別。
- **要求電圧 <= 10 の行はアイドル扱いで無視**（発電機停止時の 0 や低電圧ノイズを除外）。
- **EP370G は ADR-012 の機場別開始日時カットオフを適用**（5630/9290/9380/9381/9690）。
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd

_DT_COL: Final[str] = "dailygraphpt_ptdatetime"
_MGMT_COL: Final[str] = "管理No"
_RT_COL: Final[str] = "累積運転時間"
_VOLTAGE_COLS_NO_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧{i}" for i in range(1, 7))
_VOLTAGE_COLS_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧_{i}" for i in range(1, 7))

# 要求電圧がこれ以下の行はアイドル扱いで無視する（発電機停止時のノイズ除外）
_IDLE_VOLTAGE_THRESHOLD: Final[float] = 10.0

# ADR-012: EP370G 機場別の開始日時カットオフ（これより前の行は除外）
_EP370G_START_DATETIMES: Final[dict[str, datetime]] = {
    "5630": datetime(2022, 6, 26, 0, 30),
    "9290": datetime(2022, 7, 21, 10, 0),
    "9380": datetime(2023, 5, 31, 0, 30),
    "9381": datetime(2023, 5, 31, 14, 0),
    "9690": datetime(2024, 3, 19, 8, 30),
}


def _resolve_voltage_cols(path: Path) -> tuple[str, ...]:
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    available = set(head.columns)
    if set(_VOLTAGE_COLS_NO_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_NO_UNDERSCORE
    if set(_VOLTAGE_COLS_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_UNDERSCORE
    raise ValueError(f"No complete 要求電圧 column set found in {path.name}")


def _plot_one(
    path: Path,
    out_path: Path,
    model_label: str,
    start_dt: datetime | None = None,
) -> None:
    voltage_cols = _resolve_voltage_cols(path)
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=[_MGMT_COL, _DT_COL, _RT_COL, *voltage_cols],
        dtype={_MGMT_COL: str},
    )
    df[_DT_COL] = pd.to_datetime(df[_DT_COL], errors="coerce")
    rows_before_cutoff = len(df)
    if start_dt is not None:
        df = df[df[_DT_COL] >= start_dt].reset_index(drop=True)
    rows_excluded = rows_before_cutoff - len(df)
    mgmt_no = str(df[_MGMT_COL].iloc[0]) if len(df) else path.stem

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    active_series: list[pd.Series] = []
    for c in voltage_cols:
        s = df[c]
        active_series.append(s[s > _IDLE_VOLTAGE_THRESHOLD])
    voltage_concat = pd.concat(active_series) if active_series else pd.Series(dtype=float)

    if len(voltage_concat):
        q_lo = float(voltage_concat.quantile(0.01))
        q_hi = float(voltage_concat.quantile(0.99))
    else:
        q_lo, q_hi = 0.0, 1.0
    if q_hi <= q_lo:
        q_lo, q_hi = 0.0, max(1.0, q_hi)
    margin = (q_hi - q_lo) * 0.1
    ylim = (q_lo - margin, q_hi + margin)

    rt = df[_RT_COL]
    total_active = 0
    for i, col in enumerate(voltage_cols):
        ax = axes_flat[i]
        y = df[col]
        active_mask = y > _IDLE_VOLTAGE_THRESHOLD
        n_active = int(active_mask.sum())
        total_active += n_active
        n_out = int(((y[active_mask] < ylim[0]) | (y[active_mask] > ylim[1])).sum())
        ax.scatter(rt[active_mask], y[active_mask], s=2, alpha=0.3, color="steelblue")
        ax.set_ylim(ylim)
        plug_no = i + 1
        title = f"Plug {plug_no}  active={n_active}"
        if n_out:
            title += f"  clipped={n_out}"
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        if i >= 3:
            ax.set_xlabel("cumulative runtime")
        if i % 3 == 0:
            ax.set_ylabel("voltage")

    dt_start = df[_DT_COL].min()
    dt_end = df[_DT_COL].max()
    dt_range = ""
    if pd.notna(dt_start) and pd.notna(dt_end):
        dt_range = f"{dt_start:%Y-%m-%d} - {dt_end:%Y-%m-%d}"
    cutoff_note = ""
    if start_dt is not None:
        cutoff_note = f"  after {start_dt:%Y-%m-%d %H:%M} (excluded={rows_excluded})"
    fig.suptitle(
        f"{model_label} target_no={path.stem}  mgmt_no={mgmt_no}  rows={len(df)}  "
        f"active={total_active}  {dt_range}  (voltage>{_IDLE_VOLTAGE_THRESHOLD:.0f}){cutoff_note}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main(model: str) -> int:
    model_key = model.lower()
    if model_key == "ep370g":
        input_dir = Path("E:/gene/input/EP370G_orig")
        out_dir = Path(__file__).parent / "plots" / "ep370g"
        model_label = "EP370G"
        start_map: dict[str, datetime] = dict(_EP370G_START_DATETIMES)
    elif model_key == "ep400g":
        input_dir = Path("E:/gene/input/EP400G_orig")
        out_dir = Path(__file__).parent / "plots" / "ep400g"
        model_label = "EP400G"
        start_map = {}
    else:
        print(f"Unknown model: {model}. Expected EP370G or EP400G.", file=sys.stderr)
        return 2

    csvs = sorted(input_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files under {input_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plotting {len(csvs)} {model_label} files into {out_dir}...", file=sys.stderr)
    failed: list[tuple[str, str]] = []
    for i, path in enumerate(csvs, 1):
        out_path = out_dir / f"{path.stem}.png"
        start_dt = start_map.get(path.stem)
        try:
            _plot_one(path, out_path, model_label, start_dt=start_dt)
        except Exception as exc:
            failed.append((path.name, str(exc)))
            print(f"  [{i}/{len(csvs)}] {path.name}: ERROR {exc}", file=sys.stderr)
            continue
        if i % 10 == 0 or i == len(csvs):
            print(f"  [{i}/{len(csvs)}] {path.name}", file=sys.stderr)

    print(f"Done. {len(csvs) - len(failed)}/{len(csvs)} succeeded.", file=sys.stderr)
    if failed:
        print("Failures:", file=sys.stderr)
        for name, msg in failed:
            print(f"  {name}: {msg}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "EP400G"
    sys.exit(main(model_arg))
