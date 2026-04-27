"""EP400G 98 機場の「要求電圧 vs 累積運転時間」散布図を機場ごとに PNG 生成（ADR-003 判断材料）。

- 1 機場 = 1 PNG、6 プラグを 2x3 サブプロットで並べる。
- 横軸: 累積運転時間、縦軸: 要求電圧、点のサイズと透明度で密度を視認。
- 列名差（EP370G 形式の `要求電圧_1..6` / EP400G 形式の `要求電圧1..6`）は自動判別。
- 出力: `migration/plots/ep400g/{target_no}.png`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd

_INPUT_DIR: Final[Path] = Path("E:/gene/input/EP400G_orig")
_OUT_DIR: Final[Path] = Path(__file__).parent / "plots" / "ep400g"

_DT_COL: Final[str] = "dailygraphpt_ptdatetime"
_MGMT_COL: Final[str] = "管理No"
_RT_COL: Final[str] = "累積運転時間"
_VOLTAGE_COLS_NO_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧{i}" for i in range(1, 7))
_VOLTAGE_COLS_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧_{i}" for i in range(1, 7))


def _resolve_voltage_cols(path: Path) -> tuple[str, ...]:
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    available = set(head.columns)
    if set(_VOLTAGE_COLS_NO_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_NO_UNDERSCORE
    if set(_VOLTAGE_COLS_UNDERSCORE).issubset(available):
        return _VOLTAGE_COLS_UNDERSCORE
    raise ValueError(f"No complete 要求電圧 column set found in {path.name}")


def _plot_one(path: Path, out_path: Path) -> None:
    voltage_cols = _resolve_voltage_cols(path)
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=[_MGMT_COL, _DT_COL, _RT_COL, *voltage_cols],
        dtype={_MGMT_COL: str},
    )
    df[_DT_COL] = pd.to_datetime(df[_DT_COL], errors="coerce")
    mgmt_no = str(df[_MGMT_COL].iloc[0]) if len(df) else path.stem

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    voltage_concat = pd.concat([df[c] for c in voltage_cols])
    q_lo = float(voltage_concat.quantile(0.01))
    q_hi = float(voltage_concat.quantile(0.99))
    if q_hi <= q_lo:
        q_lo, q_hi = 0.0, max(1.0, float(voltage_concat.max() or 1.0))
    margin = (q_hi - q_lo) * 0.1
    ylim = (q_lo - margin, q_hi + margin)

    rt = df[_RT_COL]
    for i, col in enumerate(voltage_cols):
        ax = axes_flat[i]
        y = df[col]
        n_out = int(((y < ylim[0]) | (y > ylim[1])).sum())
        ax.scatter(rt, y, s=2, alpha=0.3, color="steelblue")
        ax.set_ylim(ylim)
        plug_no = i + 1
        title = f"Plug {plug_no}"
        if n_out:
            title += f"  (outliers clipped: {n_out})"
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
    fig.suptitle(
        f"EP400G target_no={path.stem}  mgmt_no={mgmt_no}  rows={len(df)}  {dt_range}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    csvs = sorted(_INPUT_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSV files under {_INPUT_DIR}", file=sys.stderr)
        return 1

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Plotting {len(csvs)} EP400G files into {_OUT_DIR}...", file=sys.stderr)
    failed: list[tuple[str, str]] = []
    for i, path in enumerate(csvs, 1):
        out_path = _OUT_DIR / f"{path.stem}.png"
        try:
            _plot_one(path, out_path)
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
    sys.exit(main())
