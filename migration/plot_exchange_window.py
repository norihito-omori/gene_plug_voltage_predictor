"""legacy 交換イベント周辺（± N 日）の日次最大電圧を目視確認するプロット。

Phase 0.5 の続き。`inspect_legacy_exchange_labels.py` の集計で 89.9% が
`not_observed` となった原因を目視で確認する（Plan A）。

1 機場 = 1 PNG、各イベント日時 ± `_WINDOW_DAYS` 日のプラグ別日次最大電圧を
6 プラグ重ね書きし、イベント日時を垂直線で示す。target_no は CLI 引数で指定。

- 運転中判定: `発電機電力 > 0`（ADR-013 C-1、`inspect_legacy_exchange_labels.py` と対称）
- 日次最大: `inspect_legacy_exchange_labels.py` と同じ集約ロジック
- ADR-012 カットオフは考慮しつつ、イベント自体が before_cutoff でもプロット対象に含める
  （目視確認のためであり、学習対象フィルタではない）

用途: しきい値・quorum・窓幅の妥当性を目で判断してから `decision-014` の校正に進む。
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd

_WINDOW_DAYS: Final[int] = 30  # イベント前後 ± 30 日を表示（inspect の ± 7 日より広め）

_DT_COL: Final[str] = "dailygraphpt_ptdatetime"
_POWER_COL: Final[str] = "発電機電力"
_VOLTAGE_COLS_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧_{i}" for i in range(1, 7))
_VOLTAGE_COLS_NO_UNDERSCORE: Final[tuple[str, ...]] = tuple(f"要求電圧{i}" for i in range(1, 7))

_LEGACY_CSV: Final[Path] = Path(
    "D:/code_commit/EP-VoltPredictor/ep_voltpredictor/exchange_timings_summary4.csv"
)
_INPUT_DIR: Final[Path] = Path("E:/gene/input/EP370G_orig")
_OUT_DIR: Final[Path] = Path(__file__).parent / "plots" / "exchange_windows"

_PLUG_COLORS: Final[tuple[str, ...]] = (
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
)


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


def _plot_location(target_no: str, events: pd.DataFrame, location_path: Path) -> Path:
    daily_max = _load_location_daily_max(location_path)
    events = events.sort_values(_DT_COL).reset_index(drop=True)
    n_events = len(events)
    if n_events == 0:
        raise ValueError(f"No events for {target_no}")

    n_cols = min(2, n_events)
    n_rows = (n_events + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.5 * n_rows), squeeze=False)

    plug_cols = [f"plug_{i}" for i in range(1, 7)]
    for idx, row in events.iterrows():
        event_dt: pd.Timestamp = row[_DT_COL]
        event_date = event_dt.normalize()
        start = event_date - pd.Timedelta(days=_WINDOW_DAYS)
        end = event_date + pd.Timedelta(days=_WINDOW_DAYS)
        window = daily_max[(daily_max["date"] >= start) & (daily_max["date"] <= end)].copy()

        ax = axes[idx // n_cols][idx % n_cols]
        if window.empty:
            ax.text(0.5, 0.5, "no data in window", ha="center", va="center", transform=ax.transAxes)
        else:
            for i, col in enumerate(plug_cols):
                s = window[col].dropna()
                if s.empty:
                    continue
                plug_dates = window.loc[s.index, "date"]
                ax.plot(plug_dates, s.values, marker="o", markersize=3, linewidth=1,
                        color=_PLUG_COLORS[i], label=f"Plug {i+1}")
        ax.axvline(event_date, color="black", linestyle="--", linewidth=1, alpha=0.6)
        # ± 7 日（inspect 窓）も薄く
        ax.axvspan(event_date - pd.Timedelta(days=7), event_date + pd.Timedelta(days=7),
                   color="gray", alpha=0.1)
        ax.set_title(
            f"{event_dt:%Y-%m-%d %H:%M}  rt={row['累積運転時間(Hour)']:.0f}h",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7, ncol=2)

    # 使わないサブプロットを消す
    for idx in range(n_events, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(
        f"target_no={target_no}  legacy exchange events (± {_WINDOW_DAYS} days)  "
        f"gray band = ± 7 day inspect window",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = _OUT_DIR / f"{target_no}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
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

    if not target_nos:
        target_nos = sorted(legacy["target_no"].unique().tolist())

    for target_no in target_nos:
        events = legacy[legacy["target_no"] == target_no]
        if events.empty:
            print(f"[{target_no}] no events", file=sys.stderr)
            continue
        csv_path = _INPUT_DIR / f"{target_no}.csv"
        if not csv_path.exists():
            print(f"[{target_no}] no input CSV: {csv_path}", file=sys.stderr)
            continue
        try:
            out = _plot_location(target_no, events, csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[{target_no}] ERROR: {exc}", file=sys.stderr)
            continue
        print(f"[{target_no}] {len(events)} events -> {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
