"""DataRobot TS 用データセット組み立て CLI: cleaned CSV → TS 学習用日次 CSV。

処理順:
  aggregate_daily_max_voltage → add_features → build_ts_frame → CSV 出力
  （add_trend_features / add_future_7day_max_target は使わない）
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gene_plug_voltage_predictor.features.features import add_features
from gene_plug_voltage_predictor.features.target import aggregate_daily_max_voltage
from gene_plug_voltage_predictor.features.ts_dataset import build_ts_frame

_logger = logging.getLogger(__name__)


def build_dataset_ts(
    cleaned_df: pd.DataFrame,
    *,
    rated_kw: float = 370.0,
) -> pd.DataFrame:
    """cleaned CSV（30分粒度）から DataRobot TS 用日次データセットを生成する。"""
    daily = aggregate_daily_max_voltage(cleaned_df)
    daily = add_features(daily, cleaned_df, rated_kw=rated_kw)
    daily = build_ts_frame(daily)
    _logger.info("built TS dataset: %d rows, %d columns", len(daily), len(daily.columns))
    return daily


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build DataRobot TS training dataset from cleaned CSV"
    )
    ap.add_argument(
        "--cleaned-csv", required=True, type=Path,
        help="clean.py の出力 CSV（utf-8-sig）",
    )
    ap.add_argument(
        "--out", required=True, type=Path,
        help="DataRobot TS 投入用 CSV の出力先",
    )
    ap.add_argument(
        "--rated-kw", type=float, default=370.0,
        help="定格出力 kW（稼働割合の閾値 = rated_kw × 0.8, default: 370.0）",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cleaned_df = pd.read_csv(args.cleaned_csv, encoding="utf-8-sig")
    dataset = build_dataset_ts(cleaned_df, rated_kw=args.rated_kw)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.out} ({len(dataset)} rows, {len(dataset.columns)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
