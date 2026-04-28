"""データセット組み立て CLI: cleaned CSV → DataRobot 学習用日次 CSV。

処理順:
  aggregate_daily_max_voltage → add_features → add_trend_features
  → add_future_7day_max_target → NaN 除外（future_{horizon}d_max / baseline）→ CSV 出力
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gene_plug_voltage_predictor.features.features import add_features
from gene_plug_voltage_predictor.features.target import (
    add_future_7day_max_target,
    aggregate_daily_max_voltage,
)
from gene_plug_voltage_predictor.features.trend_features import add_trend_features

_logger = logging.getLogger(__name__)


def build_dataset(
    cleaned_df: pd.DataFrame,
    *,
    rated_kw: float = 370.0,
    horizon: int = 7,
) -> pd.DataFrame:
    daily = aggregate_daily_max_voltage(cleaned_df)
    daily = add_features(daily, cleaned_df, rated_kw=rated_kw)
    daily = add_trend_features(daily)
    daily = add_future_7day_max_target(daily, horizon=horizon)

    target_col = f"future_{horizon}d_max"
    before = len(daily)
    daily = daily.dropna(subset=[target_col, "baseline"]).reset_index(drop=True)
    dropped = before - len(daily)
    if dropped:
        _logger.info("dropped %d rows with NaN target or baseline", dropped)
    if len(daily) == 0:
        _logger.warning("dataset is empty after NaN removal")
    return daily


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build DataRobot training dataset from cleaned CSV"
    )
    ap.add_argument(
        "--cleaned-csv", required=True, type=Path,
        help="clean.py の出力 CSV（utf-8-sig）",
    )
    ap.add_argument(
        "--out", required=True, type=Path,
        help="DataRobot 投入用 CSV の出力先",
    )
    ap.add_argument(
        "--rated-kw", type=float, default=370.0,
        help="定格出力 kW（稼働割合の閾値 = rated_kw × 0.8, default: 370.0）",
    )
    ap.add_argument(
        "--horizon", type=int, default=7,
        help="予測ホライズン日数（default: 7）",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cleaned_df = pd.read_csv(args.cleaned_csv, encoding="utf-8-sig")
    dataset = build_dataset(cleaned_df, rated_kw=args.rated_kw, horizon=args.horizon)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.out} ({len(dataset)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
