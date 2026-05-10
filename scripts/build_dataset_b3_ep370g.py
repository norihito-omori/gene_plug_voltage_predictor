"""
B3-c 用 dataset 構築 — DataRobot TS 二値分類「14日以内越境」+ 加速度特徴量

入力: outputs/cleaned_ep370g_v5.csv, outputs/dataset_ep370g_ts_v7.csv
出力: outputs/dataset_ep370g_ts_binary_within14d_v1.csv

設計:
  - target: will_cross_within_14d = 1〜14日後 (T_MIN=1〜T_MAX=14) に baseline+16 越境
  - forecast_window: 1〜7（DR が 1〜7 日先の越境確率を出力。target は同日に貼る）
  - 加速度特徴量を追加: rate{11..14}, rrate{11..14}_w{3,5,7}, slope{11..14}_w{3,5,7}
  - 越境後 (該当 gen の越境日以降) の行は除外
  - 評価: 学習後、各日のスコアから「アラート発火日」を求めて timely%/too_early/missed を計算
"""
from __future__ import annotations
import pandas as pd
import numpy as np

CLEANED = "outputs/cleaned_ep370g_v5.csv"
DAILY = "outputs/dataset_ep370g_ts_v7.csv"
OUT = "outputs/dataset_ep370g_ts_binary_within14d_v1.csv"

LEVELS = [11, 12, 13, 14]
WINDOWS = [3, 5, 7]
MIN_OP_HOURS = 1.0
T_MIN, T_MAX = 1, 14


def _slope(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(valid), dtype=float)
    return np.polyfit(x, valid, 1)[0]


def main():
    print("Loading cleaned...", flush=True)
    cleaned_raw = pd.read_csv(CLEANED, encoding="utf-8-sig", nrows=0)
    cols = cleaned_raw.columns.tolist()
    power_col, voltage_col, plug_id_col, datetime_col = cols[3], cols[7], cols[9], cols[1]

    cleaned = pd.read_csv(
        CLEANED, encoding="utf-8-sig",
        usecols=[datetime_col, power_col, voltage_col, plug_id_col, "gen_no", "baseline"],
    )
    cleaned["_date"] = pd.to_datetime(cleaned[datetime_col]).dt.normalize()
    running = cleaned[cleaned[power_col] > 0].copy()

    print("Computing rate features...", flush=True)
    for n in LEVELS:
        running[f"_at{n}"] = (running[voltage_col] == (running["baseline"].round() + n)).astype(float)
    running["_slot"] = 1.0
    agg = running.groupby([plug_id_col, "_date"]).agg(
        op_hours=("_slot", "sum"),
        **{f"_at{n}": (f"_at{n}", "sum") for n in LEVELS},
    ).mul(0.5).reset_index().rename(columns={"_date": "date"})
    agg["date"] = pd.to_datetime(agg["date"])
    for n in LEVELS:
        agg[f"rate{n}"] = np.where(
            agg["op_hours"] >= MIN_OP_HOURS,
            agg[f"_at{n}"] / agg["op_hours"],
            np.nan,
        )

    print("Loading daily dataset...", flush=True)
    daily = pd.read_csv(DAILY, encoding="utf-8-sig", parse_dates=["date"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily_plug_col = "管理No_プラグNo"

    print("Merging rate features into daily...", flush=True)
    rate_cols = ["op_hours"] + [f"rate{n}" for n in LEVELS]
    daily = daily.merge(
        agg[[plug_id_col, "date"] + rate_cols],
        left_on=[daily_plug_col, "date"], right_on=[plug_id_col, "date"], how="left",
    )
    if plug_id_col != daily_plug_col:
        daily = daily.drop(columns=[plug_id_col])

    print("Computing rolling/slope features...", flush=True)
    daily = daily.sort_values([daily_plug_col, "date"]).reset_index(drop=True)
    for n in LEVELS:
        for w in WINDOWS:
            daily[f"rrate{n}_w{w}"] = (
                daily.groupby(daily_plug_col)[f"rate{n}"]
                .transform(lambda s: s.rolling(w, min_periods=1).mean())
            )
            daily[f"slope{n}_w{w}"] = (
                daily.groupby(daily_plug_col)[f"rate{n}"]
                .transform(lambda s: s.rolling(w, min_periods=2).apply(_slope, raw=True))
            )

    # 越境イベント
    print("Building target (will_cross_within_14d)...", flush=True)
    daily["machine_id"] = daily[daily_plug_col].str.rsplit("_", n=1).str[0]
    daily["cross_thr"] = daily["baseline"] + 16

    op_only = daily[(daily["is_operating"] == 1) & (daily["daily_max"] >= daily["cross_thr"])].copy()
    cross = (
        op_only.groupby(["machine_id", "gen_no"])["date"].min()
        .reset_index().rename(columns={"date": "first_crossing_date"})
    )
    daily = daily.merge(cross, on=["machine_id", "gen_no"], how="left")
    daily["days_to_cross"] = (daily["first_crossing_date"] - daily["date"]).dt.days
    daily["will_cross_within_14d"] = (
        (daily["days_to_cross"] >= T_MIN) & (daily["days_to_cross"] <= T_MAX)
    ).astype(int)

    print("Excluding post-crossing rows...", flush=True)
    pre_cross_mask = (
        daily["first_crossing_date"].isna()
        | (daily["date"] < daily["first_crossing_date"])
    )
    print(f"  total rows: {len(daily)}, pre-crossing: {pre_cross_mask.sum()}", flush=True)
    daily = daily[pre_cross_mask].copy()

    drop_cols = ["machine_id", "cross_thr", "first_crossing_date", "days_to_cross", "op_hours"]
    daily = daily.drop(columns=[c for c in drop_cols if c in daily.columns])

    print(f"\nFinal: {len(daily)} rows, {len(daily.columns)} cols")
    pos = int(daily["will_cross_within_14d"].sum())
    print(f"will_cross_within_14d positive count: {pos} / {len(daily)} ({pos/len(daily)*100:.2f}%)")
    print(f"\nColumns: {list(daily.columns)}")

    daily.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
