"""
ルールベース (hours_at_31kV 7日累計 > threshold) vs DataRobot v5/v6 の比較評価。

使い方:
  python scripts/evaluate_rule_vs_dr.py                  # v5 と比較
  python scripts/evaluate_rule_vs_dr.py --dr-pred outputs/ep370g_ts_v6_holdout_pred.csv --label v6

評価指標:
  - 超過前に警告できた台数 / 20台 (recall)
  - 平均先行日数 (avg lead days)
  - 偽陽性警告数 (no-crossing machines warned)
"""
from __future__ import annotations

import argparse
import pandas as pd

THRESHOLD_CROSS = 32.0
HOLDOUT_START = "2024-04-01"
HOLDOUT_END   = "2025-03-31"
RULE_THRESHOLDS = [2, 3, 5, 7, 10, 14, 21]  # 7-day sum thresholds to sweep

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dr-pred", default="outputs/ep370g_ts_v5_holdout_pred.csv")
parser.add_argument("--label", default="v5")
args = parser.parse_args()

dataset = pd.read_csv("outputs/dataset_ep370g_ts_v6.csv", encoding="utf-8-sig",
                      parse_dates=["date"])
dataset["machine_id"] = dataset["管理No_プラグNo"].str.rsplit("_", n=1).str[0]

holdout = dataset[dataset["date"].between(HOLDOUT_START, HOLDOUT_END)].copy()

eval_df = pd.read_csv("outputs/ep370g_ts_v5_first_crossing_eval.csv")
crossing_df = eval_df[eval_df["has_crossing"] == True].copy()
crossing_df["first_crossing_date"] = pd.to_datetime(crossing_df["first_crossing_date"])
crossing_df["machine_id"] = crossing_df["machine_id"].astype(str)

no_crossing_machines = set(eval_df[eval_df["has_crossing"] == False]["machine_id"].astype(str))
crossing_machines = set(crossing_df["machine_id"])

# ---------------------------------------------------------------------------
# Rule-based: machine-level 7-day rolling sum of hours_at_31kv
# Use max across plugs per machine per day, then rolling sum
# ---------------------------------------------------------------------------
machine_daily = (
    holdout.groupby(["machine_id", "date"])["hours_at_31kv"]
    .max()  # worst plug per machine
    .reset_index()
    .sort_values(["machine_id", "date"])
)

# 7-day rolling sum per machine
machine_daily["hours_31kv_7d_sum"] = (
    machine_daily.groupby("machine_id")["hours_at_31kv"]
    .transform(lambda s: s.rolling(7, min_periods=1).sum())
)

print("=" * 65)
print(f"Rule-based: hours_at_31kV 7-day rolling sum > threshold")
print(f"Holdout: {HOLDOUT_START} to {HOLDOUT_END}  |  Crossing machines: {len(crossing_machines)}")
print("=" * 65)
print(f"\n{'threshold':>10}  {'warned/20':>10}  {'avg_lead_days':>14}  {'fp_alerts':>10}")

best_rule = None
for thr in RULE_THRESHOLDS:
    tp_count = 0
    lead_days = []
    fp_count = 0

    for machine in crossing_machines:
        row = crossing_df[crossing_df["machine_id"] == machine].iloc[0]
        cross_date = row["first_crossing_date"]
        mdata = machine_daily[
            (machine_daily["machine_id"] == machine)
            & (machine_daily["date"] < cross_date)
        ]
        alerts = mdata[mdata["hours_31kv_7d_sum"] > thr]
        if not alerts.empty:
            warn_date = alerts["date"].min()
            tp_count += 1
            lead_days.append((cross_date - warn_date).days)

    for machine in no_crossing_machines:
        mdata = machine_daily[machine_daily["machine_id"] == machine]
        alerts = mdata[mdata["hours_31kv_7d_sum"] > thr]
        if not alerts.empty:
            fp_count += 1

    avg_lead = sum(lead_days) / len(lead_days) if lead_days else 0
    print(f"{thr:>10.0f}  {tp_count:>5}/{len(crossing_machines):<4}  {avg_lead:>14.1f}  {fp_count:>10}")
    if best_rule is None or (tp_count > best_rule["tp"] or
                              (tp_count == best_rule["tp"] and avg_lead > best_rule["avg_lead"])):
        best_rule = {"thr": thr, "tp": tp_count, "avg_lead": avg_lead, "fp": fp_count}

print(f"\nBest rule: threshold={best_rule['thr']}h  warned={best_rule['tp']}/{len(crossing_machines)}  avg_lead={best_rule['avg_lead']:.1f}d  fp={best_rule['fp']}")

# ---------------------------------------------------------------------------
# DataRobot evaluation
# ---------------------------------------------------------------------------
try:
    pred = pd.read_csv(args.dr_pred)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True).dt.tz_localize(None)
    pred["forecast_point"] = pd.to_datetime(pred["forecast_point"], utc=True).dt.tz_localize(None)
    pred["machine_id"] = pred["series_id"].str.rsplit("_", n=1).str[0]

    # Need actual daily_max at forecast_point to filter fp_actual < 32
    fp_actuals = dataset[["管理No_プラグNo", "date", "daily_max"]].copy()
    fp_actuals["machine_id"] = fp_actuals["管理No_プラグNo"].str.rsplit("_", n=1).str[0]
    fp_max = fp_actuals.groupby(["machine_id", "date"])["daily_max"].max().reset_index()
    fp_max = fp_max.rename(columns={"date": "forecast_point", "daily_max": "fp_actual_max"})

    pred = pred.merge(fp_max, on=["machine_id", "forecast_point"], how="left")

    print(f"\n{'=' * 65}")
    print(f"DataRobot {args.label}: prediction >= {THRESHOLD_CROSS}kV before first crossing")
    print(f"(fp_actual_max < {THRESHOLD_CROSS} guard applied)")
    print("=" * 65)

    dr_tp = 0
    dr_lead = []
    dr_fp = 0
    dr_rows = []

    for machine in crossing_machines:
        row = crossing_df[crossing_df["machine_id"] == machine].iloc[0]
        cross_date = row["first_crossing_date"]
        mp = pred[
            (pred["machine_id"] == machine)
            & (pred["forecast_point"] < cross_date)
            & (pred["fp_actual_max"] < THRESHOLD_CROSS)
            & (pred["prediction"] >= THRESHOLD_CROSS)
        ]
        if not mp.empty:
            warn_fp = mp["forecast_point"].min()
            dr_tp += 1
            dr_lead.append((cross_date - warn_fp).days)
            dr_rows.append({"machine": machine, "cross": cross_date.date(),
                            "warn": warn_fp.date(), "lead": (cross_date - warn_fp).days})
        else:
            dr_rows.append({"machine": machine, "cross": cross_date.date(),
                            "warn": None, "lead": None})

    for machine in no_crossing_machines:
        mp = pred[
            (pred["machine_id"] == machine)
            & (pred["fp_actual_max"] < THRESHOLD_CROSS)
            & (pred["prediction"] >= THRESHOLD_CROSS)
        ]
        if not mp.empty:
            dr_fp += 1

    avg_dr_lead = sum(dr_lead) / len(dr_lead) if dr_lead else 0
    print(f"Warned before crossing: {dr_tp}/{len(crossing_machines)}")
    print(f"Avg lead days: {avg_dr_lead:.1f}")
    print(f"False positive machines: {dr_fp}/{len(no_crossing_machines)}")

    print(f"\n{'machine':>10}  {'crossing':>12}  {'dr_warn':>12}  {'lead_days':>10}")
    for r in dr_rows:
        lead_str = f"{r['lead']}d" if r["lead"] is not None else "-"
        warn_str = str(r["warn"]) if r["warn"] else "not warned"
        print(f"{r['machine']:>10}  {str(r['cross']):>12}  {warn_str:>12}  {lead_str:>10}")

    # Summary comparison
    print(f"\n{'=' * 65}")
    print("SUMMARY COMPARISON")
    print(f"{'Method':>30}  {'warned/20':>10}  {'avg_lead':>10}  {'fp':>5}")
    print(f"{'Rule-based (best)':>30}  {best_rule['tp']:>5}/{len(crossing_machines):<4}  {best_rule['avg_lead']:>10.1f}  {best_rule['fp']:>5}")
    print(f"{f'DataRobot {args.label}':>30}  {dr_tp:>5}/{len(crossing_machines):<4}  {avg_dr_lead:>10.1f}  {dr_fp:>5}")

except FileNotFoundError:
    print(f"\n[INFO] DR prediction file not found: {args.dr_pred}")
    print("       Run DataRobot Autopilot first, then fetch predictions.")
    print("       Rule-based evaluation above is complete.")
