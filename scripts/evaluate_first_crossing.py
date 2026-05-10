"""
ビジネス指向評価: 初回32kV超過前に警告を出せたか

評価ルール:
- 各 series (管理No_プラグNo) について、holdout 期間中に初めて daily_max >= 32kV になった日を特定
- DataRobot: forecast_point が first_crossing_date より前の時点で prediction >= 32 があれば「事前警告あり」
  (any FD=1-7 の予測値がいずれかで 32kV 以上になれば警告)
- Persistence: 7日前の actual が 32kV 以上なら「警告」(構造的に first crossing 前は不可能)
- 超過実績がない series は評価対象外 (正常系)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

THRESHOLD = 32.0
HOLDOUT_START = "2024-04-01"
HOLDOUT_END   = "2025-03-31"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
actual_raw = pd.read_csv(
    "outputs/dataset_ep370g_ts_v4.csv",
    encoding="utf-8-sig",
    parse_dates=["date"],
)
# col[1] = 管理No_プラグNo (series_id), col[2] = daily_max
id_col = actual_raw.columns[1]
actual_raw = actual_raw.rename(columns={id_col: "series_id"})

pred = pd.read_csv("outputs/ep370g_ts_v5_holdout_pred.csv")
pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True).dt.tz_localize(None)
pred["forecast_point"] = pd.to_datetime(pred["forecast_point"], utc=True).dt.tz_localize(None)

# Holdout actual
holdout_actual = actual_raw[
    actual_raw["date"].between(HOLDOUT_START, HOLDOUT_END)
].copy()

# ---------------------------------------------------------------------------
# Per-machine (管理No単位) evaluation
# DataRobot series_id = "5630_1" ... "5630_6" → machine = "5630"
# 6本のプラグのうち1本でも超過 → 全交換
# ---------------------------------------------------------------------------
holdout_actual["machine_id"] = holdout_actual["series_id"].str.rsplit("_", n=1).str[0]
pred["machine_id"] = pred["series_id"].str.rsplit("_", n=1).str[0]

machines = sorted(holdout_actual["machine_id"].unique())

rows = []
for machine in machines:
    act = holdout_actual[holdout_actual["machine_id"] == machine].copy()
    act_sorted = act.sort_values("date")

    # First crossing date across all plugs in this machine
    crossing = act_sorted[act_sorted["daily_max"] >= THRESHOLD]
    if crossing.empty:
        # No crossing in holdout - not evaluated
        rows.append({
            "machine_id": machine,
            "first_crossing_date": None,
            "has_crossing": False,
            "dr_warned_before": None,
            "dr_warn_date": None,
            "dr_days_before": None,
            "persistence_warned_before": None,
            "note": "no crossing in holdout",
        })
        continue

    first_crossing_date = crossing["date"].min()

    # -----------------------------------------------------------------------
    # DataRobot evaluation
    # forecast_point < first_crossing_date and prediction >= THRESHOLD
    # -----------------------------------------------------------------------
    machine_pred = pred[pred["machine_id"] == machine].copy()
    # forecast_point is the date from which we forecast; timestamp = prediction date
    # We want: the model WARNED before first_crossing_date
    # A warning fires if: timestamp == first_crossing_date AND prediction >= 32
    #   issued from forecast_point < first_crossing_date
    # More broadly: any forecast issued from a forecast_point before first_crossing_date
    #   that predicts timestamp <= some window and prediction >= 32

    # Simplest definition: any row where forecast_point < first_crossing_date
    # and prediction >= THRESHOLD (regardless of which future day it targets)
    pre_crossing_preds = machine_pred[
        machine_pred["forecast_point"] < first_crossing_date
    ]
    dr_alerts = pre_crossing_preds[pre_crossing_preds["prediction"] >= THRESHOLD]

    if not dr_alerts.empty:
        # Earliest forecast_point that issued an alert
        earliest_alert_fp = dr_alerts["forecast_point"].min()
        days_before = (first_crossing_date - earliest_alert_fp).days
        rows.append({
            "machine_id": machine,
            "first_crossing_date": first_crossing_date.date(),
            "has_crossing": True,
            "dr_warned_before": True,
            "dr_warn_date": earliest_alert_fp.date(),
            "dr_days_before": days_before,
            "persistence_warned_before": False,
            "note": "",
        })
    else:
        rows.append({
            "machine_id": machine,
            "first_crossing_date": first_crossing_date.date(),
            "has_crossing": True,
            "dr_warned_before": False,
            "dr_warn_date": None,
            "dr_days_before": None,
            "persistence_warned_before": False,
            "note": "persistence structurally cannot warn before first crossing",
        })

result = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
crossing_machines = result[result["has_crossing"]]
n_crossing = len(crossing_machines)
n_dr_warned = crossing_machines["dr_warned_before"].sum()

print("=" * 60)
print("Business-oriented evaluation: First Crossing Warning")
print(f"Threshold: {THRESHOLD} kV  |  Holdout: {HOLDOUT_START} - {HOLDOUT_END}")
print("=" * 60)
print(f"\nMachines with crossing in holdout: {n_crossing}")
print(f"DataRobot warned BEFORE first crossing: {int(n_dr_warned)} / {n_crossing}")
print(f"Persistence warned before first crossing: 0 / {n_crossing}  (structurally impossible)")
print()
print(crossing_machines[[
    "machine_id", "first_crossing_date",
    "dr_warned_before", "dr_warn_date", "dr_days_before"
]].to_string(index=False))

# Also show machines with no crossing
no_crossing = result[~result["has_crossing"]]
print(f"\nMachines with NO crossing in holdout: {len(no_crossing)}")

# Save
result.to_csv("outputs/ep370g_ts_v5_first_crossing_eval.csv", index=False, encoding="utf-8-sig")
print("\nSaved: outputs/ep370g_ts_v5_first_crossing_eval.csv")

# ---------------------------------------------------------------------------
# Detailed per-machine: days before crossing distribution
# ---------------------------------------------------------------------------
warned = crossing_machines[crossing_machines["dr_warned_before"] == True]
if not warned.empty:
    print(f"\nDays before first crossing (DataRobot warned):")
    print(warned[["machine_id", "dr_warn_date", "first_crossing_date", "dr_days_before"]].to_string(index=False))
