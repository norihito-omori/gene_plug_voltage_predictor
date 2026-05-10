"""
30分データから hours_at_31kV (1日に31kVにいた時間) を集計し、
初回32kV越境前の推移を可視化する。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

THRESHOLD_CROSS = 32
THRESHOLD_ALERT = 31
OPERATING_COL = "発電機電力"
VOLTAGE_COL = "要求電圧"
DATETIME_COL = "dailygraphpt_ptdatetime"
MACHINE_COL = "管理No"
PLUG_COL = "管理No_プラグNo"
WINDOW_DAYS = 90  # crossing 前後何日を表示するか

# ---------------------------------------------------------------------------
# Load raw 30-min data
# ---------------------------------------------------------------------------
print("Loading 30-min raw data...")
raw = pd.read_csv(
    "outputs/cleaned_ep370g_v4.csv",
    encoding="utf-8-sig",
    usecols=[MACHINE_COL, DATETIME_COL, VOLTAGE_COL, OPERATING_COL],
    parse_dates=[DATETIME_COL],
)
raw = raw[raw[OPERATING_COL] > 0].copy()  # operating rows only
raw["date"] = raw[DATETIME_COL].dt.normalize()

# hours_at_31kV per machine per day
# each slot = 30 min = 0.5 hour
raw["is_31kv"] = (raw[VOLTAGE_COL] == THRESHOLD_ALERT).astype(int)
raw["is_ge_31kv"] = (raw[VOLTAGE_COL] >= THRESHOLD_ALERT).astype(int)

daily = (
    raw.groupby([MACHINE_COL, "date"])
    .agg(
        hours_at_31kv=("is_31kv", lambda x: x.sum() * 0.5),
        hours_ge_31kv=("is_ge_31kv", lambda x: x.sum() * 0.5),
        daily_max=(VOLTAGE_COL, "max"),
        n_slots=("is_31kv", "count"),
    )
    .reset_index()
)
daily["date"] = pd.to_datetime(daily["date"])

# ---------------------------------------------------------------------------
# Load crossing dates
# ---------------------------------------------------------------------------
eval_df = pd.read_csv("outputs/ep370g_ts_v5_first_crossing_eval.csv")
crossing_df = eval_df[eval_df["has_crossing"] == True].copy()
crossing_df["first_crossing_date"] = pd.to_datetime(crossing_df["first_crossing_date"])
crossing_df["machine_id"] = crossing_df["machine_id"].astype(int)

machines = sorted(crossing_df["machine_id"].tolist())
n_machines = len(machines)

# ---------------------------------------------------------------------------
# Plot: 4 columns grid, each machine shows hours_at_31kV around crossing
# ---------------------------------------------------------------------------
n_cols = 4
n_rows = (n_machines + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes_flat = axes.flatten()

for idx, machine in enumerate(machines):
    ax = axes_flat[idx]
    row = crossing_df[crossing_df["machine_id"] == machine].iloc[0]
    cross_date = row["first_crossing_date"]

    # Window: WINDOW_DAYS before crossing
    start = cross_date - pd.Timedelta(days=WINDOW_DAYS)
    mdata = daily[
        (daily[MACHINE_COL] == machine)
        & (daily["date"] >= start)
        & (daily["date"] <= cross_date + pd.Timedelta(days=7))
    ].sort_values("date")

    ax.bar(
        mdata["date"],
        mdata["hours_at_31kv"],
        color="steelblue",
        alpha=0.7,
        width=1.0,
        label="hours at 31kV",
    )
    # 7-day rolling sum (trend signal)
    mdata = mdata.set_index("date").asfreq("D")
    rolling = mdata["hours_at_31kv"].rolling(7, min_periods=1).sum()
    ax.plot(rolling.index, rolling.values / 7, color="orange", linewidth=1.5,
            label="7d rolling avg")

    ax.axvline(cross_date, color="red", linewidth=2, linestyle="--", label="first 32kV")
    ax.set_title(f"Machine {machine}\ncrossing: {cross_date.date()}", fontsize=9)
    ax.set_ylabel("hours/day at 31kV", fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(bottom=0)
    if idx == 0:
        ax.legend(fontsize=7)

# Hide unused subplots
for idx in range(n_machines, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle(
    "Hours per day at 31kV — 90 days before first 32kV crossing\n"
    "(orange = 7-day rolling avg, red dashed = first crossing)",
    fontsize=12,
    y=1.01,
)
plt.tight_layout()
out_path = "outputs/hours_at_31kv_before_crossing.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Summary stats: how many machines show monotone increase in final 14 days?
# ---------------------------------------------------------------------------
print("\n=== hours_at_31kV: last 14 days before crossing ===")
print(f"{'machine':>10}  {'avg_d-14_to_d-8':>16}  {'avg_d-7_to_d-1':>15}  {'trend':>6}  {'max_hours/day':>14}")
for machine in machines:
    row = crossing_df[crossing_df["machine_id"] == machine].iloc[0]
    cross_date = row["first_crossing_date"]
    mdata = daily[
        (daily[MACHINE_COL] == machine)
        & (daily["date"] < cross_date)
    ].sort_values("date")

    early = mdata[mdata["date"] >= cross_date - pd.Timedelta(days=14)]
    late  = mdata[mdata["date"] >= cross_date - pd.Timedelta(days=7)]
    avg_early = early["hours_at_31kv"].mean() if not early.empty else float("nan")
    avg_late  = late["hours_at_31kv"].mean()  if not late.empty  else float("nan")
    max_h = mdata["hours_at_31kv"].max()
    trend = "↑ rising" if avg_late > avg_early + 0.1 else ("→ flat" if abs(avg_late - avg_early) <= 0.1 else "↓ fall")
    print(f"{machine:>10}  {avg_early:>16.2f}  {avg_late:>15.2f}  {trend:>8}  {max_h:>14.1f}")
