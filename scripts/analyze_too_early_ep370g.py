"""
EP370G 推奨ルール (slope11_w5>0.04 hist=10d ∧ slope14_w3>0.025) で
too_early と判定される 11 件を特定し、メタデータを抽出する。

timely=27件、too_early=11件、too_late=1件、missed=5件 の内訳を比較し、
too_early に特徴的な性質（機台/季節/baseline/稼働パターン）の偏りを探す。
"""
from __future__ import annotations
import pandas as pd
import numpy as np

CLEANED = "outputs/cleaned_ep370g_v5.csv"
DAILY = "outputs/dataset_ep370g_ts_v7.csv"

LEVELS = [11, 14]
WINDOWS = [3, 5]  # 必要なものだけ
MIN_OP_HOURS = 1.0
T_MIN, T_MAX = 7, 14

# 推奨ルール
L1_COL, L1_THR, HIST = "slope11_w5", 0.040, 10
L2_COL, L2_THR = "slope14_w3", 0.025


def _slope(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(valid), dtype=float)
    return np.polyfit(x, valid, 1)[0]


def load_machine_daily():
    cleaned_raw = pd.read_csv(CLEANED, encoding="utf-8-sig", nrows=0)
    cols = cleaned_raw.columns.tolist()
    power_col, voltage_col, plug_id_col, datetime_col = cols[3], cols[7], cols[9], cols[1]

    cleaned = pd.read_csv(
        CLEANED, encoding="utf-8-sig",
        usecols=[datetime_col, power_col, voltage_col, plug_id_col, "gen_no", "baseline"],
    )
    cleaned["_date"] = pd.to_datetime(cleaned[datetime_col]).dt.normalize()
    running = cleaned[cleaned[power_col] > 0].copy()

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

    daily = pd.read_csv(DAILY, encoding="utf-8-sig", parse_dates=["date"])
    plug_col = daily.columns[1] if daily.columns[0] == "date" else daily.columns[0]
    daily["machine_id"] = daily[plug_col].str.rsplit("_", n=1).str[0]
    daily["date"] = pd.to_datetime(daily["date"])

    merged = daily.merge(
        agg[[plug_id_col, "date"] + [f"rate{n}" for n in LEVELS]],
        left_on=[plug_col, "date"], right_on=[plug_id_col, "date"], how="left",
    )
    merged["cross_thr"] = merged["baseline"] + 16

    op2 = merged[merged["is_operating"] == 1].copy()
    md = (
        op2.groupby(["machine_id", "date"])
        .agg(
            daily_max=("daily_max", "max"),
            cross_thr=("cross_thr", "max"),
            baseline=("baseline", "max"),
            op_ratio=("稼働割合", "max"),
            **{f"rate{n}": (f"rate{n}", "max") for n in LEVELS},
        )
        .reset_index().sort_values(["machine_id", "date"])
    )

    for n in LEVELS:
        for w in WINDOWS:
            md[f"slope{n}_w{w}"] = (
                md.groupby("machine_id")[f"rate{n}"]
                .transform(lambda s: s.rolling(w, min_periods=2).apply(_slope, raw=True))
            )

    op = daily[daily["is_operating"] == 1].copy()
    op["cross_thr"] = op["baseline"] + 16
    crossing_events = (
        op[op["daily_max"] >= op["cross_thr"]]
        .groupby(["machine_id", "gen_no"])["date"].min()
        .reset_index().rename(columns={"date": "first_crossing_date"})
    )
    gen_periods = (
        op.groupby(["machine_id", "gen_no"])["date"]
        .agg(gen_start="min", gen_end="max").reset_index()
    )
    return md, crossing_events, gen_periods


def classify_events(md, crossing_events, gen_periods):
    """各越境イベントを timely/too_early/too_late/missed に分類し、メタデータを付与"""
    md = md.copy()
    md["l1_fire"] = (md[L1_COL] > L1_THR).astype(int)
    md["l1_hist"] = (
        md.groupby("machine_id")["l1_fire"]
        .transform(lambda x: x.rolling(HIST, min_periods=1).max())
    )
    md["l2_fire"] = (md[L2_COL] > L2_THR).astype(int)
    md["alert"] = ((md["l1_hist"] > 0) & (md["l2_fire"] > 0)).astype(int)

    rows = []
    for _, ev in crossing_events.iterrows():
        machine, gen_no, cross_date = ev["machine_id"], ev["gen_no"], ev["first_crossing_date"]
        gen_row = gen_periods[(gen_periods["machine_id"] == machine) & (gen_periods["gen_no"] == gen_no)]
        gen_start = gen_row["gen_start"].iloc[0]

        sub = md[
            (md["machine_id"] == machine)
            & (md["date"] >= gen_start)
            & (md["date"] < cross_date)
            & (md["daily_max"] < md["cross_thr"])
        ].copy()

        baseline = sub["baseline"].max() if not sub.empty else np.nan
        cross_thr = baseline + 16

        first_alert = sub[sub["alert"] > 0]["date"].min() if not sub[sub["alert"] > 0].empty else pd.NaT

        # 期間統計
        n_days = len(sub)
        avg_op_ratio = sub["op_ratio"].mean() if not sub.empty else np.nan
        max_rate11 = sub["rate11"].max() if not sub.empty else np.nan
        max_rate14 = sub["rate14"].max() if not sub.empty else np.nan
        days_at_bl11 = (sub["rate11"] > 0).sum() if not sub.empty else 0
        days_at_bl14 = (sub["rate14"] > 0).sum() if not sub.empty else 0

        if pd.isna(first_alert):
            cls = "missed"
            lead = np.nan
        else:
            lead = (cross_date - first_alert).days
            if T_MIN <= lead <= T_MAX:
                cls = "timely"
            elif lead < T_MIN:
                cls = "too_late"
            else:
                cls = "too_early"

        rows.append({
            "machine_id": machine,
            "gen_no": gen_no,
            "cross_date": cross_date,
            "first_alert": first_alert,
            "lead_days": lead,
            "class": cls,
            "baseline": baseline,
            "cross_thr": cross_thr,
            "gen_duration_days": n_days,
            "avg_op_ratio": avg_op_ratio,
            "max_rate11": max_rate11,
            "max_rate14": max_rate14,
            "days_at_bl11": days_at_bl11,
            "days_at_bl14": days_at_bl14,
        })
    return pd.DataFrame(rows)


print("Loading data...", flush=True)
md, crossing_events, gen_periods = load_machine_daily()
print(f"  events={len(crossing_events)}", flush=True)

print("Classifying events...", flush=True)
events = classify_events(md, crossing_events, gen_periods)

# 分類サマリ
print("\n=== Classification summary ===")
print(events["class"].value_counts())

# クラス別の統計比較
print("\n=== Per-class statistics (mean) ===")
stats_cols = ["lead_days", "baseline", "gen_duration_days", "avg_op_ratio",
              "max_rate11", "max_rate14", "days_at_bl11", "days_at_bl14"]
print(events.groupby("class")[stats_cols].mean().round(3).T.to_string())

print("\n=== Per-class statistics (median) ===")
print(events.groupby("class")[stats_cols].median().round(3).T.to_string())

# too_early 11件の詳細
print("\n=== too_early events (sorted by lead_days desc) ===")
too_early = events[events["class"] == "too_early"].sort_values("lead_days", ascending=False)
print(too_early[["machine_id", "gen_no", "cross_date", "first_alert", "lead_days",
                 "baseline", "gen_duration_days", "avg_op_ratio",
                 "max_rate11", "max_rate14", "days_at_bl11", "days_at_bl14"]].to_string(index=False))

# 機台別出現回数
print("\n=== machine_id distribution per class ===")
machine_dist = events.groupby(["class", "machine_id"]).size().unstack(fill_value=0).T
print(machine_dist.to_string())

# 季節（cross_date の月）
events["cross_month"] = pd.to_datetime(events["cross_date"]).dt.month
print("\n=== cross month distribution per class ===")
month_dist = events.groupby(["class", "cross_month"]).size().unstack(fill_value=0).T
print(month_dist.to_string())

# alert month
events["alert_month"] = pd.to_datetime(events["first_alert"]).dt.month
print("\n=== first_alert month distribution per class ===")
am_dist = events.groupby(["class", "alert_month"]).size().unstack(fill_value=0).T
print(am_dist.to_string())

# baseline ヒストグラム
print("\n=== baseline distribution per class ===")
events["baseline_bin"] = pd.cut(events["baseline"], bins=[0, 10, 14, 18, 22, 30],
                                  labels=["<=10", "11-14", "15-18", "19-22", "23+"])
bl_dist = events.groupby(["class", "baseline_bin"], observed=False).size().unstack(fill_value=0).T
print(bl_dist.to_string())

# 出力 CSV
events_out = events.drop(columns=["baseline_bin"]).copy()
events_out.to_csv("outputs/too_early_analysis_ep370g.csv", index=False, encoding="utf-8-sig")
print("\nSaved: outputs/too_early_analysis_ep370g.csv")
