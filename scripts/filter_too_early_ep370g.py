"""
too_early=11 を削減するフィルタ候補を試算する。

ベースライン: slope11_w5>0.04 hist=10d ∧ slope14_w3>0.025
  → timely=27, too_early=11, too_late=1, missed=5, FP率=20.3%

候補:
  F1: アラート持続性 — 直近 N 日のうち M 日以上発火していたら採用
  F2: サイクル経過日数下限 — gen 開始から min_days 日経過後のみ採用
  F3: L2 持続性 — slope14_w3 の発火が直近 N 日中 M 日以上
  F4: 機台別ルール — 9221/9230 では追加要件
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from itertools import product

CLEANED = "outputs/cleaned_ep370g_v5.csv"
DAILY = "outputs/dataset_ep370g_ts_v7.csv"

LEVELS = [11, 14]
WINDOWS = [3, 5]
MIN_OP_HOURS = 1.0
T_MIN, T_MAX = 7, 14

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
            agg["op_hours"] >= MIN_OP_HOURS, agg[f"_at{n}"] / agg["op_hours"], np.nan,
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
    no_cross_gens = gen_periods.merge(
        crossing_events[["machine_id", "gen_no"]],
        on=["machine_id", "gen_no"], how="left", indicator=True
    )
    no_cross_gens = no_cross_gens[no_cross_gens["_merge"] == "left_only"].drop(columns="_merge")
    return md, crossing_events, gen_periods, no_cross_gens


def evaluate_alert(md, alert_arr, crossing_events, gen_periods, no_cross_gens):
    """alert_arr (md の各行に対応する 0/1 配列) を評価"""
    machine_arr = md["machine_id"].values
    date_arr = md["date"].values
    daily_max_arr = md["daily_max"].values
    cross_thr_arr = md["cross_thr"].values

    tp = fp = timely = too_early = too_late = 0
    for _, ev in crossing_events.iterrows():
        machine, gen_no, cross_date = ev["machine_id"], ev["gen_no"], ev["first_crossing_date"]
        gen_row = gen_periods[(gen_periods["machine_id"] == machine) & (gen_periods["gen_no"] == gen_no)]
        gen_start = gen_row["gen_start"].iloc[0]
        mask = (
            (machine_arr == machine)
            & (date_arr >= np.datetime64(gen_start))
            & (date_arr < np.datetime64(cross_date))
            & (daily_max_arr < cross_thr_arr)
        )
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        sub_alert = alert_arr[idx]
        if not (sub_alert > 0).any():
            continue
        tp += 1
        fire_idx = idx[sub_alert > 0]
        first_date = date_arr[fire_idx].min()
        lead = (cross_date - pd.Timestamp(first_date)).days
        if T_MIN <= lead <= T_MAX:
            timely += 1
        elif lead < T_MIN:
            too_late += 1
        else:
            too_early += 1

    for _, nc in no_cross_gens.iterrows():
        machine, gs, ge = nc["machine_id"], nc["gen_start"], nc["gen_end"]
        mask = (machine_arr == machine) & (date_arr >= np.datetime64(gs)) & (date_arr <= np.datetime64(ge))
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        if (alert_arr[idx] > 0).any():
            fp += 1

    n_events = len(crossing_events)
    n_nc = len(no_cross_gens)
    return dict(
        recall=tp / n_events * 100,
        timely_rate=timely / n_events * 100,
        too_early=too_early, too_late=too_late, missed=n_events - tp,
        timely=timely, fp=fp, fp_rate=fp / n_nc * 100,
    )


print("Loading data...", flush=True)
md, crossing_events, gen_periods, no_cross_gens = load_machine_daily()
print(f"  events={len(crossing_events)} no_crossing={len(no_cross_gens)}", flush=True)

# ベースアラート
md["l1_fire"] = (md[L1_COL] > L1_THR).astype(int)
md["l1_hist"] = (
    md.groupby("machine_id")["l1_fire"]
    .transform(lambda x: x.rolling(HIST, min_periods=1).max())
)
md["l2_fire"] = (md[L2_COL] > L2_THR).astype(int)
md["base_alert"] = ((md["l1_hist"] > 0) & (md["l2_fire"] > 0)).astype(int)

# gen_start を md にマージ（経過日数算出用）
gen_periods_md = gen_periods.copy()
gen_periods_md["machine_id"] = gen_periods_md["machine_id"].astype(str)
md["machine_id"] = md["machine_id"].astype(str)

# 各 (machine_id, date) に対応する gen_start を拾う
# gen_periods_md の (machine_id, gen_start, gen_end) と md の date がどの gen に属するかを判定
md_gen = md[["machine_id", "date"]].merge(gen_periods_md, on="machine_id", how="left")
md_gen = md_gen[(md_gen["date"] >= md_gen["gen_start"]) & (md_gen["date"] <= md_gen["gen_end"])]
md_gen = md_gen[["machine_id", "date", "gen_no", "gen_start"]]
md = md.merge(md_gen, on=["machine_id", "date"], how="left")
md["days_in_gen"] = (md["date"] - md["gen_start"]).dt.days

# === Baseline ===
base_r = evaluate_alert(md, md["base_alert"].values, crossing_events, gen_periods, no_cross_gens)
print(f"\n=== Baseline ===")
print(f"  timely={base_r['timely']} too_early={base_r['too_early']} too_late={base_r['too_late']} "
      f"missed={base_r['missed']} fp_rate={base_r['fp_rate']:.1f}%")

# === F1: アラート持続性 ===
print(f"\n=== F1: 直近 N 日中 M 日以上 base_alert ===")
print(f"{'N':>3} {'M':>3} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp_rate':>7}")
results_f1 = []
for n_win, m_thr in [(3, 2), (3, 3), (5, 2), (5, 3), (5, 4), (7, 3), (7, 4), (7, 5),
                       (10, 3), (10, 5), (14, 5), (14, 7)]:
    md["sustain"] = (
        md.groupby("machine_id")["base_alert"]
        .transform(lambda x: x.rolling(n_win, min_periods=1).sum())
    )
    alert = (md["sustain"] >= m_thr).astype(int).values
    r = evaluate_alert(md, alert, crossing_events, gen_periods, no_cross_gens)
    results_f1.append({"N": n_win, "M": m_thr, **r})
    print(f"{n_win:>3} {m_thr:>3} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp_rate']:>6.1f}%")

# === F2: サイクル経過日数下限 ===
print(f"\n=== F2: gen 開始から min_days 経過後のみ発火 ===")
print(f"{'min_days':>9} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp_rate':>7}")
for min_days in [30, 60, 90, 120, 150]:
    alert = ((md["base_alert"] > 0) & (md["days_in_gen"] >= min_days)).astype(int).values
    r = evaluate_alert(md, alert, crossing_events, gen_periods, no_cross_gens)
    print(f"{min_days:>9} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp_rate']:>6.1f}%")

# === F3: L2 持続性 (l2_fire を持続条件にする) ===
print(f"\n=== F3: L2 (slope14) 直近 N 日中 M 日以上発火 + L1 hist ===")
print(f"{'N':>3} {'M':>3} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp_rate':>7}")
for n_win, m_thr in [(3, 2), (3, 3), (5, 2), (5, 3), (5, 4), (7, 3), (7, 4), (7, 5)]:
    md["l2_sustain"] = (
        md.groupby("machine_id")["l2_fire"]
        .transform(lambda x: x.rolling(n_win, min_periods=1).sum())
    )
    alert = ((md["l1_hist"] > 0) & (md["l2_sustain"] >= m_thr)).astype(int).values
    r = evaluate_alert(md, alert, crossing_events, gen_periods, no_cross_gens)
    print(f"{n_win:>3} {m_thr:>3} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp_rate']:>6.1f}%")

# === F4: F1 と F2 の組合せ ===
print(f"\n=== F4: F1 + F2 組合せ (sustain N=5 M>=3 AND days_in_gen>=min) ===")
print(f"{'N':>3} {'M':>3} {'min_d':>5} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp_rate':>7}")
for n_win, m_thr, min_d in product([5, 7], [2, 3], [60, 90, 120]):
    md["sustain"] = (
        md.groupby("machine_id")["base_alert"]
        .transform(lambda x: x.rolling(n_win, min_periods=1).sum())
    )
    alert = ((md["sustain"] >= m_thr) & (md["days_in_gen"] >= min_d)).astype(int).values
    r = evaluate_alert(md, alert, crossing_events, gen_periods, no_cross_gens)
    print(f"{n_win:>3} {m_thr:>3} {min_d:>5} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp_rate']:>6.1f}%")
