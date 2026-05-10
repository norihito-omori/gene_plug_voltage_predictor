"""
too_early 削減の追加候補 — L2 条件に rate14 絶対レベル要件を加える

仮説: too_early 事例は rate14 が「上昇しつつあるが絶対値はまだ低い」段階で発火している
（days_at_bl14 中央値 17日 vs timely 11日）。rate14 の絶対値が一定以上を要求すれば、
真に越境直前の急上昇のみを残せる可能性。

候補:
  G1: 既存 L2 (slope14_w3>0.025) AND rate14_w3 >= R_min
  G2: 既存 L2 AND rate14 (当日) >= R_min
  G3: L1 履歴を厳格化 — slope11 fire の累積日数 >= K
  G4: rate14 absolute alone — rate14_w3 >= R を L2 にして、L1 を維持
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
        .agg(daily_max=("daily_max", "max"), cross_thr=("cross_thr", "max"),
             **{f"rate{n}": (f"rate{n}", "max") for n in LEVELS})
        .reset_index().sort_values(["machine_id", "date"])
    )
    for n in LEVELS:
        for w in WINDOWS:
            md[f"rrate{n}_w{w}"] = (
                md.groupby("machine_id")[f"rate{n}"]
                .transform(lambda s: s.rolling(w, min_periods=1).mean())
            )
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
        recall=tp / n_events * 100, timely_rate=timely / n_events * 100,
        too_early=too_early, too_late=too_late, missed=n_events - tp,
        timely=timely, fp=fp, fp_rate=fp / n_nc * 100,
    )


print("Loading data...", flush=True)
md, crossing_events, gen_periods, no_cross_gens = load_machine_daily()
print(f"  events={len(crossing_events)} no_crossing={len(no_cross_gens)}", flush=True)

md["l1_fire"] = (md[L1_COL] > L1_THR).astype(int)
md["l1_hist"] = (
    md.groupby("machine_id")["l1_fire"]
    .transform(lambda x: x.rolling(HIST, min_periods=1).max())
)
md["l2_fire"] = (md[L2_COL] > L2_THR).astype(int)
md["base_alert"] = ((md["l1_hist"] > 0) & (md["l2_fire"] > 0)).astype(int)

base_r = evaluate_alert(md, md["base_alert"].values, crossing_events, gen_periods, no_cross_gens)
print(f"\n=== Baseline ===")
print(f"  timely={base_r['timely']} too_early={base_r['too_early']} too_late={base_r['too_late']} "
      f"missed={base_r['missed']} fp_rate={base_r['fp_rate']:.1f}%")

# === G1: L2 + rate14_w3 absolute ===
print(f"\n=== G1: L1_hist AND L2_fire AND rate14_w3 >= R ===")
print(f"{'R':>6} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp_rate':>7}")
for r_min in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    alert = ((md["l1_hist"] > 0) & (md["l2_fire"] > 0) & (md["rrate14_w3"] >= r_min)).astype(int).values
    r = evaluate_alert(md, alert, crossing_events, gen_periods, no_cross_gens)
    print(f"{r_min:>6.2f} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp_rate']:>6.1f}%")

# === G2: L2 + rate14 (today) absolute ===
print(f"\n=== G2: L1_hist AND L2_fire AND rate14_today >= R ===")
print(f"{'R':>6} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp_rate':>7}")
for r_min in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    alert = ((md["l1_hist"] > 0) & (md["l2_fire"] > 0) & (md["rate14"] >= r_min)).astype(int).values
    r = evaluate_alert(md, alert, crossing_events, gen_periods, no_cross_gens)
    print(f"{r_min:>6.2f} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp_rate']:>6.1f}%")

# === G3: rate14_w3 alone (slope 不要) ===
print(f"\n=== G3: L1_hist AND rate14_w3 >= R ===")
print(f"{'R':>6} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp_rate':>7}")
for r_min in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
    alert = ((md["l1_hist"] > 0) & (md["rrate14_w3"] >= r_min)).astype(int).values
    r = evaluate_alert(md, alert, crossing_events, gen_periods, no_cross_gens)
    print(f"{r_min:>6.2f} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp_rate']:>6.1f}%")

# === G4: G1 + days_in_gen ===
gen_periods_md = gen_periods.copy()
md_gen = md[["machine_id", "date"]].merge(gen_periods_md, on="machine_id", how="left")
md_gen = md_gen[(md_gen["date"] >= md_gen["gen_start"]) & (md_gen["date"] <= md_gen["gen_end"])]
md_gen = md_gen[["machine_id", "date", "gen_no", "gen_start"]]
md = md.merge(md_gen, on=["machine_id", "date"], how="left")
md["days_in_gen"] = (md["date"] - md["gen_start"]).dt.days

print(f"\n=== G4: G1 (R=best) + days_in_gen >= D ===")
print(f"{'R':>6} {'D':>4} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp_rate':>7}")
for r_min, d in product([0.10, 0.15, 0.20, 0.25, 0.30], [30, 60, 90]):
    alert = ((md["l1_hist"] > 0) & (md["l2_fire"] > 0)
             & (md["rrate14_w3"] >= r_min) & (md["days_in_gen"] >= d)).astype(int).values
    r = evaluate_alert(md, alert, crossing_events, gen_periods, no_cross_gens)
    print(f"{r_min:>6.2f} {d:>4} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp_rate']:>6.1f}%")
