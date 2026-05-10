"""
EP370G ML ルールの詳細探索 — slope11 × slope14 系統 + 隣接ウィンドウ・閾値を細粒度で

exp-015 で発見した 2 候補:
  ① rate11_w3>0.10 hist=7d ∧ slope14_w3>0.10  → timely=59.1% too_early=9 missed=6
  ② slope11_w7>0.02 hist=7d ∧ slope14_w3>0.02 → timely=59.1% too_early=11 missed=5

② 系統で missed=5 を維持しつつ too_early をさらに削減できるパラメータを探索。
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from itertools import product

CLEANED = "outputs/cleaned_ep370g_v5.csv"
DAILY = "outputs/dataset_ep370g_ts_v7.csv"

LEVELS = [11, 14]
WINDOWS = [3, 5, 7, 10]
RATE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SLOPE_THRESHOLDS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100]
HISTORY_DAYS = [3, 5, 7, 10, 14]

MIN_OP_HOURS = 1.0
T_MIN, T_MAX = 7, 14


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

    cleaned = pd.read_csv(CLEANED, encoding="utf-8-sig",
                          usecols=[datetime_col, power_col, voltage_col, plug_id_col, "gen_no", "baseline"])
    cleaned["_date"] = pd.to_datetime(cleaned[datetime_col]).dt.normalize()
    running = cleaned[cleaned[power_col] > 0].copy()

    for n in LEVELS:
        running[f"_at{n}"] = (running[voltage_col] == (running["baseline"].round() + n)).astype(float)
    running["_slot"] = 1.0

    agg = running.groupby([plug_id_col, "_date"]).agg(
        op_hours=("_slot", "sum"),
        **{f"_at{n}": (f"_at{n}", "sum") for n in LEVELS}
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


def build_event_indices(md, crossing_events, gen_periods, no_cross_gens):
    machine_arr = md["machine_id"].values
    date_arr = md["date"].values
    daily_max_arr = md["daily_max"].values
    cross_thr_arr = md["cross_thr"].values

    cross_indices = []
    for _, ev in crossing_events.iterrows():
        machine, gen_no, cross_date = ev["machine_id"], ev["gen_no"], ev["first_crossing_date"]
        gen_row = gen_periods[(gen_periods["machine_id"] == machine) & (gen_periods["gen_no"] == gen_no)]
        gen_start = gen_row["gen_start"].iloc[0] if not gen_row.empty else pd.Timestamp.min
        mask = (
            (machine_arr == machine)
            & (date_arr >= np.datetime64(gen_start))
            & (date_arr < np.datetime64(cross_date))
            & (daily_max_arr < cross_thr_arr)
        )
        cross_indices.append((cross_date, np.where(mask)[0]))

    nc_indices = []
    for _, nc in no_cross_gens.iterrows():
        machine, gs, ge = nc["machine_id"], nc["gen_start"], nc["gen_end"]
        mask = (
            (machine_arr == machine)
            & (date_arr >= np.datetime64(gs))
            & (date_arr <= np.datetime64(ge))
        )
        nc_indices.append(np.where(mask)[0])
    return cross_indices, nc_indices, date_arr


def eval_ml(combined_arr, cross_indices, nc_indices, date_arr, n_events, n_nc):
    tp = fp = timely = too_early = too_late = 0
    leads = []
    for cross_date, idx in cross_indices:
        if len(idx) == 0:
            continue
        sub = combined_arr[idx]
        if not (sub > 0).any():
            continue
        tp += 1
        fire_idx = idx[sub > 0]
        first_date = date_arr[fire_idx].min()
        lead = (cross_date - pd.Timestamp(first_date)).days
        leads.append(lead)
        if T_MIN <= lead <= T_MAX:
            timely += 1
        elif lead < T_MIN:
            too_late += 1
        else:
            too_early += 1
    for idx in nc_indices:
        if len(idx) == 0:
            continue
        if (combined_arr[idx] > 0).any():
            fp += 1
    recall = tp / n_events * 100
    timely_rate = timely / n_events * 100
    med = sorted(leads)[len(leads) // 2] if leads else 0
    fp_rate = fp / n_nc * 100 if n_nc else 0
    return dict(tp=tp, recall=recall, timely=timely, too_early=too_early,
                too_late=too_late, missed=n_events - tp, timely_rate=timely_rate,
                med_lead=med, fp=fp, fp_rate=fp_rate)


print("Loading data...", flush=True)
md, crossing_events, gen_periods, no_cross_gens = load_machine_daily()
n_events = len(crossing_events)
n_nc = len(no_cross_gens)
print(f"  events={n_events} no_crossing={n_nc}", flush=True)

print("Building event indices...", flush=True)
cross_indices, nc_indices, date_arr = build_event_indices(md, crossing_events, gen_periods, no_cross_gens)

# L1 history キャッシュ
l1_history_cache = {}
def get_l1_hist(col, thr, hist_days):
    key = (col, thr, hist_days)
    if key not in l1_history_cache:
        s = (md[col] > thr).astype(int)
        l1_history_cache[key] = (
            s.groupby(md["machine_id"])
            .transform(lambda x: x.rolling(hist_days, min_periods=1).max())
            .values
        )
    return l1_history_cache[key]

# L2 fire キャッシュ
l2_fire_cache = {}
def get_l2_fire(col, thr):
    key = (col, thr)
    if key not in l2_fire_cache:
        l2_fire_cache[key] = (md[col] > thr).astype(int).values
    return l2_fire_cache[key]


# 探索: L1 = slope11_w{3,5,7,10} × thr ∈ SLOPE × hist
#       L2 = slope14_w{3,5,7,10} × thr ∈ SLOPE
print("Searching slope11 × slope14 grid...", flush=True)
results = []
for l1_w, l1_thr, l2_w, l2_thr, hist in product(
    WINDOWS, SLOPE_THRESHOLDS, WINDOWS, SLOPE_THRESHOLDS, HISTORY_DAYS
):
    l1_col = f"slope11_w{l1_w}"
    l2_col = f"slope14_w{l2_w}"
    l1h = get_l1_hist(l1_col, l1_thr, hist)
    l2f = get_l2_fire(l2_col, l2_thr)
    combined = l1h * l2f
    r = eval_ml(combined, cross_indices, nc_indices, date_arr, n_events, n_nc)
    results.append({
        **r, "l1_pat": "slope", "l1_w": l1_w, "l1_thr": l1_thr,
        "l2_pat": "slope", "l2_w": l2_w, "l2_thr": l2_thr, "history": hist,
    })

# 探索: L1 = rate11_w{3,5,7,10} × thr ∈ RATE × hist (rate-slope ハイブリッド)
print("Searching rate11 × slope14 grid...", flush=True)
for l1_w, l1_thr, l2_w, l2_thr, hist in product(
    WINDOWS, RATE_THRESHOLDS, WINDOWS, SLOPE_THRESHOLDS, HISTORY_DAYS
):
    l1_col = f"rrate11_w{l1_w}"
    l2_col = f"slope14_w{l2_w}"
    l1h = get_l1_hist(l1_col, l1_thr, hist)
    l2f = get_l2_fire(l2_col, l2_thr)
    combined = l1h * l2f
    r = eval_ml(combined, cross_indices, nc_indices, date_arr, n_events, n_nc)
    results.append({
        **r, "l1_pat": "rate", "l1_w": l1_w, "l1_thr": l1_thr,
        "l2_pat": "slope", "l2_w": l2_w, "l2_thr": l2_thr, "history": hist,
    })

grid = pd.DataFrame(results)
grid.to_csv("outputs/refine_ep370g_ml_grid.csv", index=False, encoding="utf-8-sig")
print(f"Total combinations: {len(grid)}", flush=True)

# === 出力 1: missed <= 5 で timely_rate 最大 ===
print("\n=== missed <= 5: timely_rate desc, too_early asc, fp asc ===", flush=True)
filt = grid[grid.missed <= 5].sort_values(
    ["timely_rate", "too_early", "fp"], ascending=[False, True, True]
).head(15)
cols = ["l1_pat", "l1_w", "l1_thr", "l2_pat", "l2_w", "l2_thr", "history",
        "recall", "timely_rate", "too_early", "too_late", "missed", "fp", "fp_rate", "med_lead"]
print(filt[cols].to_string(index=False), flush=True)

# === 出力 2: missed <= 5 AND too_early <= 9 で timely_rate 最大 ===
print("\n=== missed <= 5 AND too_early <= 9 ===", flush=True)
filt = grid[(grid.missed <= 5) & (grid.too_early <= 9)].sort_values(
    ["timely_rate", "fp"], ascending=[False, True]
).head(15)
print(filt[cols].to_string(index=False), flush=True)

# === 出力 3: too_early <= 7 で timely_rate 最大（攻めの設定）===
print("\n=== too_early <= 7 ===", flush=True)
filt = grid[grid.too_early <= 7].sort_values(
    ["timely_rate", "missed", "fp"], ascending=[False, True, True]
).head(15)
print(filt[cols].to_string(index=False), flush=True)

# === 出力 4: 累積パレート (timely_rate desc, too_early asc) ===
print("\n=== Pareto: timely_rate >= 56 AND missed <= 6 ===", flush=True)
filt = grid[(grid.timely_rate >= 56) & (grid.missed <= 6)].sort_values(
    ["timely_rate", "too_early", "missed", "fp"], ascending=[False, True, True, True]
).head(20)
print(filt[cols].to_string(index=False), flush=True)
