"""
稼働率正規化シグナル rate_at_(baseline+N) のグリッドサーチ
越境定義: daily_max >= baseline + 16
シグナル: hours_at_(baseline+N) / op_hours の W日ローリング平均
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from itertools import product

DATASETS = {
    "EP370G": {"cleaned": "outputs/cleaned_ep370g_v5.csv", "daily": "outputs/dataset_ep370g_ts_v7.csv"},
    "EP400G": {"cleaned": "outputs/cleaned_ep400g_v3.csv", "daily": "outputs/dataset_ep400g_ts_v3.csv"},
}
WINDOWS = [3, 5, 7, 10, 14, 21]
RATE_THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
MIN_OP_HOURS = 1.0
T_MIN, T_MAX = 7, 14


def load_machine_daily(paths, plug_col, daily):
    cleaned_raw = pd.read_csv(paths["cleaned"], encoding="utf-8-sig", nrows=0)
    cols = cleaned_raw.columns.tolist()
    power_col, voltage_col, plug_id_col, datetime_col = cols[3], cols[7], cols[9], cols[1]

    cleaned = pd.read_csv(paths["cleaned"], encoding="utf-8-sig",
                          usecols=[datetime_col, power_col, voltage_col, plug_id_col, "gen_no", "baseline"])
    cleaned["_date"] = pd.to_datetime(cleaned[datetime_col]).dt.normalize()
    running = cleaned[cleaned[power_col] > 0].copy()

    for n in [13, 14, 15]:
        running[f"_at{n}"] = (running[voltage_col] == (running["baseline"].round() + n)).astype(float)
    running["_slot"] = 1.0

    agg = running.groupby([plug_id_col, "_date"]).agg(
        op_hours=("_slot", "sum"),
        **{f"_at{n}": (f"_at{n}", "sum") for n in [13, 14, 15]}
    ).mul(0.5).reset_index().rename(columns={"_date": "date"})
    agg["date"] = pd.to_datetime(agg["date"])

    for n in [13, 14, 15]:
        agg[f"rate{n}"] = np.where(
            agg["op_hours"] >= MIN_OP_HOURS,
            agg[f"_at{n}"] / agg["op_hours"],
            np.nan
        )

    daily2 = daily.copy()
    daily2["date"] = pd.to_datetime(daily2["date"])
    merged = daily2.merge(
        agg[[plug_id_col, "date"] + [f"rate{n}" for n in [13, 14, 15]]],
        left_on=[plug_col, "date"], right_on=[plug_id_col, "date"], how="left"
    )
    merged["cross_thr"] = merged["baseline"] + 16

    op2 = merged[merged["is_operating"] == 1].copy()
    md = (
        op2.groupby(["machine_id", "date"])
        .agg(daily_max=("daily_max", "max"), cross_thr=("cross_thr", "max"),
             **{f"rate{n}": (f"rate{n}", "max") for n in [13, 14, 15]})
        .reset_index().sort_values(["machine_id", "date"])
    )
    for n in [13, 14, 15]:
        for w in WINDOWS:
            md[f"rr{n}_w{w}"] = (
                md.groupby("machine_id")[f"rate{n}"]
                .transform(lambda s: s.rolling(w, min_periods=1).mean())
            )
    return md


def eval_signal(col, machine_daily, crossing_events, gen_periods, no_cross_gens, n_events, n_nc, thr):
    tp = fp = timely = too_early = too_late = 0
    leads = []
    for _, ev in crossing_events.iterrows():
        machine, gen_no, cross_date = ev["machine_id"], ev["gen_no"], ev["first_crossing_date"]
        gen_row = gen_periods[(gen_periods["machine_id"] == machine) & (gen_periods["gen_no"] == gen_no)]
        gen_start = gen_row["gen_start"].iloc[0] if not gen_row.empty else pd.Timestamp.min
        pre = machine_daily[
            (machine_daily["machine_id"] == machine)
            & (machine_daily["date"] >= gen_start)
            & (machine_daily["date"] < cross_date)
            & (machine_daily["daily_max"] < machine_daily["cross_thr"])
        ]
        alerts = pre[pre[col] > thr]
        if not alerts.empty:
            tp += 1
            lead = (cross_date - alerts["date"].min()).days
            leads.append(lead)
            if T_MIN <= lead <= T_MAX:
                timely += 1
            elif lead < T_MIN:
                too_late += 1
            else:
                too_early += 1
    for _, nc in no_cross_gens.iterrows():
        machine, gs, ge = nc["machine_id"], nc["gen_start"], nc["gen_end"]
        period = machine_daily[
            (machine_daily["machine_id"] == machine)
            & (machine_daily["date"] >= gs)
            & (machine_daily["date"] <= ge)
        ]
        if (period[col] > thr).any():
            fp += 1
    recall = tp / n_events * 100
    timely_rate = timely / n_events * 100
    med = sorted(leads)[len(leads) // 2] if leads else 0
    fp_rate = fp / n_nc * 100 if n_nc else 0
    return dict(tp=tp, recall=recall, timely=timely, too_early=too_early,
                too_late=too_late, missed=n_events - tp, timely_rate=timely_rate,
                med_lead=med, fp=fp, fp_rate=fp_rate)


for model, paths in DATASETS.items():
    print("=" * 80)
    print(f"{model}")
    print("=" * 80)

    daily = pd.read_csv(paths["daily"], encoding="utf-8-sig", parse_dates=["date"])
    plug_col = daily.columns[1] if daily.columns[0] == "date" else daily.columns[0]
    daily["machine_id"] = daily[plug_col].str.rsplit("_", n=1).str[0]
    op = daily[daily["is_operating"] == 1].copy()
    op["cross_thr"] = op["baseline"] + 16

    crossing_events = (
        op[op["daily_max"] >= op["cross_thr"]]
        .groupby(["machine_id", "gen_no"])["date"].min()
        .reset_index().rename(columns={"date": "first_crossing_date"})
    )
    n_events = len(crossing_events)

    gen_periods = (
        op.groupby(["machine_id", "gen_no"])["date"]
        .agg(gen_start="min", gen_end="max").reset_index()
    )
    no_cross_gens = gen_periods.merge(
        crossing_events[["machine_id", "gen_no"]],
        on=["machine_id", "gen_no"], how="left", indicator=True
    )
    no_cross_gens = no_cross_gens[no_cross_gens["_merge"] == "left_only"].drop(columns="_merge")
    n_nc = len(no_cross_gens)

    print(f"  Loading cleaned data for {model}...")
    machine_daily = load_machine_daily(paths, plug_col, daily)

    hdr = (f"  {'win':>4}  {'thr':>6}  {'recall':>7}  {'timely%':>8}  "
           f"{'t_early':>8}  {'t_late':>7}  {'missed':>7}  {'fp':>5}  {'fp%':>6}  {'med':>6}")

    best_p = {}
    for level_n in [13, 14, 15]:
        results = []
        for w, thr in product(WINDOWS, RATE_THRESHOLDS):
            col = f"rr{level_n}_w{w}"
            r = eval_signal(col, machine_daily, crossing_events, gen_periods,
                            no_cross_gens, n_events, n_nc, thr)
            results.append({**r, "window": w, "threshold": thr})
        grid = pd.DataFrame(results)
        out_csv = f"outputs/relative_rate_grid_{model}_lv{level_n}.csv"
        grid.to_csv(out_csv, index=False, encoding="utf-8-sig")

        top = grid.sort_values(["recall", "fp", "timely_rate"],
                               ascending=[False, True, False]).head(5)
        b = top.iloc[0]
        best_p[level_n] = (int(b.window), b.threshold)

        label = {13: "Lv1 (bl+13, 3kV手前)", 14: "Lv2 (bl+14, 2kV手前)", 15: "Lv3 (bl+15, 1kV手前)"}[level_n]
        print(f"\n  [{label}] 上位5件")
        print(hdr)
        for _, r in top.iterrows():
            print(f"  {int(r.window):>4}  {r.threshold:>6.2f}  {r.recall:>6.1f}%  "
                  f"{r.timely_rate:>7.1f}%  {int(r.too_early):>8}  {int(r.too_late):>7}  "
                  f"{int(r.missed):>7}  {int(r.fp):>5}  {r.fp_rate:>5.1f}%  {r.med_lead:>5.0f}d")

    # 多段アラート評価
    print(f"\n  [多段アラート: 各レベルのベストパラメータ]")
    for n in [13, 14, 15]:
        print(f"    Lv{n-12} (bl+{n}): window={best_p[n][0]}d  threshold={best_p[n][1]:.2f}")

    tp = fp = timely = too_early = too_late = 0
    leads = []
    escalation_counts = {1: 0, 2: 0, 3: 0}

    for _, ev in crossing_events.iterrows():
        machine, gen_no, cross_date = ev["machine_id"], ev["gen_no"], ev["first_crossing_date"]
        gen_row = gen_periods[(gen_periods["machine_id"] == machine) & (gen_periods["gen_no"] == gen_no)]
        gen_start = gen_row["gen_start"].iloc[0] if not gen_row.empty else pd.Timestamp.min
        pre = machine_daily[
            (machine_daily["machine_id"] == machine)
            & (machine_daily["date"] >= gen_start)
            & (machine_daily["date"] < cross_date)
            & (machine_daily["daily_max"] < machine_daily["cross_thr"])
        ]
        first_dates = {}
        for level_n in [13, 14, 15]:
            w, thr = best_p[level_n]
            col = f"rr{level_n}_w{w}"
            alerts = pre[pre[col] > thr]
            if not alerts.empty:
                first_dates[level_n] = alerts["date"].min()

        if first_dates:
            tp += 1
            earliest_n = min(first_dates, key=lambda n: first_dates[n])
            earliest_date = first_dates[earliest_n]
            escalation_counts[earliest_n - 12] += 1
            lead = (cross_date - earliest_date).days
            leads.append(lead)
            if T_MIN <= lead <= T_MAX:
                timely += 1
            elif lead < T_MIN:
                too_late += 1
            else:
                too_early += 1

    for _, nc in no_cross_gens.iterrows():
        machine, gs, ge = nc["machine_id"], nc["gen_start"], nc["gen_end"]
        period = machine_daily[
            (machine_daily["machine_id"] == machine)
            & (machine_daily["date"] >= gs)
            & (machine_daily["date"] <= ge)
        ]
        if any((period[f"rr{n}_w{best_p[n][0]}"] > best_p[n][1]).any() for n in [13, 14, 15]):
            fp += 1

    recall = tp / n_events * 100
    timely_rate = timely / n_events * 100
    med = sorted(leads)[len(leads) // 2] if leads else 0
    fp_rate = fp / n_nc * 100 if n_nc else 0
    print(f"\n  recall={recall:.1f}%  timely={timely_rate:.1f}%  "
          f"too_early={too_early}  too_late={too_late}  missed={n_events-tp}  "
          f"fp={fp}({fp_rate:.1f}%)  med_lead={med}d")
    print(f"  初回発火レベル: Lv1={escalation_counts[1]}件  Lv2={escalation_counts[2]}件  Lv3={escalation_counts[3]}件")
    print()
