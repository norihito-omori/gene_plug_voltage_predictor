"""
加速度シグナル（slope / diff / combined）の timely アラート最適化
越境定義: daily_max >= baseline + 16
仮説: rate の絶対値ではなく上昇傾向で「進行中の急速劣化」を選別 → too_early 削減

シグナル:
  slope    : rate_at_(baseline+N) の W 日線形回帰の傾き
  diff     : rate_at_(baseline+N)(t) - rate_at_(baseline+N)(t-W)
  combined : rate_w3 > thr_rate AND slope_w7 > thr_slope
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from itertools import product

DATASETS = {
    "EP370G": {"cleaned": "outputs/cleaned_ep370g_v5.csv", "daily": "outputs/dataset_ep370g_ts_v7.csv"},
    "EP400G": {"cleaned": "outputs/cleaned_ep400g_v3.csv", "daily": "outputs/dataset_ep400g_ts_v3.csv"},
}
LEVELS = [13, 14]
WINDOWS = [3, 5, 7, 10, 14]
SLOPE_THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
RATE_THRESHOLDS_COMBINED = [0.10, 0.20, 0.30]
SLOPE_THRESHOLDS_COMBINED = [0.005, 0.01, 0.02, 0.03]
COMBINED_RATE_WINDOW = 3
COMBINED_SLOPE_WINDOW = 7

MIN_OP_HOURS = 1.0
T_MIN, T_MAX = 7, 14

# exp-013 ベースライン
BASELINE = {
    "EP370G": {"timely_rate": 38.6, "too_early": 14, "missed": 6, "fp_rate": 17.9, "med_lead": 10},
    "EP400G": {"timely_rate": 28.6, "too_early": 0, "missed": 5, "fp_rate": 3.8, "med_lead": 3},
}


def _slope(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(valid), dtype=float)
    return np.polyfit(x, valid, 1)[0]


def load_machine_daily(paths, plug_col, daily):
    cleaned_raw = pd.read_csv(paths["cleaned"], encoding="utf-8-sig", nrows=0)
    cols = cleaned_raw.columns.tolist()
    power_col, voltage_col, plug_id_col, datetime_col = cols[3], cols[7], cols[9], cols[1]

    cleaned = pd.read_csv(paths["cleaned"], encoding="utf-8-sig",
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

    daily2 = daily.copy()
    daily2["date"] = pd.to_datetime(daily2["date"])
    merged = daily2.merge(
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

    # rate のローリング平均（combined 用、および参考）
    for n in LEVELS:
        md[f"rrate{n}_w{COMBINED_RATE_WINDOW}"] = (
            md.groupby("machine_id")[f"rate{n}"]
            .transform(lambda s: s.rolling(COMBINED_RATE_WINDOW, min_periods=1).mean())
        )
        for w in WINDOWS:
            md[f"slope{n}_w{w}"] = (
                md.groupby("machine_id")[f"rate{n}"]
                .transform(lambda s: s.rolling(w, min_periods=2).apply(_slope, raw=True))
            )
            md[f"diff{n}_w{w}"] = (
                md.groupby("machine_id")[f"rate{n}"]
                .transform(lambda s: s - s.shift(w))
            )
    return md


def eval_single(col, machine_daily, crossing_events, gen_periods, no_cross_gens, n_events, n_nc, thr):
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


def eval_combined(rate_col, slope_col, machine_daily, crossing_events, gen_periods,
                  no_cross_gens, n_events, n_nc, thr_rate, thr_slope):
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
        alerts = pre[(pre[rate_col] > thr_rate) & (pre[slope_col] > thr_slope)]
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
        if ((period[rate_col] > thr_rate) & (period[slope_col] > thr_slope)).any():
            fp += 1
    recall = tp / n_events * 100
    timely_rate = timely / n_events * 100
    med = sorted(leads)[len(leads) // 2] if leads else 0
    fp_rate = fp / n_nc * 100 if n_nc else 0
    return dict(tp=tp, recall=recall, timely=timely, too_early=too_early,
                too_late=too_late, missed=n_events - tp, timely_rate=timely_rate,
                med_lead=med, fp=fp, fp_rate=fp_rate)


def print_top(grid: pd.DataFrame, label: str, n: int = 5):
    top = grid.sort_values(["timely_rate", "too_early", "missed"],
                           ascending=[False, True, True]).head(n)
    print(f"\n  [{label}] 上位 {n} 件")
    print(f"  {'win':>4}  {'thr':>10}  {'recall':>7}  {'timely%':>8}  "
          f"{'t_early':>8}  {'t_late':>7}  {'missed':>7}  {'fp':>5}  {'fp%':>6}  {'med':>6}")
    for _, r in top.iterrows():
        thr_label = (f"{r.threshold_rate:.2f}/{r.threshold_slope:.3f}"
                     if "threshold_rate" in r else f"{r.threshold:.3f}")
        print(f"  {int(r.window):>4}  {thr_label:>10}  {r.recall:>6.1f}%  "
              f"{r.timely_rate:>7.1f}%  {int(r.too_early):>8}  {int(r.too_late):>7}  "
              f"{int(r.missed):>7}  {int(r.fp):>5}  {r.fp_rate:>5.1f}%  {r.med_lead:>5.0f}d")


def best_row(grid: pd.DataFrame) -> pd.Series:
    return grid.sort_values(["timely_rate", "too_early", "missed"],
                            ascending=[False, True, True]).iloc[0]


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

    print(f"  events={n_events}  no_crossing_cycles={n_nc}")
    print(f"  Loading 30-min cleaned data and computing slope/diff features...")
    md = load_machine_daily(paths, plug_col, daily)

    summary_best = {}

    # ----- 単独シグナル: slope / diff -----
    for sig_name, col_prefix, thresholds in [
        ("slope", "slope", SLOPE_THRESHOLDS),
        ("diff", "diff", SLOPE_THRESHOLDS),
    ]:
        for n in LEVELS:
            results = []
            for w, thr in product(WINDOWS, thresholds):
                col = f"{col_prefix}{n}_w{w}"
                r = eval_single(col, md, crossing_events, gen_periods,
                                no_cross_gens, n_events, n_nc, thr)
                results.append({**r, "window": w, "threshold": thr})
            grid = pd.DataFrame(results)
            out_csv = f"outputs/acceleration_grid_{model}_{sig_name}_lv{n}.csv"
            grid.to_csv(out_csv, index=False, encoding="utf-8-sig")
            label = f"{sig_name}_lv{n} (bl+{n})"
            print_top(grid, label)
            summary_best[(sig_name, n)] = best_row(grid)

    # ----- combined シグナル -----
    for n in LEVELS:
        rate_col = f"rrate{n}_w{COMBINED_RATE_WINDOW}"
        slope_col = f"slope{n}_w{COMBINED_SLOPE_WINDOW}"
        results = []
        for tr, ts in product(RATE_THRESHOLDS_COMBINED, SLOPE_THRESHOLDS_COMBINED):
            r = eval_combined(rate_col, slope_col, md, crossing_events, gen_periods,
                              no_cross_gens, n_events, n_nc, tr, ts)
            results.append({**r, "window": COMBINED_SLOPE_WINDOW,
                            "threshold_rate": tr, "threshold_slope": ts})
        grid = pd.DataFrame(results)
        out_csv = f"outputs/acceleration_grid_{model}_combined_lv{n}.csv"
        grid.to_csv(out_csv, index=False, encoding="utf-8-sig")
        label = f"combined_lv{n}  (rate_w{COMBINED_RATE_WINDOW} & slope_w{COMBINED_SLOPE_WINDOW})"
        print_top(grid, label)
        summary_best[("combined", n)] = best_row(grid)

    # ----- ベースライン比較 -----
    print(f"\n  --- exp-013 ベースラインとの比較 ({model}) ---")
    b = BASELINE[model]
    print(f"  {'signal':>14}  {'lv':>3}  {'timely%':>8}  {'t_early':>8}  "
          f"{'missed':>7}  {'fp%':>6}  {'med':>5}")
    print(f"  {'baseline (rate_w3>0.30)':>14}  {'14':>3}  "
          f"{b['timely_rate']:>7.1f}%  {b['too_early']:>8}  "
          f"{b['missed']:>7}  {b['fp_rate']:>5.1f}%  {b['med_lead']:>4}d")
    for (sig, n), r in summary_best.items():
        print(f"  {sig:>14}  {n:>3}  {r.timely_rate:>7.1f}%  {int(r.too_early):>8}  "
              f"{int(r.missed):>7}  {r.fp_rate:>5.1f}%  {r.med_lead:>4.0f}d")

    # 評価ゲート判定（EP370G のみ）
    if model == "EP370G":
        print(f"\n  --- 評価ゲート判定 (EP370G: too_early<14 かつ timely%>=36) ---")
        any_pass = False
        for (sig, n), r in summary_best.items():
            passed = (r.too_early < 14) and (r.timely_rate >= 36)
            mark = "OK" if passed else "--"
            any_pass = any_pass or passed
            print(f"  [{mark}] {sig} lv{n}: timely={r.timely_rate:.1f}%  "
                  f"too_early={int(r.too_early)}")
        print(f"\n  {'PASS' if any_pass else 'FAIL'}: EP370G 評価ゲート")

    print()
