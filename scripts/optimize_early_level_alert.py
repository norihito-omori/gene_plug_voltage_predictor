"""
早期レベル（bl+11, bl+12, bl+13）のシグナル探索 — EP400G の too_late 削減目的

exp-014 で EP400G は bl+14 シグナルでは too_late=5 が解消できないことが判明。
リードタイムを稼ぐためにより早期の電圧帯シグナルを評価する。

評価:
  - rate（exp-013 のローリング平均）
  - slope（exp-014 の加速度）
  各シグナル × N=11/12/13 × ウィンドウ × 閾値
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from itertools import product

DATASETS = {
    "EP370G": {"cleaned": "outputs/cleaned_ep370g_v5.csv", "daily": "outputs/dataset_ep370g_ts_v7.csv"},
    "EP400G": {"cleaned": "outputs/cleaned_ep400g_v3.csv", "daily": "outputs/dataset_ep400g_ts_v3.csv"},
}
LEVELS = [11, 12, 13]
WINDOWS = [3, 5, 7, 10, 14, 21]
RATE_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]
SLOPE_THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

MIN_OP_HOURS = 1.0
T_MIN, T_MAX = 7, 14

# exp-013/014 ベースライン (EP400G)
BASELINE_EP400G = {
    "rule": "rate_at_(bl+14) の 3日 rolling mean > 0.30",
    "timely_rate": 28.6, "too_early": 0, "too_late": 5, "missed": 5,
    "fp_rate": 3.8, "med_lead": 3,
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


def print_top(grid, label, sort_by, n=8):
    top = grid.sort_values(sort_by[0], ascending=sort_by[1]).head(n)
    print(f"\n  [{label}] 上位 {n} 件 ({'/'.join(sort_by[0])} で並び替え)")
    print(f"  {'win':>4}  {'thr':>6}  {'recall':>7}  {'timely%':>8}  "
          f"{'t_early':>8}  {'t_late':>7}  {'missed':>7}  {'fp':>5}  {'fp%':>6}  {'med':>6}")
    for _, r in top.iterrows():
        print(f"  {int(r.window):>4}  {r.threshold:>6.3f}  {r.recall:>6.1f}%  "
              f"{r.timely_rate:>7.1f}%  {int(r.too_early):>8}  {int(r.too_late):>7}  "
              f"{int(r.missed):>7}  {int(r.fp):>5}  {r.fp_rate:>5.1f}%  {r.med_lead:>5.0f}d")


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
    print(f"  Loading 30-min cleaned data...")
    md = load_machine_daily(paths, plug_col, daily)

    summary = {}
    for sig_name, col_prefix, thresholds in [
        ("rate", "rrate", RATE_THRESHOLDS),
        ("slope", "slope", SLOPE_THRESHOLDS),
    ]:
        for n in LEVELS:
            results = []
            for w, thr in product(WINDOWS, thresholds):
                col = f"{col_prefix}{n}_w{w}"
                r = eval_single(col, md, crossing_events, gen_periods,
                                no_cross_gens, n_events, n_nc, thr)
                results.append({**r, "window": w, "threshold": thr})
            grid = pd.DataFrame(results)
            out_csv = f"outputs/early_level_grid_{model}_{sig_name}_lv{n}.csv"
            grid.to_csv(out_csv, index=False, encoding="utf-8-sig")

            label = f"{sig_name}_lv{n} (bl+{n})"
            # too_late 削減を最重要視（EP400G の課題対応）→ too_late asc, then timely% desc
            print_top(grid, label,
                      sort_by=(["too_late", "timely_rate", "too_early"], [True, False, True]),
                      n=5)
            summary[(sig_name, n)] = grid

    # EP400G に集中して最良候補を抽出
    if model == "EP400G":
        print(f"\n  ====== EP400G too_late 削減候補 ======")
        print(f"  baseline: {BASELINE_EP400G['rule']}")
        print(f"  baseline: timely={BASELINE_EP400G['timely_rate']}% too_early={BASELINE_EP400G['too_early']} "
              f"too_late={BASELINE_EP400G['too_late']} missed={BASELINE_EP400G['missed']} "
              f"fp%={BASELINE_EP400G['fp_rate']} med={BASELINE_EP400G['med_lead']}d")

        print(f"\n  --- too_late <= 3 かつ recall >= 64.3% かつ too_early <= 2 ---")
        print(f"  {'sig':>6} {'lv':>3} {'win':>4} {'thr':>6}  {'recall':>7}  {'timely%':>8}  "
              f"{'t_early':>8}  {'t_late':>7}  {'missed':>7}  {'fp%':>6}  {'med':>5}")
        any_found = False
        for (sig, n), grid in summary.items():
            filt = grid[(grid.too_late <= 3) & (grid.recall >= 64.3) & (grid.too_early <= 2)]
            filt = filt.sort_values(["too_late", "timely_rate", "too_early"], ascending=[True, False, True])
            for _, r in filt.head(3).iterrows():
                any_found = True
                print(f"  {sig:>6} {n:>3} {int(r.window):>4} {r.threshold:>6.3f}  "
                      f"{r.recall:>6.1f}%  {r.timely_rate:>7.1f}%  {int(r.too_early):>8}  "
                      f"{int(r.too_late):>7}  {int(r.missed):>7}  {r.fp_rate:>5.1f}%  {r.med_lead:>4.0f}d")
        if not any_found:
            print(f"  該当なし（too_late <= 3 を満たすルールが存在しない）")

        print(f"\n  --- timely_rate 最大（参考） ---")
        print(f"  {'sig':>6} {'lv':>3} {'win':>4} {'thr':>6}  {'recall':>7}  {'timely%':>8}  "
              f"{'t_early':>8}  {'t_late':>7}  {'missed':>7}  {'fp%':>6}  {'med':>5}")
        for (sig, n), grid in summary.items():
            top = grid.sort_values(["timely_rate", "too_late", "too_early"],
                                   ascending=[False, True, True]).head(2)
            for _, r in top.iterrows():
                print(f"  {sig:>6} {n:>3} {int(r.window):>4} {r.threshold:>6.3f}  "
                      f"{r.recall:>6.1f}%  {r.timely_rate:>7.1f}%  {int(r.too_early):>8}  "
                      f"{int(r.too_late):>7}  {int(r.missed):>7}  {r.fp_rate:>5.1f}%  {r.med_lead:>4.0f}d")
    print()
