"""
多段時間差シグナル — 早期レベルの発火タイミングを利用したタイムリー化

EP400G の too_late=5 を削減する目的。早期レベル (bl+11/12/13) は「越境のはるか前」
から発火するが too_early に振れる。これを活かす 2 方式を評価する。

方式 A (TSF: Time Since First fire):
  - 早期レベル L1 が初めて発火した日 t0 を記録
  - アラート: t = t0 + delay_days  (delay_days を timely 窓に合わせて選択)
  - 発火条件: L1 シグナル > thr_L1 が観測された日から D 日後

方式 B (ML: Multi-Level confirmation):
  - L1 シグナルが過去 D 日以内に発火 AND 今日 L2 シグナル > thr_L2
  - L1 を「先行確認」として FP 削減
  - 「L1 で予兆 → L2 で確定」の自然な多段化

ベースライン (exp-014):
  - EP370G: slope_3d(bl+14) > 0.100  → timely=54.5%, too_early=11, missed=6
  - EP400G: rate_3d(bl+14) > 0.30   → timely=28.6%, too_early=0, too_late=5, missed=5
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from itertools import product

DATASETS = {
    "EP370G": {"cleaned": "outputs/cleaned_ep370g_v5.csv", "daily": "outputs/dataset_ep370g_ts_v7.csv"},
    "EP400G": {"cleaned": "outputs/cleaned_ep400g_v3.csv", "daily": "outputs/dataset_ep400g_ts_v3.csv"},
}
LEVELS = [11, 12, 13, 14]
WINDOWS = [3, 5, 7, 10]  # 事前計算するウィンドウ集合（rrate / slope）
ML_WINDOWS = [3, 7]  # ML 用は計算量制御で 2 値のみ
TSF_WINDOWS = [3, 5, 7, 10]  # TSF 用
RATE_THRESHOLDS = [0.10, 0.20, 0.30]
SLOPE_THRESHOLDS = [0.01, 0.02, 0.05, 0.10]

# 方式 A 用 delay_days
TSF_DELAYS = [3, 5, 7, 10, 14, 21, 28]

# 方式 B 用 D (履歴ウィンドウ)
ML_HISTORY_DAYS = [7, 14, 21]

MIN_OP_HOURS = 1.0
T_MIN, T_MAX = 7, 14

BASELINE = {
    "EP370G": {"rule": "slope_3d(bl+14)>0.100", "timely_rate": 54.5, "too_early": 11,
               "too_late": 3, "missed": 6, "fp_rate": 17.9, "med_lead": 11},
    "EP400G": {"rule": "rate_3d(bl+14)>0.30", "timely_rate": 28.6, "too_early": 0,
               "too_late": 5, "missed": 5, "fp_rate": 3.8, "med_lead": 3},
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


def _classify_pre(pre, alert_dates, cross_date):
    """alert_dates から timely/too_early/too_late を分類して返す"""
    if alert_dates.empty:
        return None  # missed
    first = alert_dates.min()
    lead = (cross_date - first).days
    return {"first": first, "lead": lead}


def build_nc_end_dates(no_cross_gens):
    return [pd.Timestamp(nc["gen_end"]) for _, nc in no_cross_gens.iterrows()]


def eval_tsf(early_fire_arr, delay, cross_indices, nc_indices, nc_end_dates,
             date_arr, n_events, n_nc):
    """方式 A: 早期発火日 + delay 日後にアラート（事前計算版）"""
    tp = fp = timely = too_early = too_late = 0
    leads = []
    delay_td = pd.Timedelta(days=delay)

    for cross_date, idx in cross_indices:
        if len(idx) == 0:
            continue
        sub = early_fire_arr[idx]
        if not (sub > 0).any():
            continue
        fire_idx = idx[sub > 0]
        first_fire = pd.Timestamp(date_arr[fire_idx].min())
        alert_date = first_fire + delay_td
        if alert_date >= cross_date:
            continue
        tp += 1
        lead = (cross_date - alert_date).days
        leads.append(lead)
        if T_MIN <= lead <= T_MAX:
            timely += 1
        elif lead < T_MIN:
            too_late += 1
        else:
            too_early += 1
    for idx, ge in zip(nc_indices, nc_end_dates):
        if len(idx) == 0:
            continue
        sub = early_fire_arr[idx]
        if not (sub > 0).any():
            continue
        fire_idx = idx[sub > 0]
        first_fire = pd.Timestamp(date_arr[fire_idx].min())
        alert_date = first_fire + delay_td
        if alert_date <= ge:
            fp += 1
    recall = tp / n_events * 100
    timely_rate = timely / n_events * 100
    med = sorted(leads)[len(leads) // 2] if leads else 0
    fp_rate = fp / n_nc * 100 if n_nc else 0
    return dict(tp=tp, recall=recall, timely=timely, too_early=too_early,
                too_late=too_late, missed=n_events - tp, timely_rate=timely_rate,
                med_lead=med, fp=fp, fp_rate=fp_rate)


def precompute_l1_history(machine_daily, l1_col, thr_l1, history_days):
    """L1 発火履歴を事前計算（過去 D 日以内に L1 が発火したか）"""
    s = (machine_daily[l1_col] > thr_l1).astype(int)
    return (
        s.groupby(machine_daily["machine_id"])
        .transform(lambda x: x.rolling(history_days, min_periods=1).max())
    )


def build_event_indices(machine_daily, crossing_events, gen_periods, no_cross_gens):
    """各イベント・各非越境サイクルの md 内 row index を事前計算"""
    machine_arr = machine_daily["machine_id"].values
    date_arr = machine_daily["date"].values
    daily_max_arr = machine_daily["daily_max"].values
    cross_thr_arr = machine_daily["cross_thr"].values

    cross_indices = []  # list of (cross_date, np.array of indices)
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


def eval_ml_precomputed(l1_history, l2_fire, cross_indices, nc_indices, date_arr,
                         n_events, n_nc):
    combined_arr = (l1_history.values * l2_fire.values)
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


def print_top(grid, label, sort_by, format_fn, n=5):
    top = grid.sort_values(sort_by[0], ascending=sort_by[1]).head(n)
    print(f"\n  [{label}] 上位 {n} 件")
    for _, r in top.iterrows():
        print(format_fn(r))


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

    print(f"  events={n_events}  no_crossing_cycles={n_nc}", flush=True)
    print(f"  Loading 30-min cleaned data...", flush=True)
    md = load_machine_daily(paths, plug_col, daily)
    print(f"  Building event indices...", flush=True)
    cross_indices, nc_indices, date_arr = build_event_indices(md, crossing_events, gen_periods, no_cross_gens)

    # ===== 方式 A: TSF (Time Since First fire) =====
    print(f"\n  ----- 方式 A: TSF (early fire + delay days) -----", flush=True)
    nc_end_dates = build_nc_end_dates(no_cross_gens)
    tsf_results = []
    # 早期発火フラグを (col, thr) 単位でキャッシュ
    early_fire_cache = {}
    def get_early_fire(col, thr):
        key = (col, thr)
        if key not in early_fire_cache:
            early_fire_cache[key] = (md[col] > thr).astype(int).values
        return early_fire_cache[key]

    for early_n in [11, 12, 13]:
        for sig_type, col_prefix, thresholds in [
            ("rate", "rrate", RATE_THRESHOLDS),
            ("slope", "slope", SLOPE_THRESHOLDS),
        ]:
            for w, thr, delay in product(TSF_WINDOWS, thresholds, TSF_DELAYS):
                col = f"{col_prefix}{early_n}_w{w}"
                early_fire = get_early_fire(col, thr)
                r = eval_tsf(early_fire, delay, cross_indices, nc_indices, nc_end_dates,
                             date_arr, n_events, n_nc)
                tsf_results.append({
                    **r, "early_lv": early_n, "sig": sig_type,
                    "early_window": w, "early_thr": thr, "delay": delay,
                })
    print(f"  TSF combinations evaluated: {len(tsf_results)}", flush=True)
    tsf_grid = pd.DataFrame(tsf_results)
    tsf_grid.to_csv(f"outputs/multistage_tsf_{model}.csv", index=False, encoding="utf-8-sig")

    # 表示用フォーマット
    def fmt_tsf(r):
        return (f"  lv{int(r.early_lv):>2} {r.sig:>5} w={int(r.early_window):>2} "
                f"thr={r.early_thr:>5.3f} +{int(r.delay):>2}d  "
                f"recall={r.recall:>5.1f}%  timely={r.timely_rate:>5.1f}%  "
                f"t_early={int(r.too_early):>2} t_late={int(r.too_late):>2} "
                f"missed={int(r.missed):>2}  fp={int(r.fp):>2}({r.fp_rate:>4.1f}%)  med={r.med_lead:>3.0f}d")

    print(f"  {'lv':>2} {'sig':>5} {'win':>4} {'thr':>9} {'+d':>4}  metrics...")
    print_top(tsf_grid, f"{model} TSF: timely% 最大",
              sort_by=(["timely_rate", "too_early", "too_late"], [False, True, True]),
              format_fn=fmt_tsf, n=8)
    print_top(tsf_grid, f"{model} TSF: too_late 最小化（recall>=64%, too_early<=5）",
              sort_by=(["too_late", "timely_rate", "too_early"], [True, False, True]),
              format_fn=fmt_tsf, n=5)

    # ===== 方式 B: ML (Multi-Level confirmation) =====
    print(f"\n  ----- 方式 B: ML (L1 history + L2 today) -----")
    ml_results = []
    L1_LEVELS = [11, 12, 13]
    L2_LEVELS = [13, 14]

    # L1 履歴を事前計算（同じ (l1_col, thr, history) なら使い回す）
    l1_cache = {}
    def get_l1_history(col, thr, history):
        key = (col, thr, history)
        if key not in l1_cache:
            l1_cache[key] = precompute_l1_history(md, col, thr, history)
        return l1_cache[key]

    # L2 発火フラグを事前計算
    l2_cache = {}
    def get_l2_fire(col, thr):
        key = (col, thr)
        if key not in l2_cache:
            l2_cache[key] = (md[col] > thr).astype(int)
        return l2_cache[key]

    for l1_n, l2_n in product(L1_LEVELS, L2_LEVELS):
        if l1_n >= l2_n:
            continue
        for l1_sig, l1_prefix, l1_thrs in [("rate", "rrate", RATE_THRESHOLDS),
                                            ("slope", "slope", SLOPE_THRESHOLDS)]:
            for l2_sig, l2_prefix, l2_thrs in [("rate", "rrate", RATE_THRESHOLDS),
                                                ("slope", "slope", SLOPE_THRESHOLDS)]:
                for l1_w, l1_thr, l2_w, l2_thr, history in product(
                    ML_WINDOWS, l1_thrs, ML_WINDOWS, l2_thrs, ML_HISTORY_DAYS
                ):
                    l1_col = f"{l1_prefix}{l1_n}_w{l1_w}"
                    l2_col = f"{l2_prefix}{l2_n}_w{l2_w}"
                    l1_hist = get_l1_history(l1_col, l1_thr, history)
                    l2_fire = get_l2_fire(l2_col, l2_thr)
                    r = eval_ml_precomputed(l1_hist, l2_fire, cross_indices, nc_indices,
                                             date_arr, n_events, n_nc)
                    ml_results.append({
                        **r,
                        "l1_lv": l1_n, "l1_sig": l1_sig, "l1_w": l1_w, "l1_thr": l1_thr,
                        "l2_lv": l2_n, "l2_sig": l2_sig, "l2_w": l2_w, "l2_thr": l2_thr,
                        "history": history,
                    })
    print(f"  ML combinations evaluated: {len(ml_results)}", flush=True)
    ml_grid = pd.DataFrame(ml_results)
    ml_grid.to_csv(f"outputs/multistage_ml_{model}.csv", index=False, encoding="utf-8-sig")

    def fmt_ml(r):
        return (f"  L1={r.l1_sig}{int(r.l1_lv)}_w{int(r.l1_w)}>{r.l1_thr:.3f}  "
                f"L2={r.l2_sig}{int(r.l2_lv)}_w{int(r.l2_w)}>{r.l2_thr:.3f}  "
                f"hist={int(r.history)}d  "
                f"recall={r.recall:>5.1f}%  timely={r.timely_rate:>5.1f}%  "
                f"t_early={int(r.too_early):>2} t_late={int(r.too_late):>2} "
                f"missed={int(r.missed):>2}  fp={int(r.fp):>2}({r.fp_rate:>4.1f}%)  med={r.med_lead:>3.0f}d")

    print_top(ml_grid, f"{model} ML: timely% 最大",
              sort_by=(["timely_rate", "too_early", "too_late"], [False, True, True]),
              format_fn=fmt_ml, n=8)
    print_top(ml_grid, f"{model} ML: too_late 最小化",
              sort_by=(["too_late", "timely_rate", "too_early"], [True, False, True]),
              format_fn=fmt_ml, n=5)

    # ===== ベースライン比較 =====
    b = BASELINE[model]
    print(f"\n  --- ベースライン比較 ({model}) ---")
    print(f"  baseline ({b['rule']}): timely={b['timely_rate']}% "
          f"too_early={b['too_early']} too_late={b['too_late']} missed={b['missed']} "
          f"fp%={b['fp_rate']} med={b['med_lead']}d")

    tsf_best = tsf_grid.sort_values(["timely_rate", "too_early", "too_late"],
                                     ascending=[False, True, True]).iloc[0]
    ml_best = ml_grid.sort_values(["timely_rate", "too_early", "too_late"],
                                   ascending=[False, True, True]).iloc[0]
    print(f"  TSF 最良 (timely%): timely={tsf_best.timely_rate:.1f}% "
          f"too_early={int(tsf_best.too_early)} too_late={int(tsf_best.too_late)} "
          f"missed={int(tsf_best.missed)} fp%={tsf_best.fp_rate:.1f} "
          f"med={tsf_best.med_lead:.0f}d")
    print(f"  ML 最良 (timely%):  timely={ml_best.timely_rate:.1f}% "
          f"too_early={int(ml_best.too_early)} too_late={int(ml_best.too_late)} "
          f"missed={int(ml_best.missed)} fp%={ml_best.fp_rate:.1f} "
          f"med={ml_best.med_lead:.0f}d")

    # too_late 削減目線
    if model == "EP400G":
        print(f"\n  --- EP400G too_late 削減候補 (too_late<=2 かつ timely%>=28.6 かつ too_early<=2) ---")
        for grid, name in [(tsf_grid, "TSF"), (ml_grid, "ML")]:
            filt = grid[(grid.too_late <= 2) & (grid.timely_rate >= 28.6) & (grid.too_early <= 2)]
            filt = filt.sort_values(["too_late", "timely_rate", "too_early"],
                                     ascending=[True, False, True])
            if len(filt) == 0:
                print(f"    {name}: 該当なし")
            else:
                print(f"    {name}: 上位 5 件")
                for _, r in filt.head(5).iterrows():
                    if name == "TSF":
                        print(fmt_tsf(r))
                    else:
                        print(fmt_ml(r))
    print()
