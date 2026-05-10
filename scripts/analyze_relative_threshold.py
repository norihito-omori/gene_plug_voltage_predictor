"""
プラグの個性（ゲタ）を反映した相対閾値 hours_at_(baseline+N) の探索
  absolute: 要求電圧 == 31 (hours_at_31kv)
  relative: 要求電圧 == round(baseline) + N  (N = 8, 9, 10, 11)

比較評価軸:
  - recall (越境検知率)
  - FP率
  - 先行中央値 (初回アラートから越境までの日数)
  - タイムリー率 (初回アラートが越境 7-14日前に収まる割合)
"""
from __future__ import annotations
import argparse
import pandas as pd

DATASETS = {
    "EP370G": {
        "cleaned": "outputs/cleaned_ep370g_v5.csv",
        "daily": "outputs/dataset_ep370g_ts_v7.csv",
    },
    "EP400G": {
        "cleaned": "outputs/cleaned_ep400g_v2.csv",
        "daily": "outputs/dataset_ep400g_ts_v2.csv",
    },
}
THRESHOLD_CROSS = 32.0
# ウィンドウ×閾値 (最適パラメータのみ比較)
WINDOWS = [3, 7, 21]
RULE_THRESHOLDS = [0.5, 2, 5]
RELATIVE_N = [8, 9, 10, 11]
T_MIN, T_MAX = 7, 14

# 各列の意味 (CSVの列番号):
# [3] 発電機電力  [7] 要求電圧

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["EP370G", "EP400G", "both"], default="both")
args = parser.parse_args()
targets = ["EP370G", "EP400G"] if args.model == "both" else [args.model]

for model in targets:
    print("=" * 80)
    print(f"{model}")
    print("=" * 80)

    cleaned_path = DATASETS[model]["cleaned"]
    daily_path = DATASETS[model]["daily"]

    daily = pd.read_csv(daily_path, encoding="utf-8-sig", parse_dates=["date"])
    daily["machine_id"] = daily.iloc[:, 1].str.rsplit("_", n=1).str[0]  # 管理No_プラグNo

    # 列名を位置で取得（文字化け対策）
    cleaned_raw = pd.read_csv(cleaned_path, encoding="utf-8-sig", nrows=0)
    cols = cleaned_raw.columns.tolist()
    power_col = cols[3]    # 発電機電力
    voltage_col = cols[7]  # 要求電圧
    plug_id_col = cols[9]  # 管理No_プラグNo
    datetime_col = cols[1] # dailygraphpt_ptdatetime

    print(f"  voltage_col=[{7}] power_col=[{3}] plug_id=[{9}]")

    # 30分データを読み込み（必要列のみ）
    print("  Loading cleaned 30-min data...")
    cleaned = pd.read_csv(
        cleaned_path, encoding="utf-8-sig",
        usecols=[datetime_col, power_col, voltage_col, plug_id_col, "gen_no", "baseline"]
    )
    cleaned["_date"] = pd.to_datetime(cleaned[datetime_col]).dt.normalize()
    cleaned["_running"] = cleaned[power_col] > 0

    # 稼働中スロットのみ
    running = cleaned[cleaned["_running"]].copy()
    print(f"  Running slots: {len(running):,}")

    # 絶対閾値: 要求電圧 == 31
    running["_at_abs31"] = (running[voltage_col] == 31).astype(float)

    # 相対閾値: 要求電圧 == round(baseline) + N
    for n in RELATIVE_N:
        running[f"_at_rel{n}"] = (running[voltage_col] == running["baseline"].round() + n).astype(float)

    # 日次集計: plug_id × date
    agg_dict = {"_at_abs31": "sum"}
    for n in RELATIVE_N:
        agg_dict[f"_at_rel{n}"] = "sum"
    hourly = (
        running.groupby([plug_id_col, "_date"])
        .agg(agg_dict)
        .mul(0.5)
        .reset_index()
        .rename(columns={"_date": "date"})
    )
    hourly["date"] = pd.to_datetime(hourly["date"])

    # daily との結合
    daily["date"] = pd.to_datetime(daily["date"])
    daily_id_col = daily.columns[1]  # 管理No_プラグNo in daily
    merged = daily.merge(hourly, left_on=[daily_id_col, "date"], right_on=[plug_id_col, "date"], how="left")
    for col in ["_at_abs31"] + [f"_at_rel{n}" for n in RELATIVE_N]:
        merged[col] = merged[col].fillna(0.0)

    # 運転中のみ
    op = merged[merged["is_operating"] == 1].copy()

    crossing_events = (
        op[op["daily_max"] >= THRESHOLD_CROSS]
        .groupby(["machine_id", "gen_no"])["date"].min()
        .reset_index().rename(columns={"date": "first_crossing_date"})
    )
    n_events = len(crossing_events)

    machine_daily = (
        op.groupby(["machine_id", "date"])
        .agg(
            daily_max=("daily_max", "max"),
            **{col: (col, "max") for col in ["_at_abs31"] + [f"_at_rel{n}" for n in RELATIVE_N]}
        )
        .reset_index().sort_values(["machine_id", "date"])
    )

    gen_periods = (
        op.groupby(["machine_id", "gen_no"])["date"]
        .agg(gen_start="min", gen_end="max").reset_index()
    )
    no_crossing_gens = gen_periods.merge(
        crossing_events[["machine_id", "gen_no"]],
        on=["machine_id", "gen_no"], how="left", indicator=True
    )
    no_crossing_gens = no_crossing_gens[no_crossing_gens["_merge"] == "left_only"].drop(columns="_merge")
    n_nc = len(no_crossing_gens)

    # ローリング合計を事前計算
    signal_cols = ["_at_abs31"] + [f"_at_rel{n}" for n in RELATIVE_N]
    for w in WINDOWS:
        for sc in signal_cols:
            col = f"{sc}_w{w}"
            machine_daily[col] = (
                machine_daily.groupby("machine_id")[sc]
                .transform(lambda s: s.rolling(w, min_periods=1).sum())
            )

    # 評価
    results = []
    for signal_name, base_col in [("abs31", "_at_abs31")] + [(f"rel{n}", f"_at_rel{n}") for n in RELATIVE_N]:
        for w in WINDOWS:
            for thr in RULE_THRESHOLDS:
                roll_col = f"{base_col}_w{w}"
                tp = fp = timely = too_early = too_late = 0
                first_leads = []

                for _, ev in crossing_events.iterrows():
                    machine, gen_no, cross_date = ev["machine_id"], ev["gen_no"], ev["first_crossing_date"]
                    gen_start_row = gen_periods[
                        (gen_periods["machine_id"] == machine) & (gen_periods["gen_no"] == gen_no)
                    ]
                    gen_start = gen_start_row["gen_start"].iloc[0] if not gen_start_row.empty else pd.Timestamp.min
                    pre = machine_daily[
                        (machine_daily["machine_id"] == machine)
                        & (machine_daily["date"] >= gen_start)
                        & (machine_daily["date"] < cross_date)
                        & (machine_daily["daily_max"] < THRESHOLD_CROSS)
                    ]
                    alerts = pre[pre[roll_col] > thr]
                    if not alerts.empty:
                        tp += 1
                        first_lead = (cross_date - alerts["date"].min()).days
                        first_leads.append(first_lead)
                        if T_MIN <= first_lead <= T_MAX:
                            timely += 1
                        elif first_lead < T_MIN:
                            too_late += 1
                        else:
                            too_early += 1

                for _, nc in no_crossing_gens.iterrows():
                    machine, gen_start, gen_end = nc["machine_id"], nc["gen_start"], nc["gen_end"]
                    period = machine_daily[
                        (machine_daily["machine_id"] == machine)
                        & (machine_daily["date"] >= gen_start)
                        & (machine_daily["date"] <= gen_end)
                    ]
                    if (period[roll_col] > thr).any():
                        fp += 1

                recall = tp / n_events * 100
                timely_rate = timely / n_events * 100
                med_lead = sorted(first_leads)[len(first_leads) // 2] if first_leads else 0
                fp_rate = fp / n_nc * 100 if n_nc else 0
                missed = n_events - tp
                results.append({
                    "signal": signal_name, "window": w, "threshold": thr,
                    "tp": tp, "recall": recall,
                    "timely": timely, "too_early": too_early, "too_late": too_late, "missed": missed,
                    "timely_rate": timely_rate, "med_lead": med_lead,
                    "fp": fp, "fp_rate": fp_rate,
                })

    grid = pd.DataFrame(results)
    out_csv = f"outputs/relative_threshold_grid_{model}.csv"
    grid.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n  events={n_events}  no_crossing_cycles={n_nc}")
    print(f"  Timely window: {T_MIN}-{T_MAX} days before crossing")
    print()

    # 各シグナル×ウィンドウで最良設定(recall最大→fp最小)を表示
    hdr = f"  {'signal':>8}  {'win':>4}  {'thr':>5}  {'recall':>7}  {'timely%':>8}  {'too_early':>10}  {'too_late':>9}  {'missed':>7}  {'fp':>5}  {'fp%':>6}  {'med_lead':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for signal_name in ["abs31"] + [f"rel{n}" for n in RELATIVE_N]:
        sg = grid[grid["signal"] == signal_name]
        # best: recall max -> fp min -> timely_rate max
        best = sg.sort_values(["recall", "fp", "timely_rate"], ascending=[False, True, False]).iloc[0]
        print(f"  {best.signal:>8}  {int(best.window):>4}  {best.threshold:>5.1f}  {best.recall:>6.1f}%  {best.timely_rate:>7.1f}%  {int(best.too_early):>10}  {int(best.too_late):>9}  {int(best.missed):>7}  {int(best.fp):>5}  {best.fp_rate:>5.1f}%  {best.med_lead:>8.0f}d")

    print()
    print(f"  Full grid saved: {out_csv}")
    print()
