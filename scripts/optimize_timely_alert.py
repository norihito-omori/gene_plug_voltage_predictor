"""
タイムリーアラート最適化
  評価軸: アラートが「初めて発火した日」が越境の T_MIN〜T_MAX 日前に収まるか。

アラートは一度発火すると越境直前まで出続けるため、
サービスマンが受け取る有効な情報は「初回発火日」のみ。
  - 初回発火 > T_MAX 日前  → too_early（早すぎ、不要交換リスク）
  - T_MIN <= 初回発火 <= T_MAX 日前 → timely（理想ウィンドウ）
  - 初回発火 < T_MIN 日前  → too_late（手配間に合わない）
  - アラートなし           → missed（越境見逃し）
"""
from __future__ import annotations
import argparse
import pandas as pd

DATASETS = {
    "EP370G": "outputs/dataset_ep370g_ts_v7.csv",
    "EP400G": "outputs/dataset_ep400g_ts_v3.csv",
}
THRESHOLD_CROSS = 32.0
WINDOWS = [3, 5, 7, 10, 14, 21, 30]
THRESHOLDS = [0.5, 1, 2, 3, 5, 7, 10, 14, 21]

# タイムリー判定ウィンドウ
T_MIN = 7    # 越境 7 日前以内に入ったアラートは「遅すぎ」
T_MAX = 14   # 越境 14 日前より前のアラートは「早すぎ」

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["EP370G", "EP400G", "both"], default="both")
parser.add_argument("--t-min", type=int, default=T_MIN)
parser.add_argument("--t-max", type=int, default=T_MAX)
args = parser.parse_args()
T_MIN, T_MAX = args.t_min, args.t_max
targets = ["EP370G", "EP400G"] if args.model == "both" else [args.model]

for model in targets:
    df = pd.read_csv(DATASETS[model], encoding="utf-8-sig", parse_dates=["date"])
    df["machine_id"] = df["管理No_プラグNo"].str.rsplit("_", n=1).str[0]
    op = df[df["is_operating"] == 1].copy()

    crossing_events = (
        op[op["daily_max"] >= THRESHOLD_CROSS]
        .groupby(["machine_id", "gen_no"])["date"].min()
        .reset_index().rename(columns={"date": "first_crossing_date"})
    )
    n_events = len(crossing_events)

    machine_daily = (
        op.groupby(["machine_id", "date"])
        .agg(daily_max=("daily_max", "max"), hours_at_31kv=("hours_at_31kv", "max"))
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

    for w in WINDOWS:
        col = f"h{w}d"
        machine_daily[col] = (
            machine_daily.groupby("machine_id")["hours_at_31kv"]
            .transform(lambda s: s.rolling(w, min_periods=1).sum())
        )

    results = []
    for w in WINDOWS:
        col = f"h{w}d"
        for thr in THRESHOLDS:
            tp = fp = timely = too_early = too_late = 0
            last_leads = []

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
                alerts = pre[pre[col] > thr]
                if not alerts.empty:
                    tp += 1
                    first_alert_date = alerts["date"].min()  # 初回発火日
                    first_lead = (cross_date - first_alert_date).days
                    last_leads.append(first_lead)
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
                if (period[col] > thr).any():
                    fp += 1

            recall = tp / n_events * 100
            timely_rate = timely / n_events * 100  # 全越境イベント中のタイムリー率
            med_last = sorted(last_leads)[len(last_leads) // 2] if last_leads else 0
            avg_last = sum(last_leads) / len(last_leads) if last_leads else 0
            missed = n_events - tp
            fp_rate = fp / n_nc * 100 if n_nc else 0
            results.append({
                "window": w, "threshold": thr,
                "tp": tp, "n_events": n_events, "recall": recall,
                "timely": timely, "too_early": too_early, "too_late": too_late,
                "timely_rate": timely_rate,
                "fp": fp, "n_nc": n_nc, "fp_rate": fp_rate,
                "avg_last_lead": avg_last, "med_last_lead": med_last,
            })

    grid = pd.DataFrame(results)
    out_csv = f"outputs/timely_alert_grid_{model}.csv"
    grid.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"{model}  越境{n_events}件 / 非越境{n_nc}サイクル")
    print(f"  タイムリー定義: 初回アラートが越境の {T_MIN}〜{T_MAX} 日前")
    print(f"  too_early = {T_MAX}日超前  timely = {T_MIN}-{T_MAX}日前  too_late = {T_MIN}日未満  missed = アラートなし")
    print("=" * 80)

    # 全結果: timely_rate 最大 -> too_early 最小 -> fp 最小
    top_a = grid.sort_values(["timely_rate", "too_early", "fp"], ascending=[False, True, True])
    print(f"\n[全グリッド上位: タイムリー率最大優先]")
    print(f"  {'win':>4}  {'thr':>5}  {'timely':>9}  {'t_rate':>7}  {'too_early':>10}  {'too_late':>9}  {'missed':>7}  {'fp':>5}  {'fp%':>6}  {'med_1st':>8}")
    for _, r in top_a.head(15).iterrows():
        missed = n_events - int(r.tp)
        print(f"  {int(r.window):>4}  {r.threshold:>5.1f}  {int(r.timely):>4}/{n_events:<4}  {r.timely_rate:>6.1f}%  {int(r.too_early):>10}  {int(r.too_late):>9}  {missed:>7}  {int(r.fp):>5}  {r.fp_rate:>5.1f}%  {r.med_last_lead:>7.0f}d")

    # 参考: recall最大ルールのタイムリー評価
    ref_w = 3 if model == "EP370G" else 21
    ref_t = 0.5
    ref = grid[(grid["window"] == ref_w) & (grid["threshold"] == ref_t)].iloc[0]
    missed_ref = n_events - int(ref.tp)
    print(f"\n[参考] recall最大ルール ({ref_w}d>{ref_t}h): "
          f"timely={int(ref.timely)}/{n_events} ({ref.timely_rate:.1f}%)  "
          f"too_early={int(ref.too_early)}  too_late={int(ref.too_late)}  missed={missed_ref}  "
          f"fp={int(ref.fp)}  med_1st={ref.med_last_lead:.0f}d")
    print()
