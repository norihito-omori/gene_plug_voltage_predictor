"""
評価単位を (machine_id, gen_no) = 交換サイクルごとの越境イベントに変更したルールベース評価。

アラート条件: hours_at_31kV の 7 日ローリング合計 > 閾値 N
有効な警告 : アラート発火日が first_crossing_date より前、かつ
             アラート発火日の actual daily_max < 32kV（既に越境済みでないこと）
"""
from __future__ import annotations
import argparse
import pandas as pd

THRESHOLD_CROSS = 32.0
RULE_THRESHOLDS = [2, 3, 5, 7, 10, 14, 21]

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["EP370G", "EP400G", "both"], default="both")
args = parser.parse_args()

DATASETS = {
    "EP370G": ("outputs/dataset_ep370g_ts_v7.csv", 370.0),
    "EP400G": ("outputs/dataset_ep400g_ts_v2.csv", 400.0),
}
targets = ["EP370G", "EP400G"] if args.model == "both" else [args.model]

for model in targets:
    path, rated_kw = DATASETS[model]
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
    df["machine_id"] = df["管理No_プラグNo"].str.rsplit("_", n=1).str[0]
    op = df[df["is_operating"] == 1].copy()

    # -----------------------------------------------------------------------
    # 越境イベント: (machine_id, gen_no) ごとの初回 daily_max >= 32kV 日
    # -----------------------------------------------------------------------
    crossing_events = (
        op[op["daily_max"] >= THRESHOLD_CROSS]
        .groupby(["machine_id", "gen_no"])["date"].min()
        .reset_index().rename(columns={"date": "first_crossing_date"})
    )
    n_events = len(crossing_events)

    # gen_no ごとの daily_max (machine単位で最悪プラグを使用)
    machine_daily = (
        op.groupby(["machine_id", "date"])
        .agg(daily_max=("daily_max", "max"), hours_at_31kv=("hours_at_31kv", "max"))
        .reset_index().sort_values(["machine_id", "date"])
    )
    machine_daily["h7d"] = (
        machine_daily.groupby("machine_id")["hours_at_31kv"]
        .transform(lambda s: s.rolling(7, min_periods=1).sum())
    )

    # 越境のない (machine, gen_no) を FP カウント用に取得
    # 「その gen_no 期間中に越境なし」かつ「その gen_no の最終観測日まで」
    gen_periods = (
        op.groupby(["machine_id", "gen_no"])["date"]
        .agg(gen_start="min", gen_end="max")
        .reset_index()
    )
    no_crossing_gens = gen_periods.merge(
        crossing_events[["machine_id", "gen_no"]],
        on=["machine_id", "gen_no"], how="left", indicator=True
    )
    no_crossing_gens = no_crossing_gens[no_crossing_gens["_merge"] == "left_only"].drop(columns="_merge")

    print("=" * 68)
    print(f"{model} - 交換サイクル単位評価")
    print(f"越境イベント: {n_events} 件  |  非越境サイクル: {len(no_crossing_gens)} 件")
    print("=" * 68)
    print(f"{'thr':>5}  {'warned':>12}  {'recall':>7}  {'avg_lead':>9}  {'med_lead':>9}  {'fp':>5}")

    best = None
    for thr in RULE_THRESHOLDS:
        tp, fp, leads = 0, 0, []

        for _, ev in crossing_events.iterrows():
            machine, gen_no, cross_date = ev["machine_id"], ev["gen_no"], ev["first_crossing_date"]
            # gen_no の開始日を取得（前世代の越境後）
            gen_start_row = gen_periods[(gen_periods["machine_id"] == machine) & (gen_periods["gen_no"] == gen_no)]
            gen_start = gen_start_row["gen_start"].iloc[0] if not gen_start_row.empty else pd.Timestamp.min

            # この gen_no 期間内かつ越境前の日のアラート
            pre = machine_daily[
                (machine_daily["machine_id"] == machine)
                & (machine_daily["date"] >= gen_start)
                & (machine_daily["date"] < cross_date)
                & (machine_daily["daily_max"] < THRESHOLD_CROSS)  # アラート時点で未越境
            ]
            alerts = pre[pre["h7d"] > thr]
            if not alerts.empty:
                tp += 1
                leads.append((cross_date - alerts["date"].min()).days)

        # FP: 越境しなかった gen_no 期間中にアラートが出たサイクル数
        for _, nc in no_crossing_gens.iterrows():
            machine, gen_start, gen_end = nc["machine_id"], nc["gen_start"], nc["gen_end"]
            period = machine_daily[
                (machine_daily["machine_id"] == machine)
                & (machine_daily["date"] >= gen_start)
                & (machine_daily["date"] <= gen_end)
            ]
            if (period["h7d"] > thr).any():
                fp += 1

        recall = tp / n_events * 100
        avg = sum(leads) / len(leads) if leads else 0.0
        med = sorted(leads)[len(leads) // 2] if leads else 0.0
        print(f"{thr:>5}  {tp:>5}/{n_events:<5}  {recall:>6.1f}%  {avg:>9.1f}  {med:>9.1f}  {fp:>5}")
        if best is None or tp > best["tp"] or (tp == best["tp"] and avg > best["avg"]):
            best = {"thr": thr, "tp": tp, "recall": recall, "avg": avg, "med": med, "fp": fp}

    print(f"\n  Best: thr={best['thr']}h  warned={best['tp']}/{n_events}  recall={best['recall']:.1f}%"
          f"  avg_lead={best['avg']:.1f}d  median={best['med']:.1f}d  fp={best['fp']}")

    # 詳細: 未警告イベントの確認
    print(f"\n  未警告イベント (thr={best['thr']}h):")
    thr = best["thr"]
    missed = []
    for _, ev in crossing_events.iterrows():
        machine, gen_no, cross_date = ev["machine_id"], ev["gen_no"], ev["first_crossing_date"]
        gen_start_row = gen_periods[(gen_periods["machine_id"] == machine) & (gen_periods["gen_no"] == gen_no)]
        gen_start = gen_start_row["gen_start"].iloc[0] if not gen_start_row.empty else pd.Timestamp.min
        pre = machine_daily[
            (machine_daily["machine_id"] == machine)
            & (machine_daily["date"] >= gen_start)
            & (machine_daily["date"] < cross_date)
            & (machine_daily["daily_max"] < THRESHOLD_CROSS)
        ]
        alerts = pre[pre["h7d"] > thr]
        if alerts.empty:
            max_h7d = pre["h7d"].max() if not pre.empty else 0
            missed.append((machine, gen_no, cross_date.date(), f"{max_h7d:.1f}h"))
    if missed:
        print(f"  {'machine':>10}  {'gen_no':>7}  {'crossing':>12}  {'max_7d_sum':>12}")
        for m, g, c, mx in missed:
            print(f"  {m:>10}  {g:>7}  {str(c):>12}  {mx:>12}")
    else:
        print("  なし")
    print()
