"""
ルールベースアラート最適化: ウィンドウ幅 × 閾値の2次元グリッドサーチ
アラート条件: hours_at_31kv の N日ローリング合計 > threshold

出力:
  - コンソール: Pareto 最適解（recall 最大 / FP 最小 / リード日数）
  - outputs/alert_rule_grid_EP370G.csv, outputs/alert_rule_grid_EP400G.csv
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

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["EP370G", "EP400G", "both"], default="both")
args = parser.parse_args()
targets = ["EP370G", "EP400G"] if args.model == "both" else [args.model]

for model in targets:
    path = DATASETS[model]
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
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
        .agg(gen_start="min", gen_end="max")
        .reset_index()
    )
    no_crossing_gens = gen_periods.merge(
        crossing_events[["machine_id", "gen_no"]],
        on=["machine_id", "gen_no"], how="left", indicator=True
    )
    no_crossing_gens = no_crossing_gens[no_crossing_gens["_merge"] == "left_only"].drop(columns="_merge")
    n_nc = len(no_crossing_gens)

    # 全ウィンドウ幅分のローリング合計を事前計算
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
            tp, fp, leads = 0, 0, []

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
                    leads.append((cross_date - alerts["date"].min()).days)

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
            fp_rate = fp / n_nc * 100 if n_nc > 0 else 0.0
            avg_lead = sum(leads) / len(leads) if leads else 0.0
            med_lead = sorted(leads)[len(leads) // 2] if leads else 0.0
            results.append({
                "window": w, "threshold": thr,
                "tp": tp, "n_events": n_events, "recall": recall,
                "fp": fp, "n_nc": n_nc, "fp_rate": fp_rate,
                "avg_lead": avg_lead, "med_lead": med_lead,
            })

    grid = pd.DataFrame(results)
    out_csv = f"outputs/alert_rule_grid_{model}.csv"
    grid.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print(f"{model}  (越境イベント: {n_events}件  非越境サイクル: {n_nc}件)")
    print("=" * 72)

    # Pareto 最適解: recall が最大のグループ → その中でFP最小 → リード日数最長
    max_recall = grid["recall"].max()
    top = grid[grid["recall"] == max_recall].sort_values(["fp", "avg_lead"], ascending=[True, False])
    print(f"\n[A] Recall 最大 ({max_recall:.1f}%)")
    print(f"  {'win':>4}  {'thr':>5}  {'warned':>10}  {'recall':>7}  {'avg_lead':>9}  {'med_lead':>9}  {'fp':>5}  {'fp_rate':>8}")
    for _, r in top.head(5).iterrows():
        print(f"  {int(r.window):>4}  {r.threshold:>5.1f}  {int(r.tp):>5}/{n_events:<4}  {r.recall:>6.1f}%  {r.avg_lead:>9.1f}  {r.med_lead:>9.1f}  {int(r.fp):>5}  {r.fp_rate:>7.1f}%")

    # FP最小 (recall >= 70%) → その中でrecall最大
    fp_min = grid[grid["recall"] >= 70.0]["fp"].min()
    low_fp = grid[(grid["recall"] >= 70.0) & (grid["fp"] == fp_min)].sort_values("recall", ascending=False)
    print(f"\n[B] FP 最小 (recall >= 70%, FP={fp_min})")
    print(f"  {'win':>4}  {'thr':>5}  {'warned':>10}  {'recall':>7}  {'avg_lead':>9}  {'med_lead':>9}  {'fp':>5}  {'fp_rate':>8}")
    for _, r in low_fp.head(5).iterrows():
        print(f"  {int(r.window):>4}  {r.threshold:>5.1f}  {int(r.tp):>5}/{n_events:<4}  {r.recall:>6.1f}%  {r.avg_lead:>9.1f}  {r.med_lead:>9.1f}  {int(r.fp):>5}  {r.fp_rate:>7.1f}%")

    # リード日数中央値最大 (recall >= 70%) → バランス型
    best_lead = grid[grid["recall"] >= 70.0].sort_values(["med_lead", "recall", "fp"], ascending=[False, False, True])
    print(f"\n[C] リード日数中央値 最大 (recall >= 70%)")
    print(f"  {'win':>4}  {'thr':>5}  {'warned':>10}  {'recall':>7}  {'avg_lead':>9}  {'med_lead':>9}  {'fp':>5}  {'fp_rate':>8}")
    for _, r in best_lead.head(5).iterrows():
        print(f"  {int(r.window):>4}  {r.threshold:>5.1f}  {int(r.tp):>5}/{n_events:<4}  {r.recall:>6.1f}%  {r.avg_lead:>9.1f}  {r.med_lead:>9.1f}  {int(r.fp):>5}  {r.fp_rate:>7.1f}%")

    # 全結果サマリ (recall >= 70% のみ)
    print(f"\n[全グリッド recap: recall >= 70%]")
    print(f"  {'win':>4}  {'thr':>5}  {'recall':>7}  {'avg_lead':>9}  {'med_lead':>9}  {'fp':>5}  {'fp_rate':>8}")
    show = grid[grid["recall"] >= 70.0].sort_values(["recall", "fp", "avg_lead"], ascending=[False, True, False])
    for _, r in show.iterrows():
        print(f"  {int(r.window):>4}  {r.threshold:>5.1f}  {r.recall:>6.1f}%  {r.avg_lead:>9.1f}  {r.med_lead:>9.1f}  {int(r.fp):>5}  {r.fp_rate:>7.1f}%")

    print(f"\nGrid CSV saved: {out_csv}\n")
