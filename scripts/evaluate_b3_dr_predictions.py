"""
B3-c DataRobot 予測の取得と timely 評価

1. プロジェクト 69fe7049e7c9c2dcdb4fa2ae の OOF 予測を取得
2. 各 (machine_id, gen_no) でアラート発火日を確率閾値ごとに求める
3. ルールベース (timely=27, too_early=11) と比較
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import datarobot as dr

from gene_plug_voltage_predictor.datarobot.client import setup_client
from gene_plug_voltage_predictor.datarobot.logging_setup import setup_logging
import logging
import time

PROJECT_ID = "69fe7bc2e7c9c2dcdb4fba40"
MODEL_ID = "69fe7cb2d37d5e03eaae1219"
PRED_OUT = Path("outputs/b3_dr_v2_oof_predictions.csv")
PRED_FULL = Path("outputs/b3_dr_v2_full_predictions.csv")
TRAIN_CSV = Path("outputs/dataset_ep370g_ts_binary_within14d_v1.csv")
DAILY_FULL = Path("outputs/dataset_ep370g_ts_v7.csv")  # 越境イベント抽出用 (越境後を含む)
T_MIN, T_MAX = 7, 14


def fetch_predictions():
    if PRED_OUT.exists():
        print(f"Predictions already exist at {PRED_OUT}, skipping download")
        return
    setup_logging()
    setup_client()
    project = dr.Project.get(PROJECT_ID)
    model = dr.Model.get(project.id, MODEL_ID)
    print(f"Fetching OOF predictions for model {model.id} (subset=ALL_BACKTESTS)")
    # TS モデルは ALL_BACKTESTS が必要
    existing = list(dr.TrainingPredictions.list(project_id=project.id))
    target = next((t for t in existing if t.model_id == model.id), None)
    if target is None:
        job = model.request_training_predictions(
            data_subset=dr.enums.DATA_SUBSET.ALL_BACKTESTS
        )
        job.wait_for_completion(max_wait=600)
        # 再度取得
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            existing = list(dr.TrainingPredictions.list(project_id=project.id))
            target = next((t for t in existing if t.model_id == model.id), None)
            if target is not None:
                break
            time.sleep(5)
    if target is None:
        print("ERROR: training predictions not found")
        sys.exit(1)
    pred_df = target.get_all_as_dataframe()
    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PRED_OUT, index=False, encoding="utf-8-sig")
    print(f"Saved: {PRED_OUT} (rows={len(pred_df)}, cols={list(pred_df.columns)})")


def fetch_full_predictions():
    """全訓練データに対する予測 (leakage あり、in-sample 性能評価用)。"""
    if PRED_FULL.exists():
        print(f"Full predictions already exist at {PRED_FULL}")
        return
    setup_logging()
    setup_client()
    project = dr.Project.get(PROJECT_ID)
    model = dr.Model.get(project.id, MODEL_ID)
    print(f"Uploading dataset for full prediction...")
    dataset = project.upload_dataset(str(TRAIN_CSV))
    print(f"Requesting predictions...")
    job = model.request_predictions(dataset.id)
    pred_df = job.get_result_when_complete(max_wait=3600)
    print(f"Got {len(pred_df)} predictions")
    pred_df.to_csv(PRED_FULL, index=False, encoding="utf-8-sig")
    print(f"Saved: {PRED_FULL}")


def evaluate():
    print(f"\n=== Using OOF predictions (validation backtests, ~35 days) ===")
    preds = pd.read_csv(PRED_OUT, encoding="utf-8-sig")
    print(f"  rows={len(preds)}, cols={list(preds.columns)}")

    print(f"\nLoading train CSV")
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig", parse_dates=["date"])
    print(f"  rows={len(train)}")

    # OOF の score 列は prediction だが回帰モデル風に出ている可能性、
    # binary では class_1.0 が陽性確率
    score_col = "prediction"
    print(f"  score_col={score_col}")

    # FD ごとの予測 → 各 (series_id, target_date=timestamp) で max を取る
    preds["timestamp"] = pd.to_datetime(preds["timestamp"]).dt.tz_localize(None)
    print(f"  unique series: {preds['series_id'].nunique()}, FD distinct: {sorted(preds['forecast_distance'].unique())}")

    # 各 (series_id, timestamp) で max スコア
    scored = (
        preds.groupby(["series_id", "timestamp"])[score_col]
        .max()
        .reset_index()
        .rename(columns={"timestamp": "date", score_col: "score"})
    )
    print(f"  unique (series, date) pairs: {len(scored)}")

    # 訓練データに紐付け
    train["date"] = pd.to_datetime(train["date"])
    train_sorted = train.merge(
        scored, left_on=["管理No_プラグNo", "date"], right_on=["series_id", "date"], how="left"
    )
    print(f"  matched rows: {train_sorted['score'].notna().sum()} / {len(train_sorted)}")
    if train_sorted["score"].notna().sum() == 0:
        print("ERROR: no scored rows after merge")
        sys.exit(1)

    print(f"  Score percentile: 50th={train_sorted['score'].quantile(0.5):.4f}, "
          f"95th={train_sorted['score'].quantile(0.95):.4f}, "
          f"99th={train_sorted['score'].quantile(0.99):.4f}, "
          f"max={train_sorted['score'].max():.4f}")

    # 越境イベントの再構成 (越境後の行を含む元 dataset を使う)
    train_sorted["machine_id"] = train_sorted["管理No_プラグNo"].str.rsplit("_", n=1).str[0]
    train_sorted["cross_thr"] = train_sorted["baseline"] + 16
    daily_full = pd.read_csv(DAILY_FULL, encoding="utf-8-sig", parse_dates=["date"])
    daily_full["machine_id"] = daily_full["管理No_プラグNo"].str.rsplit("_", n=1).str[0]
    daily_full["cross_thr"] = daily_full["baseline"] + 16
    op = daily_full[(daily_full["is_operating"] == 1)].copy()
    cross = (
        op[op["daily_max"] >= op["cross_thr"]]
        .groupby(["machine_id", "gen_no"])["date"].min()
        .reset_index().rename(columns={"date": "first_crossing_date"})
    )
    print(f"Crossing events: {len(cross)}")

    gen_periods = (
        op.groupby(["machine_id", "gen_no"])["date"]
        .agg(gen_start="min", gen_end="max").reset_index()
    )
    no_cross = gen_periods.merge(
        cross[["machine_id", "gen_no"]], on=["machine_id", "gen_no"], how="left", indicator=True
    )
    no_cross = no_cross[no_cross["_merge"] == "left_only"].drop(columns="_merge")
    print(f"Non-crossing gens: {len(no_cross)}")

    # OOF 範囲を取得
    oof_min = train_sorted.loc[train_sorted["score"].notna(), "date"].min()
    oof_max = train_sorted.loc[train_sorted["score"].notna(), "date"].max()
    print(f"\n  OOF date range: {oof_min.date()} .. {oof_max.date()}")

    # 厳密な subset: gen 全体が OOF 内に収まる事例のみ
    # (gen_start_dates が必要 -> gen_periods を再計算)
    daily_full2 = pd.read_csv(DAILY_FULL, encoding="utf-8-sig", parse_dates=["date"])
    daily_full2["machine_id"] = daily_full2["管理No_プラグNo"].str.rsplit("_", n=1).str[0]
    op_full = daily_full2[daily_full2["is_operating"] == 1]
    gen_periods = (
        op_full.groupby(["machine_id", "gen_no"])["date"]
        .agg(gen_start="min", gen_end="max").reset_index()
    )
    cross_with_start = cross.merge(gen_periods, on=["machine_id", "gen_no"], how="left")
    # gen_start が OOF 開始以降かつ first_crossing_date が OOF 終了 + 14d 以内
    cross_in_oof = cross_with_start[
        (cross_with_start["gen_start"] >= oof_min)
        & (cross_with_start["first_crossing_date"] <= oof_max + pd.Timedelta(days=14))
    ].copy()
    print(f"  Crossing events fully in OOF (gen_start >= oof_min): {len(cross_in_oof)}/{len(cross)}")

    nc_in_oof = no_cross[(no_cross["gen_start"] >= oof_min) & (no_cross["gen_end"] <= oof_max)].copy()
    print(f"  Non-crossing gens fully in OOF: {len(nc_in_oof)}/{len(no_cross)}")

    # 閾値スイープ
    print(f"\n=== Threshold sweep (events in OOF: {len(cross_in_oof)} events, {len(nc_in_oof)} non-crossing) ===")
    print(f"{'thr':>6} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp':>4} {'fp_rate':>7}")
    for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        timely = too_early = too_late = tp = fp = 0
        for _, ev in cross_in_oof.iterrows():
            machine, gen_no, cross_date = ev["machine_id"], ev["gen_no"], ev["first_crossing_date"]
            sub = train_sorted[
                (train_sorted["machine_id"] == machine)
                & (train_sorted["gen_no"] == gen_no)
                & (train_sorted["date"] < cross_date)
                & (train_sorted["daily_max"] < train_sorted["cross_thr"])
                & train_sorted["score"].notna()
            ]
            if sub.empty:
                continue
            fired = sub[sub["score"] >= thr]
            if fired.empty:
                continue
            tp += 1
            first_alert = fired["date"].min()
            lead = (cross_date - first_alert).days
            if T_MIN <= lead <= T_MAX:
                timely += 1
            elif lead < T_MIN:
                too_late += 1
            else:
                too_early += 1
        for _, nc in nc_in_oof.iterrows():
            machine, gen_no = nc["machine_id"], nc["gen_no"]
            sub = train_sorted[(train_sorted["machine_id"] == machine) & (train_sorted["gen_no"] == gen_no)
                               & train_sorted["score"].notna()]
            if not sub.empty and (sub["score"] >= thr).any():
                fp += 1
        n_events = len(cross_in_oof)
        n_nc = len(nc_in_oof)
        missed = n_events - tp
        fp_rate = fp / n_nc * 100 if n_nc else 0
        print(f"{thr:>6.2f} {timely:>6} {too_early:>9} {too_late:>8} "
              f"{missed:>6} {fp:>4} {fp_rate:>6.1f}%")

    # ルールベースを同じ subset で評価
    print(f"\n=== Rule baseline (slope11_w5>0.04 hist=10d ∧ slope14_w3>0.025) on same subset ===")
    eval_rule_baseline(cross_in_oof, nc_in_oof, oof_min, oof_max)


def eval_rule_baseline(cross_in_oof, nc_in_oof, oof_min, oof_max):
    """ルールベースを OOF subset で評価。ML との fair 比較のため。"""
    cleaned = pd.read_csv("outputs/cleaned_ep370g_v5.csv", encoding="utf-8-sig", nrows=0)
    cols = cleaned.columns.tolist()
    power_col, voltage_col, plug_id_col, datetime_col = cols[3], cols[7], cols[9], cols[1]
    df = pd.read_csv("outputs/cleaned_ep370g_v5.csv", encoding="utf-8-sig",
                     usecols=[datetime_col, power_col, voltage_col, plug_id_col, "gen_no", "baseline"])
    df["_date"] = pd.to_datetime(df[datetime_col]).dt.normalize()
    running = df[df[power_col] > 0].copy()
    for n in [11, 14]:
        running[f"_at{n}"] = (running[voltage_col] == (running["baseline"].round() + n)).astype(float)
    running["_slot"] = 1.0
    agg = running.groupby([plug_id_col, "_date"]).agg(
        op_hours=("_slot", "sum"),
        **{f"_at{n}": (f"_at{n}", "sum") for n in [11, 14]},
    ).mul(0.5).reset_index().rename(columns={"_date": "date"})
    agg["date"] = pd.to_datetime(agg["date"])
    for n in [11, 14]:
        agg[f"rate{n}"] = np.where(agg["op_hours"] >= 1.0, agg[f"_at{n}"] / agg["op_hours"], np.nan)

    daily = pd.read_csv(DAILY_FULL, encoding="utf-8-sig", parse_dates=["date"])
    daily["date"] = pd.to_datetime(daily["date"])
    merged = daily.merge(
        agg[[plug_id_col, "date"] + [f"rate{n}" for n in [11, 14]]],
        left_on=["管理No_プラグNo", "date"], right_on=[plug_id_col, "date"], how="left",
    )
    merged["cross_thr"] = merged["baseline"] + 16
    merged["machine_id"] = merged["管理No_プラグNo"].str.rsplit("_", n=1).str[0]
    op2 = merged[merged["is_operating"] == 1].copy()
    md = (
        op2.groupby(["machine_id", "date"])
        .agg(daily_max=("daily_max", "max"), cross_thr=("cross_thr", "max"),
             rate11=("rate11", "max"), rate14=("rate14", "max"))
        .reset_index().sort_values(["machine_id", "date"])
    )

    def _slope_local(values):
        valid = values[~np.isnan(values)]
        if len(valid) < 2:
            return np.nan
        x = np.arange(len(valid), dtype=float)
        return np.polyfit(x, valid, 1)[0]

    md["slope11_w5"] = md.groupby("machine_id")["rate11"].transform(
        lambda s: s.rolling(5, min_periods=2).apply(_slope_local, raw=True))
    md["slope14_w3"] = md.groupby("machine_id")["rate14"].transform(
        lambda s: s.rolling(3, min_periods=2).apply(_slope_local, raw=True))
    md["l1_fire"] = (md["slope11_w5"] > 0.040).astype(int)
    md["l1_hist"] = md.groupby("machine_id")["l1_fire"].transform(
        lambda x: x.rolling(10, min_periods=1).max())
    md["l2_fire"] = (md["slope14_w3"] > 0.025).astype(int)
    md["alert"] = ((md["l1_hist"] > 0) & (md["l2_fire"] > 0)).astype(int)

    timely = too_early = too_late = tp = fp = 0
    for _, ev in cross_in_oof.iterrows():
        machine, gen_no, cross_date = ev["machine_id"], ev["gen_no"], ev["first_crossing_date"]
        sub = md[(md["machine_id"] == machine) & (md["date"] < cross_date)
                 & (md["date"] >= oof_min) & (md["daily_max"] < md["cross_thr"])]
        if sub.empty or not (sub["alert"] > 0).any():
            continue
        tp += 1
        first_alert = sub[sub["alert"] > 0]["date"].min()
        lead = (cross_date - first_alert).days
        if T_MIN <= lead <= T_MAX:
            timely += 1
        elif lead < T_MIN:
            too_late += 1
        else:
            too_early += 1
    for _, nc in nc_in_oof.iterrows():
        machine = nc["machine_id"]
        gs, ge = nc["gen_start"], nc["gen_end"]
        sub = md[(md["machine_id"] == machine)
                 & (md["date"] >= max(gs, oof_min))
                 & (md["date"] <= min(ge, oof_max))]
        if not sub.empty and (sub["alert"] > 0).any():
            fp += 1
    n_events = len(cross_in_oof)
    n_nc = len(nc_in_oof)
    missed = n_events - tp
    fp_rate = fp / n_nc * 100 if n_nc else 0
    print(f"  Rule on OOF subset: timely={timely}/{n_events} too_early={too_early} "
          f"too_late={too_late} missed={missed} fp={fp} fp_rate={fp_rate:.1f}%")


if __name__ == "__main__":
    fetch_predictions()
    evaluate()
