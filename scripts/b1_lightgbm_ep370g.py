"""
B1: LightGBM による timely アラート予測 (EP370G)

ラベル: 各 (machine_id, date) について、7-14 日後に baseline+16 越境する日なら 1
特徴量: 既存の daily 特徴 + rate11/12/13/14 + slope_w{3,5,7} 群 + days_in_gen
評価: アラート発火日の lead で timely/too_early/too_late を分類

時系列分割: cross_date でソートし、古い 2/3 を train、新しい 1/3 を eval（leakage 防止）。
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from itertools import product

import lightgbm as lgb

CLEANED = "outputs/cleaned_ep370g_v5.csv"
DAILY = "outputs/dataset_ep370g_ts_v7.csv"

LEVELS = [11, 12, 13, 14]
WINDOWS = [3, 5, 7]
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

    cleaned = pd.read_csv(
        CLEANED, encoding="utf-8-sig",
        usecols=[datetime_col, power_col, voltage_col, plug_id_col, "gen_no", "baseline"],
    )
    cleaned["_date"] = pd.to_datetime(cleaned[datetime_col]).dt.normalize()
    running = cleaned[cleaned[power_col] > 0].copy()

    for n in LEVELS:
        running[f"_at{n}"] = (running[voltage_col] == (running["baseline"].round() + n)).astype(float)
    running["_slot"] = 1.0
    agg = running.groupby([plug_id_col, "_date"]).agg(
        op_hours=("_slot", "sum"),
        **{f"_at{n}": (f"_at{n}", "sum") for n in LEVELS},
    ).mul(0.5).reset_index().rename(columns={"_date": "date"})
    agg["date"] = pd.to_datetime(agg["date"])
    for n in LEVELS:
        agg[f"rate{n}"] = np.where(
            agg["op_hours"] >= MIN_OP_HOURS, agg[f"_at{n}"] / agg["op_hours"], np.nan,
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
        .agg(
            daily_max=("daily_max", "max"),
            cross_thr=("cross_thr", "max"),
            baseline=("baseline", "max"),
            op_ratio=("稼働割合", "max"),
            cum_hours=("累積運転時間", "max"),
            voltage_vs_baseline=("voltage_vs_baseline", "max"),
            hours_at_31kv=("hours_at_31kv", "max"),
            op_hours_since_exchange=("operating_hours_since_exchange", "max"),
            **{f"rate{n}": (f"rate{n}", "max") for n in LEVELS},
        )
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


def build_labels(md, crossing_events, gen_periods, no_cross_gens):
    """
    各行のラベル:
      label = 1: その日から 7-14 日後に「初回越境」が発生（timely target）
      label = 0: その他（非越境 or その日が越境前 0-6/15+ 日）

    また、各 (machine_id, gen_no) を識別するメタ列を付与
    """
    md = md.copy()
    md["machine_id"] = md["machine_id"].astype(str)

    # gen 情報を結合
    gp = gen_periods.copy()
    gp["machine_id"] = gp["machine_id"].astype(str)
    md_g = md[["machine_id", "date"]].merge(gp, on="machine_id", how="left")
    md_g = md_g[(md_g["date"] >= md_g["gen_start"]) & (md_g["date"] <= md_g["gen_end"])]
    md_g = md_g[["machine_id", "date", "gen_no", "gen_start"]]
    md = md.merge(md_g, on=["machine_id", "date"], how="left")
    md["days_in_gen"] = (md["date"] - md["gen_start"]).dt.days

    # cross_date を結合
    ce = crossing_events.copy()
    ce["machine_id"] = ce["machine_id"].astype(str)
    md = md.merge(ce, on=["machine_id", "gen_no"], how="left")
    md["days_to_cross"] = (md["first_crossing_date"] - md["date"]).dt.days

    # ラベル: 越境前の 7-14 日窓
    md["label"] = ((md["days_to_cross"] >= T_MIN) & (md["days_to_cross"] <= T_MAX)).astype(int)

    # 訓練対象から除外: 越境済の日 (daily_max >= cross_thr) と gen 不明の日
    md["is_pre_crossing"] = (
        md["gen_no"].notna() & (
            (md["first_crossing_date"].isna()) |  # 非越境 gen
            ((md["daily_max"] < md["cross_thr"]) & (md["date"] < md["first_crossing_date"]))
        )
    )

    # 越境済の日は学習対象外、推論はもとから不要 (ルール評価と同じ)
    return md


print("Loading data...", flush=True)
md, crossing_events, gen_periods, no_cross_gens = load_machine_daily()
print(f"  events={len(crossing_events)} no_crossing={len(no_cross_gens)}", flush=True)

print("Building labels...", flush=True)
md = build_labels(md, crossing_events, gen_periods, no_cross_gens)

# 訓練対象: gen 内かつ (越境前 OR 非越境) の日
train_mask = md["is_pre_crossing"] & md["gen_no"].notna()
print(f"  rows={train_mask.sum()}, positive_label={md.loc[train_mask, 'label'].sum()}", flush=True)

# 特徴量列
FEATURE_COLS = [
    "baseline", "op_ratio", "cum_hours", "voltage_vs_baseline",
    "hours_at_31kv", "op_hours_since_exchange", "daily_max", "days_in_gen",
] + [f"rate{n}" for n in LEVELS] \
  + [f"rrate{n}_w{w}" for n in LEVELS for w in WINDOWS] \
  + [f"slope{n}_w{w}" for n in LEVELS for w in WINDOWS]

print(f"  n_features={len(FEATURE_COLS)}", flush=True)

# 時系列分割: cross_date でソートし、古い 70% を train、新しい 30% を eval
events_sorted = crossing_events.sort_values("first_crossing_date")
events_sorted["machine_id"] = events_sorted["machine_id"].astype(str)
n_train_ev = int(len(events_sorted) * 0.7)
train_event_keys = set(zip(events_sorted.iloc[:n_train_ev]["machine_id"],
                             events_sorted.iloc[:n_train_ev]["gen_no"]))
eval_event_keys = set(zip(events_sorted.iloc[n_train_ev:]["machine_id"],
                            events_sorted.iloc[n_train_ev:]["gen_no"]))
split_date = events_sorted.iloc[n_train_ev]["first_crossing_date"]
print(f"  split_date={split_date.date()} train_events={len(train_event_keys)} eval_events={len(eval_event_keys)}", flush=True)

# 非越境 gen は cross_date がないので gen_end で分割
nc = no_cross_gens.copy()
nc["machine_id"] = nc["machine_id"].astype(str)
nc_train = nc[nc["gen_end"] < split_date]
nc_eval = nc[nc["gen_end"] >= split_date]
train_nc_keys = set(zip(nc_train["machine_id"], nc_train["gen_no"]))
eval_nc_keys = set(zip(nc_eval["machine_id"], nc_eval["gen_no"]))
print(f"  train_nc={len(train_nc_keys)} eval_nc={len(eval_nc_keys)}", flush=True)

md["key"] = list(zip(md["machine_id"], md["gen_no"]))
md_train = md[train_mask & md["key"].isin(train_event_keys | train_nc_keys)].copy()
md_eval = md[train_mask & md["key"].isin(eval_event_keys | eval_nc_keys)].copy()
print(f"  train_rows={len(md_train)} eval_rows={len(md_eval)}", flush=True)

# 学習
X_train = md_train[FEATURE_COLS]
y_train = md_train["label"]
X_eval = md_eval[FEATURE_COLS]
y_eval = md_eval["label"]

print("\nTraining LightGBM...", flush=True)
pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
print(f"  pos_weight={pos_weight:.2f}", flush=True)

model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
    scale_pos_weight=pos_weight, random_state=42, verbose=-1,
)
model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)],
          callbacks=[lgb.early_stopping(30, verbose=False)])

# 全データに対する推論（評価はルールベースと同じ枠組みで全イベントについて）
md_full = md[train_mask].copy()
md_full["score"] = model.predict_proba(md_full[FEATURE_COLS])[:, 1]


def evaluate_threshold(md_scored, thr, crossing_events, gen_periods, no_cross_gens, event_keys=None, nc_keys=None):
    """確率閾値 thr で発火させて評価。event_keys/nc_keys が None なら全件、指定あればそれだけ。"""
    md_scored = md_scored.copy()
    md_scored["alert"] = (md_scored["score"] >= thr).astype(int)

    tp = fp = timely = too_early = too_late = 0
    for _, ev in crossing_events.iterrows():
        machine, gen_no, cross_date = str(ev["machine_id"]), ev["gen_no"], ev["first_crossing_date"]
        if event_keys is not None and (machine, gen_no) not in event_keys:
            continue
        sub = md_scored[
            (md_scored["machine_id"] == machine)
            & (md_scored["gen_no"] == gen_no)
            & (md_scored["date"] < cross_date)
            & (md_scored["daily_max"] < md_scored["cross_thr"])
        ]
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
    n_events = len(event_keys) if event_keys is not None else len(crossing_events)

    for _, nc in no_cross_gens.iterrows():
        machine, gen_no = str(nc["machine_id"]), nc["gen_no"]
        if nc_keys is not None and (machine, gen_no) not in nc_keys:
            continue
        sub = md_scored[(md_scored["machine_id"] == machine) & (md_scored["gen_no"] == gen_no)]
        if sub.empty:
            continue
        if (sub["alert"] > 0).any():
            fp += 1
    n_nc = len(nc_keys) if nc_keys is not None else len(no_cross_gens)

    return dict(
        timely=timely, too_early=too_early, too_late=too_late,
        missed=n_events - tp, fp=fp,
        timely_rate=timely / n_events * 100 if n_events else 0,
        fp_rate=fp / n_nc * 100 if n_nc else 0,
        n_events=n_events, n_nc=n_nc,
    )


# 閾値スイープ
print("\n=== Eval set (held-out events) ===")
print(f"{'thr':>5} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp':>4} {'fp_rate':>7}")
for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
    r = evaluate_threshold(md_full, thr, crossing_events, gen_periods, no_cross_gens,
                           event_keys=eval_event_keys, nc_keys=eval_nc_keys)
    print(f"{thr:>5.2f} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp']:>4} {r['fp_rate']:>6.1f}%   "
          f"(n_ev={r['n_events']} n_nc={r['n_nc']})")

print("\n=== All events (in-sample + eval together) ===")
print(f"{'thr':>5} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp':>4} {'fp_rate':>7}")
for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
    r = evaluate_threshold(md_full, thr, crossing_events, gen_periods, no_cross_gens)
    print(f"{thr:>5.2f} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp']:>4} {r['fp_rate']:>6.1f}%")

# 特徴量重要度
print("\n=== Top 15 feature importance ===")
fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(fi.head(15).to_string())

# 出力
md_full[["machine_id", "gen_no", "date", "score", "label", "daily_max", "cross_thr"]].to_csv(
    "outputs/b1_ep370g_scores.csv", index=False, encoding="utf-8-sig"
)
print("\nSaved: outputs/b1_ep370g_scores.csv")
