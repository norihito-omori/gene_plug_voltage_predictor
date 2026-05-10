"""
B1 v2: LightGBM 改善版 — 過学習対策

v1 の問題:
  - days_in_gen / cum_hours が重要度上位 → サイクル後半の経過時間に依存し holdout で破綻
  - holdout timely=3/14 (21%)、in-sample timely=26/30 (87%)

v2 の改善:
  - 時間関連特徴量を除外: days_in_gen, cum_hours, op_hours_since_exchange を削除
  - 残すのは「電圧シグナル中心」の特徴量のみ
  - 正則化強化: max_depth=4, num_leaves=15, min_child_samples=50
  - early_stopping を強める
"""
from __future__ import annotations
import pandas as pd
import numpy as np

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
            voltage_vs_baseline=("voltage_vs_baseline", "max"),
            op_ratio=("稼働割合", "max"),
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
    md = md.copy()
    md["machine_id"] = md["machine_id"].astype(str)
    gp = gen_periods.copy()
    gp["machine_id"] = gp["machine_id"].astype(str)
    md_g = md[["machine_id", "date"]].merge(gp, on="machine_id", how="left")
    md_g = md_g[(md_g["date"] >= md_g["gen_start"]) & (md_g["date"] <= md_g["gen_end"])]
    md_g = md_g[["machine_id", "date", "gen_no", "gen_start"]]
    md = md.merge(md_g, on=["machine_id", "date"], how="left")
    ce = crossing_events.copy()
    ce["machine_id"] = ce["machine_id"].astype(str)
    md = md.merge(ce, on=["machine_id", "gen_no"], how="left")
    md["days_to_cross"] = (md["first_crossing_date"] - md["date"]).dt.days
    md["label"] = ((md["days_to_cross"] >= T_MIN) & (md["days_to_cross"] <= T_MAX)).astype(int)
    md["is_pre_crossing"] = (
        md["gen_no"].notna() & (
            md["first_crossing_date"].isna() |
            ((md["daily_max"] < md["cross_thr"]) & (md["date"] < md["first_crossing_date"]))
        )
    )
    return md


print("Loading data...", flush=True)
md, crossing_events, gen_periods, no_cross_gens = load_machine_daily()
print(f"  events={len(crossing_events)} no_crossing={len(no_cross_gens)}", flush=True)

print("Building labels...", flush=True)
md = build_labels(md, crossing_events, gen_periods, no_cross_gens)

train_mask = md["is_pre_crossing"] & md["gen_no"].notna()

# v2: 時間関連特徴を除外
FEATURE_COLS = [
    "baseline", "voltage_vs_baseline", "op_ratio", "daily_max",
] + [f"rate{n}" for n in LEVELS] \
  + [f"rrate{n}_w{w}" for n in LEVELS for w in WINDOWS] \
  + [f"slope{n}_w{w}" for n in LEVELS for w in WINDOWS]

print(f"  n_features={len(FEATURE_COLS)}", flush=True)

events_sorted = crossing_events.sort_values("first_crossing_date")
events_sorted["machine_id"] = events_sorted["machine_id"].astype(str)
n_train_ev = int(len(events_sorted) * 0.7)
train_event_keys = set(zip(events_sorted.iloc[:n_train_ev]["machine_id"],
                             events_sorted.iloc[:n_train_ev]["gen_no"]))
eval_event_keys = set(zip(events_sorted.iloc[n_train_ev:]["machine_id"],
                            events_sorted.iloc[n_train_ev:]["gen_no"]))
split_date = events_sorted.iloc[n_train_ev]["first_crossing_date"]
print(f"  split_date={split_date.date()} train_events={len(train_event_keys)} eval_events={len(eval_event_keys)}", flush=True)

nc = no_cross_gens.copy()
nc["machine_id"] = nc["machine_id"].astype(str)
train_nc_keys = set(zip(nc[nc["gen_end"] < split_date]["machine_id"], nc[nc["gen_end"] < split_date]["gen_no"]))
eval_nc_keys = set(zip(nc[nc["gen_end"] >= split_date]["machine_id"], nc[nc["gen_end"] >= split_date]["gen_no"]))

md["key"] = list(zip(md["machine_id"], md["gen_no"]))
md_train = md[train_mask & md["key"].isin(train_event_keys | train_nc_keys)].copy()
md_eval = md[train_mask & md["key"].isin(eval_event_keys | eval_nc_keys)].copy()
print(f"  train_rows={len(md_train)} eval_rows={len(md_eval)}", flush=True)

X_train = md_train[FEATURE_COLS]
y_train = md_train["label"]
X_eval = md_eval[FEATURE_COLS]
y_eval = md_eval["label"]

print("\nTraining LightGBM (regularized)...", flush=True)
pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.03,
    max_depth=4, num_leaves=15,
    min_child_samples=50, reg_alpha=0.5, reg_lambda=0.5,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    scale_pos_weight=pos_weight, random_state=42, verbose=-1,
)
model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)],
          callbacks=[lgb.early_stopping(20, verbose=False)])

md_full = md[train_mask].copy()
md_full["score"] = model.predict_proba(md_full[FEATURE_COLS])[:, 1]


def evaluate_threshold(md_scored, thr, crossing_events, no_cross_gens, event_keys=None, nc_keys=None):
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
    for _, n_row in no_cross_gens.iterrows():
        machine, gen_no = str(n_row["machine_id"]), n_row["gen_no"]
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


print("\n=== Eval set (held-out 14 events) ===")
print(f"{'thr':>5} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp':>4} {'fp_rate':>7}")
for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    r = evaluate_threshold(md_full, thr, crossing_events, no_cross_gens,
                           event_keys=eval_event_keys, nc_keys=eval_nc_keys)
    print(f"{thr:>5.2f} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp']:>4} {r['fp_rate']:>6.1f}%")

print("\n=== Train set (in-sample, 30 events) ===")
print(f"{'thr':>5} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp':>4} {'fp_rate':>7}")
for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    r = evaluate_threshold(md_full, thr, crossing_events, no_cross_gens,
                           event_keys=train_event_keys, nc_keys=train_nc_keys)
    print(f"{thr:>5.2f} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp']:>4} {r['fp_rate']:>6.1f}%")

print("\n=== All events (train+eval, 44 events) ===")
print(f"{'thr':>5} {'timely':>6} {'too_early':>9} {'too_late':>8} {'missed':>6} {'fp':>4} {'fp_rate':>7}")
for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    r = evaluate_threshold(md_full, thr, crossing_events, no_cross_gens)
    print(f"{thr:>5.2f} {r['timely']:>6} {r['too_early']:>9} {r['too_late']:>8} "
          f"{r['missed']:>6} {r['fp']:>4} {r['fp_rate']:>6.1f}%")

print("\n=== Top 15 feature importance ===")
fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(fi.head(15).to_string())
