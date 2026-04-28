"""DataRobot TS 用データセット変換。カレンダー展開・forward-fill・TS自動生成列除外。"""
from __future__ import annotations

import pandas as pd

TS_AUTO_COLS: list[str] = [
    "daily_max_lag_1",
    "daily_max_lag_3",
    "daily_max_lag_7",
    "voltage_trend_7d",
    "daily_max_rolling_mean_7d",
    "daily_max_rolling_std_7d",
    "days_since_exchange",
    "future_7d_max",
]

_FFILL_COLS = ["daily_max", "baseline", "gen_no", "累積運転時間", "voltage_vs_baseline"]


def build_ts_frame(
    daily_df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    date_col: str = "date",
    daily_max_col: str = "daily_max",
    gen_no_col: str = "gen_no",
    runtime_col: str = "累積運転時間",
    operating_ratio_col: str = "稼働割合",
) -> pd.DataFrame:
    """add_features() 出力をカレンダー日ベースに展開して DataRobot TS 用に整形する。

    非運転日を forward-fill で補完し、DataRobot TS が自動生成する lag/rolling 列を除外する。
    .copy() で入力を変更しない。

    非運転日の 累積運転時間 は ffill されるため、operating_hours_since_exchange は
    非運転日でも前日と同じ値（flat）になる。これは実運転時間ベースの劣化指標として
    「停止中は劣化が進行しない」ことを反映した意図的な設計。
    """
    result = daily_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([plug_id_col, date_col]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for plug_id, group in result.groupby(plug_id_col, sort=True):
        group = group.set_index(date_col)
        full_idx = pd.date_range(group.index.min(), group.index.max(), freq="D")

        # is_operating: 元の運転日=1、補完行=0
        is_op = pd.Series(1, index=group.index, name="is_operating")

        # カレンダー展開
        expanded = group.reindex(full_idx)
        expanded["is_operating"] = 0
        expanded.loc[is_op.index, "is_operating"] = 1

        # forward-fill
        for col in _FFILL_COLS:
            if col in expanded.columns:
                expanded[col] = expanded[col].ffill()

        # 稼働割合: 非運転日は 0
        if operating_ratio_col in expanded.columns:
            expanded[operating_ratio_col] = expanded[operating_ratio_col].fillna(0.0)

        # plug_id_col を復元
        expanded[plug_id_col] = plug_id

        # gen_no を Int64 に変換（ffill 後は float になることがある）
        if gen_no_col in expanded.columns:
            expanded[gen_no_col] = expanded[gen_no_col].astype("Int64")

        expanded.index.name = date_col
        expanded = expanded.reset_index()

        frames.append(expanded)

    result = pd.concat(frames, ignore_index=True)

    # operating_hours_since_exchange: plug × gen_no の累積運転時間起点リセット
    origin = result.groupby([plug_id_col, gen_no_col])[runtime_col].transform("min")
    result["operating_hours_since_exchange"] = result[runtime_col] - origin

    # 管理No: plug_id_col を "_" で分割して左側を取得
    if result[plug_id_col].str.contains("_").all():
        result["管理No"] = result[plug_id_col].str.split("_").str[0]
    else:
        raise ValueError(
            f"plug_id_col '{plug_id_col}' does not contain '_': "
            "cannot extract 管理No"
        )

    # TS自動生成列を除外（存在する場合のみ）
    drop_cols = [c for c in TS_AUTO_COLS if c in result.columns]
    if drop_cols:
        result = result.drop(columns=drop_cols)

    return result
