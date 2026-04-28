"""tests/test_features_ts_dataset.py"""
from __future__ import annotations

import pandas as pd

from gene_plug_voltage_predictor.features.ts_dataset import build_ts_frame


def _make_daily_df(
    plug_id: str,
    dates: list[str],
    daily_max: list[float],
    gen_no: int | list[int] = 0,
    runtime: list[float] | None = None,
) -> pd.DataFrame:
    n = len(dates)
    gen_nos = [gen_no] * n if isinstance(gen_no, int) else gen_no
    runtime_vals = runtime if runtime is not None else [float(i * 100) for i in range(n)]
    return pd.DataFrame({
        "管理No_プラグNo": plug_id,
        "date": pd.to_datetime(dates),
        "daily_max": daily_max,
        "baseline": [22.0] * n,
        "gen_no": gen_nos,
        "稼働割合": [0.9] * n,
        "累積運転時間": runtime_vals,
        "voltage_vs_baseline": [v - 22.0 for v in daily_max],
    })


def test_calendar_expansion() -> None:
    """非運転日（月〜金のうち水曜欠け等）の行が補完されてカレンダー日が連続している。"""
    # 2024-01-01（月）, 2024-01-03（水）— 2024-01-02（火）が欠け
    daily_df = _make_daily_df(
        "5630_1",
        ["2024-01-01", "2024-01-03"],
        [220.0, 222.0],
    )
    result = build_ts_frame(daily_df)
    plug_df = result[result["管理No_プラグNo"] == "5630_1"].sort_values("date")
    dates = pd.to_datetime(plug_df["date"].tolist())
    # 2024-01-01 〜 2024-01-03 の3行が存在する
    assert len(plug_df) == 3
    assert dates[0] == pd.Timestamp("2024-01-01")
    assert dates[1] == pd.Timestamp("2024-01-02")
    assert dates[2] == pd.Timestamp("2024-01-03")


def test_forward_fill() -> None:
    """baseline/gen_no 等の特徴量は非運転日に forward-fill され、daily_max は NaN のまま。"""
    daily_df = _make_daily_df(
        "5630_1",
        ["2024-01-01", "2024-01-03"],
        [220.0, 225.0],
    )
    result = build_ts_frame(daily_df)
    plug_df = (
        result[result["管理No_プラグNo"] == "5630_1"].sort_values("date").reset_index(drop=True)
    )
    # 非運転日(2024-01-02)の daily_max は NaN（DataRobot TS がスキップするため）
    import math
    assert math.isnan(plug_df.loc[1, "daily_max"])
    # 特徴量（baseline）は ffill される
    assert plug_df.loc[1, "baseline"] == 22.0


def test_is_operating() -> None:
    """運転日は is_operating=1、補完された非運転日は is_operating=0。"""
    daily_df = _make_daily_df(
        "5630_1",
        ["2024-01-01", "2024-01-03"],
        [220.0, 222.0],
    )
    result = build_ts_frame(daily_df)
    plug_df = (
        result[result["管理No_プラグNo"] == "5630_1"].sort_values("date").reset_index(drop=True)
    )
    assert plug_df.loc[0, "is_operating"] == 1  # 2024-01-01（運転日）
    assert plug_df.loc[1, "is_operating"] == 0  # 2024-01-02（補完行）
    assert plug_df.loc[2, "is_operating"] == 1  # 2024-01-03（運転日）


def test_operating_hours_reset() -> None:
    """gen_no 変化時に operating_hours_since_exchange が 0 にリセットされる。"""
    # gen_no=0 の累積運転時間: 100, 200, 300
    # gen_no=1 の累積運転時間: 400, 500（交換後リセット）
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    gen_nos = [0, 0, 0, 1, 1]
    runtime = [100.0, 200.0, 300.0, 400.0, 500.0]
    daily_df = _make_daily_df("5630_1", dates, [220.0] * 5, gen_no=gen_nos, runtime=runtime)
    result = build_ts_frame(daily_df)
    plug_df = (
        result[result["管理No_プラグNo"] == "5630_1"].sort_values("date").reset_index(drop=True)
    )
    # gen_no=0 の初日は 0
    assert plug_df.loc[0, "operating_hours_since_exchange"] == 0.0
    # gen_no=1 の初日は 0（400 - 400 = 0）
    assert plug_df.loc[3, "operating_hours_since_exchange"] == 0.0
    # gen_no=1 の翌日は 100（500 - 400 = 100）
    assert plug_df.loc[4, "operating_hours_since_exchange"] == 100.0


def test_no_ts_auto_columns() -> None:
    """lag/rolling/days_since_exchange/future_7d_max が出力に含まれない。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01", "2024-01-02"], [220.0, 222.0])
    # TS自動生成列を手動で追加してから build_ts_frame に渡す
    daily_df["daily_max_lag_1"] = 219.0
    daily_df["daily_max_lag_3"] = 218.0
    daily_df["daily_max_lag_7"] = 217.0
    daily_df["voltage_trend_7d"] = 0.5
    daily_df["daily_max_rolling_mean_7d"] = 220.0
    daily_df["daily_max_rolling_std_7d"] = 1.0
    daily_df["days_since_exchange"] = 1
    daily_df["future_7d_max"] = 225.0
    result = build_ts_frame(daily_df)
    ts_auto_cols = [
        "daily_max_lag_1", "daily_max_lag_3", "daily_max_lag_7",
        "voltage_trend_7d", "daily_max_rolling_mean_7d", "daily_max_rolling_std_7d",
        "days_since_exchange", "future_7d_max",
    ]
    for col in ts_auto_cols:
        assert col not in result.columns, f"{col} should be excluded"


def test_kanri_no_column() -> None:
    """管理No 列が管理No_プラグNo の左側から正しく抽出されている。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    result = build_ts_frame(daily_df)
    assert "管理No" in result.columns
    assert result.iloc[0]["管理No"] == "5630"


def test_no_cross_plug_ffill() -> None:
    """複数プラグが混在しても、ffill がプラグ境界をまたがない。"""
    # plug A: gen_no=0 の最終日が 2024-01-03
    daily_a = _make_daily_df("5630_1", ["2024-01-01", "2024-01-03"], [220.0, 222.0], gen_no=0)
    # plug B: gen_no=1 の初日が 2024-01-04
    daily_b = _make_daily_df("5630_2", ["2024-01-04", "2024-01-05"], [230.0, 231.0], gen_no=1)
    daily_df = pd.concat([daily_a, daily_b], ignore_index=True)
    result = build_ts_frame(daily_df)
    # plug B の gen_no は 1 のまま（plug A の gen_no=0 で汚染されない）
    b_df = result[result["管理No_プラグNo"] == "5630_2"].sort_values("date").reset_index(drop=True)
    assert int(b_df.loc[0, "gen_no"]) == 1
    assert int(b_df.loc[1, "gen_no"]) == 1
