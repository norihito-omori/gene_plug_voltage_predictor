"""tests/test_cli_build_dataset.py"""
from __future__ import annotations

import pandas as pd

from gene_plug_voltage_predictor.cli.build_dataset import build_dataset


def _make_cleaned_df() -> pd.DataFrame:
    """2 plug × 20 日分の最小 cleaned_df を生成する。"""
    rows = []
    for plug in ["5630_1", "5630_2"]:
        for day in range(1, 21):
            dt = f"2024-01-{day:02d} 00:30"
            rows.append({
                "管理No_プラグNo": plug,
                "dailygraphpt_ptdatetime": dt,
                "発電機電力": 300.0,
                "累積運転時間": float(100 + day),
                "baseline": 200.0,
                "gen_no": 0,
                "要求電圧": float(210 + day),
            })
    df = pd.DataFrame(rows)
    df["dailygraphpt_ptdatetime"] = pd.to_datetime(df["dailygraphpt_ptdatetime"])
    return df


def test_build_dataset_produces_expected_columns() -> None:
    """出力 DataFrame に必要な列が全て含まれる。"""
    cleaned_df = _make_cleaned_df()
    result = build_dataset(cleaned_df, rated_kw=370.0, horizon=7)
    expected_cols = {
        "管理No_プラグNo",
        "date",
        "daily_max",
        "baseline",
        "gen_no",
        "voltage_vs_baseline",
        "daily_max_lag_1",
        "daily_max_lag_3",
        "daily_max_lag_7",
        "稼働割合",
        "累積運転時間",
        "future_7d_max",
    }
    assert expected_cols.issubset(set(result.columns)), (
        f"Missing columns: {expected_cols - set(result.columns)}"
    )


def test_build_dataset_drops_nan_target_rows() -> None:
    """future_7d_max = NaN の行（末尾 horizon 行）が除外される。"""
    cleaned_df = _make_cleaned_df()
    result = build_dataset(cleaned_df, rated_kw=370.0, horizon=7)
    assert result["future_7d_max"].notna().all(), "future_7d_max must have no NaN"


def test_build_dataset_drops_nan_baseline_rows() -> None:
    """baseline = NaN の行が除外される。"""
    cleaned_df = _make_cleaned_df()
    # 1 plug の baseline を全行 NaN に書き換える
    cleaned_df.loc[cleaned_df["管理No_プラグNo"] == "5630_1", "baseline"] = float("nan")
    result = build_dataset(cleaned_df, rated_kw=370.0, horizon=7)
    assert result["baseline"].notna().all(), "baseline must have no NaN"
