# トレンド・統計特徴量追加 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `features/trend_features.py` に `add_trend_features()` を実装し、`build_dataset.py` のパイプラインに組み込むことで voltage_trend_7d / days_since_exchange / rolling mean / rolling std の4列を学習データセットに追加する。

**Architecture:** `add_features()` の出力（`gen_no` 列を含む日次 DataFrame）を受け取り、`daily_df` 単体から4列を計算して返す純粋関数 `add_trend_features()` を新規ファイルに実装する。`build_dataset.py` は `add_features()` と `add_future_7day_max_target()` の間で `add_trend_features()` を呼ぶ1行を追加するだけ。rolling 系はすべて plug 単位の groupby rolling で計算し、`.values` で index alignment を回避する（既存コードと同じパターン）。

**Tech Stack:** Python 3.13, pandas, numpy, pytest

---

## ファイル構成

| 操作 | パス | 役割 |
|------|------|------|
| 新規作成 | `src/gene_plug_voltage_predictor/features/trend_features.py` | `add_trend_features()` 実装 |
| 新規作成 | `tests/test_features_trend.py` | 8テスト |
| 更新 | `src/gene_plug_voltage_predictor/cli/build_dataset.py` | パイプラインに `add_trend_features()` を組み込む |

---

## 背景知識（ゼロコンテキスト向け）

### `add_features()` の出力スキーマ（`features.py` の出力）

`add_trend_features()` の入力となる `daily_df` は以下の列を持つ:

```
管理No_プラグNo, date, daily_max, baseline, gen_no, voltage_vs_baseline,
daily_max_lag_1, daily_max_lag_3, daily_max_lag_7, 稼働割合, 累積運転時間
```

### 追加する4列

| 列名 | 定義 |
|------|------|
| `voltage_trend_7d` | 直近7行（運転日ベース）の linear regression 傾き（V/日）。min_periods=2、先頭1行はNaN |
| `days_since_exchange` | 現世代（gen_no）の初日からの経過カレンダー日数（0-origin） |
| `daily_max_rolling_mean_7d` | 直近7行の移動平均。min_periods=1 |
| `daily_max_rolling_std_7d` | 直近7行の移動標準偏差。min_periods=2、先頭1行はNaN |

### 既存コードのパターン（従うこと）

`features.py` の lag 計算と同じパターンを使う:

```python
# groupby rolling → .values で index alignment をバイパス
result["some_col"] = (
    result.groupby(plug_id_col, sort=True)[daily_max_col]
    .rolling(window, min_periods=n)
    .mean()
    .reset_index(level=0, drop=True)
    .values
)
```

---

## Task 1: `add_trend_features()` の実装とテスト

**Files:**
- Create: `src/gene_plug_voltage_predictor/features/trend_features.py`
- Create: `tests/test_features_trend.py`

### Step 1-1: テストファイルを作成する

- [ ] `tests/test_features_trend.py` を以下の内容で作成する:

```python
"""tests/test_features_trend.py"""
from __future__ import annotations

import pandas as pd
import pytest

from gene_plug_voltage_predictor.features.trend_features import add_trend_features


def _make_daily_df(
    plug_id: str,
    dates: list[str],
    daily_max: list[float],
    gen_no: int | list[int] = 0,
) -> pd.DataFrame:
    n = len(dates)
    gen_nos = [gen_no] * n if isinstance(gen_no, int) else gen_no
    return pd.DataFrame({
        "管理No_プラグNo": plug_id,
        "date": pd.to_datetime(dates),
        "daily_max": daily_max,
        "gen_no": gen_nos,
    })


def test_add_trend_features_columns() -> None:
    """4列が全て追加されている。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    daily_df = _make_daily_df("5630_1", dates, [float(200 + i) for i in range(10)])
    result = add_trend_features(daily_df)
    expected_cols = {
        "voltage_trend_7d",
        "days_since_exchange",
        "daily_max_rolling_mean_7d",
        "daily_max_rolling_std_7d",
    }
    assert expected_cols.issubset(set(result.columns))


def test_add_trend_features_returns_copy() -> None:
    """入力 daily_df を変更しない（copy を返す）。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    original_cols = set(daily_df.columns)
    _ = add_trend_features(daily_df)
    assert set(daily_df.columns) == original_cols


def test_voltage_trend_7d_slope() -> None:
    """単調増加系列で傾き > 0、単調減少で傾き < 0。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 9)]

    # 単調増加: +2 V/日
    inc_values = [200.0 + i * 2.0 for i in range(8)]
    result_inc = add_trend_features(_make_daily_df("5630_1", dates, inc_values))
    slope_inc = result_inc.loc[
        result_inc["date"] == pd.Timestamp("2024-01-08"), "voltage_trend_7d"
    ].iloc[0]
    assert slope_inc > 0

    # 単調減少: -2 V/日
    dec_values = [200.0 - i * 2.0 for i in range(8)]
    result_dec = add_trend_features(_make_daily_df("5630_2", dates, dec_values))
    slope_dec = result_dec.loc[
        result_dec["date"] == pd.Timestamp("2024-01-08"), "voltage_trend_7d"
    ].iloc[0]
    assert slope_dec < 0


def test_voltage_trend_7d_first_row_nan() -> None:
    """先頭行（1点のみ）の voltage_trend_7d は NaN（min_periods=2）。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01", "2024-01-02"], [220.0, 222.0])
    result = add_trend_features(daily_df)
    assert pd.isna(
        result.loc[result["date"] == pd.Timestamp("2024-01-01"), "voltage_trend_7d"].iloc[0]
    )


def test_days_since_exchange_starts_at_zero() -> None:
    """各世代（gen_no）の初日が 0。"""
    dates = [
        "2024-01-01", "2024-01-02", "2024-01-03",
        "2024-01-04", "2024-01-05", "2024-01-06",
    ]
    gen_nos = [0, 0, 0, 1, 1, 1]
    daily_df = _make_daily_df("5630_1", dates, [220.0] * 6, gen_no=gen_nos)
    result = add_trend_features(daily_df)
    # gen_no=0 の初日
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-01"), "days_since_exchange"
    ].iloc[0] == 0
    # gen_no=1 の初日（交換直後）
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-04"), "days_since_exchange"
    ].iloc[0] == 0


def test_days_since_exchange_increments() -> None:
    """翌日が 1、その翌日が 2 と増加する。"""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    daily_df = _make_daily_df("5630_1", dates, [220.0, 221.0, 222.0])
    result = add_trend_features(daily_df)
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-01"), "days_since_exchange"
    ].iloc[0] == 0
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-02"), "days_since_exchange"
    ].iloc[0] == 1
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-03"), "days_since_exchange"
    ].iloc[0] == 2


def test_rolling_mean_7d() -> None:
    """既知値で rolling mean が正しく計算される。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 8)]
    values = [210.0, 212.0, 208.0, 215.0, 211.0, 213.0, 209.0]
    daily_df = _make_daily_df("5630_1", dates, values)
    result = add_trend_features(daily_df)
    # day 7 の rolling mean（7日分の平均）
    expected = sum(values) / 7
    actual = result.loc[
        result["date"] == pd.Timestamp("2024-01-07"), "daily_max_rolling_mean_7d"
    ].iloc[0]
    assert abs(actual - expected) < 1e-9
    # day 1 は min_periods=1 なので values[0] そのまま
    assert result.loc[
        result["date"] == pd.Timestamp("2024-01-01"), "daily_max_rolling_mean_7d"
    ].iloc[0] == 210.0


def test_rolling_std_7d_nan_for_single_row() -> None:
    """1行のみの場合 rolling_std は NaN（min_periods=2）。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    result = add_trend_features(daily_df)
    assert pd.isna(
        result.loc[
            result["date"] == pd.Timestamp("2024-01-01"), "daily_max_rolling_std_7d"
        ].iloc[0]
    )


def test_no_cross_plug_leakage() -> None:
    """plug A の rolling が plug B の結果に影響しない。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 8)]
    daily_a = _make_daily_df("A_1", dates, [999.0] * 7)
    daily_b = _make_daily_df("B_1", dates, [100.0] * 7)
    daily_df = pd.concat([daily_a, daily_b], ignore_index=True)
    result = add_trend_features(daily_df)
    b_mean = result.loc[
        (result["管理No_プラグNo"] == "B_1")
        & (result["date"] == pd.Timestamp("2024-01-07")),
        "daily_max_rolling_mean_7d",
    ].iloc[0]
    assert b_mean == 100.0
```

### Step 1-2: テストが失敗することを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_features_trend.py -v
```

期待出力: `ImportError` または `ModuleNotFoundError`

### Step 1-3: `trend_features.py` を実装する

- [ ] `src/gene_plug_voltage_predictor/features/trend_features.py` を以下の内容で作成する:

```python
"""トレンド・統計特徴量生成。voltage_trend_7d / days_since_exchange / rolling mean / rolling std。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _slope(values: np.ndarray) -> float:
    """values の線形回帰傾きを返す。2点未満の場合は NaN。

    rolling.apply(raw=True) から呼ばれ、numpy 配列を受け取る。
    """
    n = len(values)
    if n < 2:
        return float("nan")
    return float(np.polyfit(range(n), values, 1)[0])


def add_trend_features(
    daily_df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    date_col: str = "date",
    daily_max_col: str = "daily_max",
    gen_no_col: str = "gen_no",
    trend_window: int = 7,
) -> pd.DataFrame:
    """daily_df にトレンド・統計特徴量 4列を付与して返す（copy）。

    add_features() の出力（gen_no 列を含む）を入力とする前提。
    rolling は非運転日をスキップする行ベース（カレンダー日ではない）。
    .values で index alignment を回避する（lag 特徴量と同じパターン）。
    """
    result = daily_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([plug_id_col, date_col]).reset_index(drop=True)

    grouped = result.groupby(plug_id_col, sort=True)[daily_max_col]

    # voltage_trend_7d: 直近 trend_window 行の線形回帰傾き
    result["voltage_trend_7d"] = (
        grouped
        .rolling(trend_window, min_periods=2)
        .apply(_slope, raw=True)
        .reset_index(level=0, drop=True)
        .values
    )

    # daily_max_rolling_mean_7d: 直近 trend_window 行の移動平均
    result["daily_max_rolling_mean_7d"] = (
        grouped
        .rolling(trend_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .values
    )

    # daily_max_rolling_std_7d: 直近 trend_window 行の移動標準偏差
    result["daily_max_rolling_std_7d"] = (
        grouped
        .rolling(trend_window, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .values
    )

    # days_since_exchange: plug × gen_no の初日からの経過カレンダー日数
    origin = (
        result.groupby([plug_id_col, gen_no_col])[date_col]
        .min()
        .rename("_origin")
        .reset_index()
    )
    result = result.merge(origin, on=[plug_id_col, gen_no_col], how="left")
    result["days_since_exchange"] = (result[date_col] - result["_origin"]).dt.days
    result = result.drop(columns=["_origin"])

    return result
```

### Step 1-4: テストが通ることを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_features_trend.py -v
```

期待出力: `8 passed`

もし失敗する場合はエラーを調べて修正する。

### Step 1-5: ruff チェック

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m ruff check src/gene_plug_voltage_predictor/features/trend_features.py tests/test_features_trend.py
```

期待出力: エラーなし

### Step 1-6: コミット

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
git add src/gene_plug_voltage_predictor/features/trend_features.py tests/test_features_trend.py
git commit -m "feat: add trend/rolling features (voltage_trend_7d, days_since_exchange, rolling mean/std)"
```

---

## Task 2: `build_dataset.py` に `add_trend_features()` を組み込む

**Files:**
- Modify: `src/gene_plug_voltage_predictor/cli/build_dataset.py`

### Step 2-1: `build_dataset.py` を更新する

- [ ] `src/gene_plug_voltage_predictor/cli/build_dataset.py` を以下の内容に更新する:

```python
"""データセット組み立て CLI: cleaned CSV → DataRobot 学習用日次 CSV。

処理順:
  aggregate_daily_max_voltage → add_features → add_trend_features
  → add_future_7day_max_target → NaN 除外（future_{horizon}d_max / baseline）→ CSV 出力
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gene_plug_voltage_predictor.features.features import add_features
from gene_plug_voltage_predictor.features.target import (
    add_future_7day_max_target,
    aggregate_daily_max_voltage,
)
from gene_plug_voltage_predictor.features.trend_features import add_trend_features

_logger = logging.getLogger(__name__)


def build_dataset(
    cleaned_df: pd.DataFrame,
    *,
    rated_kw: float = 370.0,
    horizon: int = 7,
) -> pd.DataFrame:
    daily = aggregate_daily_max_voltage(cleaned_df)
    daily = add_features(daily, cleaned_df, rated_kw=rated_kw)
    daily = add_trend_features(daily)
    daily = add_future_7day_max_target(daily, horizon=horizon)

    target_col = f"future_{horizon}d_max"
    before = len(daily)
    daily = daily.dropna(subset=[target_col, "baseline"]).reset_index(drop=True)
    dropped = before - len(daily)
    if dropped:
        _logger.info("dropped %d rows with NaN target or baseline", dropped)
    if len(daily) == 0:
        _logger.warning("dataset is empty after NaN removal")
    return daily


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build DataRobot training dataset from cleaned CSV"
    )
    ap.add_argument(
        "--cleaned-csv", required=True, type=Path,
        help="clean.py の出力 CSV（utf-8-sig）",
    )
    ap.add_argument(
        "--out", required=True, type=Path,
        help="DataRobot 投入用 CSV の出力先",
    )
    ap.add_argument(
        "--rated-kw", type=float, default=370.0,
        help="定格出力 kW（稼働割合の閾値 = rated_kw × 0.8, default: 370.0）",
    )
    ap.add_argument(
        "--horizon", type=int, default=7,
        help="予測ホライズン日数（default: 7）",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cleaned_df = pd.read_csv(args.cleaned_csv, encoding="utf-8-sig")
    dataset = build_dataset(cleaned_df, rated_kw=args.rated_kw, horizon=args.horizon)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.out} ({len(dataset)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 2-2: 既存の `test_cli_build_dataset.py` が通ることを確認する

`test_build_dataset_produces_expected_columns` のテストは `issubset` で必要列の存在を確認するため、新規4列が追加されても既存テストは通る。

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_cli_build_dataset.py -v
```

期待出力: `4 passed`

### Step 2-3: 全テストスイートが通ることを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/ -q --tb=short
```

期待出力: `122 passed`（既存 114 + 新規 8）

### Step 2-4: ruff チェック

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m ruff check src/gene_plug_voltage_predictor/cli/build_dataset.py
```

期待出力: エラーなし

### Step 2-5: コミット

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
git add src/gene_plug_voltage_predictor/cli/build_dataset.py
git commit -m "feat: wire add_trend_features into build_dataset pipeline"
```

---

## 注意事項

### rolling の行ベース vs カレンダー日ベース

`groupby(...).rolling(7)` は行数ベース。非運転日（`daily_df` に行なし）はスキップされる。
これは `add_future_7day_max_target()` がカレンダー日ベースとは異なる設計だが、
特徴量として「直近7運転日」の傾きを使う方が物理的に自然なため、この仕様とする。

### `.values` によるインデックス整合性

`groupby(...).rolling(...).mean()` の結果は MultiIndex を持つ。
`.reset_index(level=0, drop=True).values` で元の DataFrame の行順に整合したフラットな配列として代入する。これは既存の lag 特徴量計算 (`features.py:71-73`) と同じパターン。

### `days_since_exchange` と非連続日付

`(date - origin).dt.days` はカレンダー日数。非運転日を飛ばして計測する（例: 月曜・火曜・水曜と月曜・木曜の場合、後者の木曜は `days=3`）。
