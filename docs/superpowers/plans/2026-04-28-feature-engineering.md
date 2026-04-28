# Feature Engineering (説明変数生成) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `features/features.py` に `add_features()` を実装し、日次 DataFrame に説明変数（baseline、gen_no、voltage_vs_baseline、lag×3、稼働割合、累積運転時間）を付与する。

**Architecture:** `add_features(daily_df, cleaned_df)` の 1 関数に全説明変数の計算をまとめる。`cleaned_df`（30 分粒度）から baseline/gen_no/稼働割合/累積運転時間を日次集約して `daily_df` に left join し、その後 lag 特徴量を plug 単位 groupby + shift で付与する。

**Tech Stack:** Python 3.13, pandas, pytest

---

## ファイル構成

| 操作 | パス | 役割 |
|------|------|------|
| 新規作成 | `src/gene_plug_voltage_predictor/features/features.py` | `add_features()` の実装 |
| 新規作成 | `tests/test_features_features.py` | 全テスト（7 ケース） |
| 変更なし | `src/gene_plug_voltage_predictor/features/target.py` | 既存、触らない |
| 変更なし | `src/gene_plug_voltage_predictor/features/__init__.py` | 既存（空）、触らない |

---

## 背景知識（ゼロコンテキスト向け）

### クリーニング後の DataFrame スキーマ（`cleaned_df`）

30 分粒度・縦持ち（melt 後）で以下の列を持つ:

```
管理No_プラグNo, dailygraphpt_ptdatetime, target_id, 発電機電力, 累積運転時間,
target_output, mcnkind_id, 要求電圧, プラグNo, 管理No, gen_no, baseline
```

- `管理No_プラグNo`: プラグ複合 ID（例: `"5630_1"`）
- `dailygraphpt_ptdatetime`: 30 分粒度の計測日時
- `発電機電力`: kW（0 以下は停止中）
- `累積運転時間`: 時間（単調増加、プラグ交換時にリセット）
- `baseline`: 世代開始 30 日間の日次最大電圧の中央値（全行に broadcast 済み）
- `gen_no`: 世代番号（0-origin、プラグ交換で +1）

### 日次テーブルスキーマ（`daily_df`）

`aggregate_daily_max_voltage(cleaned_df)` の出力:

```
管理No_プラグNo, date, daily_max
```

### `add_features` が付与する列

| 列名 | 定義 |
|------|------|
| `baseline` | cleaned_df の plug × 日付の `baseline` 最初の値（broadcast されているので first で OK） |
| `gen_no` | cleaned_df の plug × 日付の `gen_no` 最初の値（同上） |
| `voltage_vs_baseline` | `daily_max - baseline` |
| `daily_max_lag_1` | plug 単位で 1 日前の `daily_max` |
| `daily_max_lag_3` | plug 単位で 3 日前の `daily_max` |
| `daily_max_lag_7` | plug 単位で 7 日前の `daily_max` |
| `稼働割合` | その日に `発電機電力 >= rated_kw * 0.8` だった行の割合（plug × 日付） |
| `累積運転時間` | plug × 日付の `累積運転時間` 最大値 |

---

## Task 1: `add_features` の基本実装（baseline/gen_no/voltage_vs_baseline）

**Files:**
- Create: `src/gene_plug_voltage_predictor/features/features.py`
- Create: `tests/test_features_features.py`

### Step 1-1: 失敗するテストを書く

- [ ] `tests/test_features_features.py` を以下の内容で作成する:

```python
"""tests/test_features_features.py"""
from __future__ import annotations

import pandas as pd
import numpy as np

from gene_plug_voltage_predictor.features.features import add_features


def _make_daily_df(
    plug_id: str,
    dates: list[str],
    daily_max: list[float],
) -> pd.DataFrame:
    return pd.DataFrame({
        "管理No_プラグNo": plug_id,
        "date": pd.to_datetime(dates),
        "daily_max": daily_max,
    })


def _make_cleaned_df(
    plug_id: str,
    datetimes: list[str],
    power: list[float],
    runtime: list[float],
    baseline: float,
    gen_no: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame({
        "管理No_プラグNo": plug_id,
        "dailygraphpt_ptdatetime": pd.to_datetime(datetimes),
        "発電機電力": power,
        "累積運転時間": runtime,
        "baseline": baseline,
        "gen_no": gen_no,
    })


def test_add_features_voltage_vs_baseline() -> None:
    """voltage_vs_baseline = daily_max - baseline が正しく計算される。"""
    daily_df = _make_daily_df(
        "5630_1",
        ["2024-01-01", "2024-01-02"],
        [220.0, 225.0],
    )
    cleaned_df = _make_cleaned_df(
        "5630_1",
        ["2024-01-01 00:30", "2024-01-01 01:00",
         "2024-01-02 00:30", "2024-01-02 01:00"],
        [300.0, 300.0, 300.0, 300.0],
        [100.0, 101.0, 102.0, 103.0],
        baseline=200.0,
    )
    result = add_features(daily_df, cleaned_df)
    assert "voltage_vs_baseline" in result.columns
    assert result.loc[result["date"] == pd.Timestamp("2024-01-01"), "voltage_vs_baseline"].iloc[0] == 20.0
    assert result.loc[result["date"] == pd.Timestamp("2024-01-02"), "voltage_vs_baseline"].iloc[0] == 25.0


def test_add_features_baseline_nan_propagates() -> None:
    """baseline=NaN のとき voltage_vs_baseline=NaN。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    cleaned_df = _make_cleaned_df(
        "5630_1",
        ["2024-01-01 00:30"],
        [300.0],
        [100.0],
        baseline=float("nan"),
    )
    result = add_features(daily_df, cleaned_df)
    assert pd.isna(
        result.loc[result["date"] == pd.Timestamp("2024-01-01"), "voltage_vs_baseline"].iloc[0]
    )


def test_add_features_returns_copy() -> None:
    """入力 daily_df を変更しない（copy を返す）。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    cleaned_df = _make_cleaned_df(
        "5630_1", ["2024-01-01 00:30"], [300.0], [100.0], baseline=200.0
    )
    original_cols = set(daily_df.columns)
    _ = add_features(daily_df, cleaned_df)
    assert set(daily_df.columns) == original_cols
```

### Step 1-2: テストが失敗することを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_features_features.py -v
```

期待出力: `ImportError` or `ModuleNotFoundError`

### Step 1-3: `features.py` を実装する

- [ ] `src/gene_plug_voltage_predictor/features/features.py` を以下の内容で作成する:

```python
"""説明変数生成。Phase 0 X 列: baseline/gen_no/voltage_vs_baseline/lag/稼働割合/累積運転時間。"""
from __future__ import annotations

import pandas as pd


def add_features(
    daily_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    date_col: str = "date",
    daily_max_col: str = "daily_max",
    baseline_col: str = "baseline",
    gen_no_col: str = "gen_no",
    runtime_col: str = "累積運転時間",
    power_col: str = "発電機電力",
    datetime_col: str = "dailygraphpt_ptdatetime",
    rated_kw: float = 370.0,
    lags: tuple[int, ...] = (1, 3, 7),
) -> pd.DataFrame:
    """日次最大電圧 DataFrame に説明変数列を付与して返す（copy）。

    cleaned_df（30分粒度）から baseline/gen_no/稼働割合/累積運転時間を日次集約して
    daily_df に left join し、lag 特徴量を plug 単位で付与する。
    """
    result = daily_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])

    # cleaned_df から日次集約テーブルを作成
    tmp = cleaned_df.copy()
    tmp["_date"] = pd.to_datetime(tmp[datetime_col]).dt.normalize()
    threshold = rated_kw * 0.8

    daily_clean = tmp.groupby([plug_id_col, "_date"], sort=True).agg(
        **{
            baseline_col: (baseline_col, "first"),
            gen_no_col: (gen_no_col, "first"),
            "稼働割合": (
                power_col,
                lambda s: (s >= threshold).sum() / len(s),
            ),
            runtime_col: (runtime_col, "max"),
        }
    ).reset_index().rename(columns={"_date": date_col})

    # daily_df に left join
    result = result.merge(
        daily_clean[[plug_id_col, date_col, baseline_col, gen_no_col, "稼働割合", runtime_col]],
        on=[plug_id_col, date_col],
        how="left",
    )

    # voltage_vs_baseline
    result["voltage_vs_baseline"] = result[daily_max_col] - result[baseline_col]

    # lag 特徴量（plug 単位）
    result = result.sort_values([plug_id_col, date_col]).reset_index(drop=True)
    for lag in lags:
        col = f"{daily_max_col}_lag_{lag}"
        result[col] = (
            result.groupby(plug_id_col, sort=True)[daily_max_col]
            .shift(lag)
            .values
        )

    return result
```

### Step 1-4: テストが通ることを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_features_features.py::test_add_features_voltage_vs_baseline tests/test_features_features.py::test_add_features_baseline_nan_propagates tests/test_features_features.py::test_add_features_returns_copy -v
```

期待出力: `3 passed`

### Step 1-5: コミット

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
git add src/gene_plug_voltage_predictor/features/features.py tests/test_features_features.py
git commit -m "feat: add add_features with baseline/gen_no/voltage_vs_baseline"
```

---

## Task 2: lag 特徴量テスト、稼働割合テスト、累積運転時間テスト

**Files:**
- Modify: `tests/test_features_features.py`

### Step 2-1: 残りのテストを追加する

- [ ] `tests/test_features_features.py` の末尾に以下を追記する:

```python
def test_add_features_lags() -> None:
    """daily_max_lag_1/3/7 が plug 単位で正しくシフトされる。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 9)]  # 8 日
    values = [float(200 + i) for i in range(8)]  # 200..207
    daily_df = _make_daily_df("5630_1", dates, values)
    cleaned_df = _make_cleaned_df(
        "5630_1",
        [f"2024-01-{d:02d} 00:30" for d in range(1, 9)],
        [300.0] * 8,
        [float(100 + i) for i in range(8)],
        baseline=200.0,
    )
    result = add_features(daily_df, cleaned_df)
    # lag_1: day2(index1) の lag_1 = day1 の daily_max = 200
    assert result.loc[result["date"] == pd.Timestamp("2024-01-02"), "daily_max_lag_1"].iloc[0] == 200.0
    # lag_3: day4(index3) の lag_3 = day1 の daily_max = 200
    assert result.loc[result["date"] == pd.Timestamp("2024-01-04"), "daily_max_lag_3"].iloc[0] == 200.0
    # lag_7: day8(index7) の lag_7 = day1 の daily_max = 200
    assert result.loc[result["date"] == pd.Timestamp("2024-01-08"), "daily_max_lag_7"].iloc[0] == 200.0
    # lag_1: day1(index0) の lag_1 = NaN（先頭）
    assert pd.isna(
        result.loc[result["date"] == pd.Timestamp("2024-01-01"), "daily_max_lag_1"].iloc[0]
    )


def test_add_features_lag_no_cross_plug() -> None:
    """plug A の lag が plug B に漏れない。"""
    daily_a = _make_daily_df("A_1", ["2024-01-01", "2024-01-02"], [999.0, 999.0])
    daily_b = _make_daily_df("B_1", ["2024-01-01", "2024-01-02"], [100.0, 100.0])
    daily_df = pd.concat([daily_a, daily_b], ignore_index=True)
    cleaned_a = _make_cleaned_df(
        "A_1",
        ["2024-01-01 00:30", "2024-01-02 00:30"],
        [300.0, 300.0], [100.0, 101.0], baseline=200.0,
    )
    cleaned_b = _make_cleaned_df(
        "B_1",
        ["2024-01-01 00:30", "2024-01-02 00:30"],
        [300.0, 300.0], [100.0, 101.0], baseline=200.0,
    )
    cleaned_df = pd.concat([cleaned_a, cleaned_b], ignore_index=True)
    result = add_features(daily_df, cleaned_df)
    # B_1 の day2 の lag_1 は B_1 の day1 の値 (100.0)、A_1 の 999 ではない
    b_lag = result.loc[
        (result["管理No_プラグNo"] == "B_1") & (result["date"] == pd.Timestamp("2024-01-02")),
        "daily_max_lag_1",
    ].iloc[0]
    assert b_lag == 100.0


def test_add_features_稼働割合() -> None:
    """発電機電力 >= rated_kw * 0.8 の行数 ÷ 全行数が正しく計算される。"""
    # 4 行中 3 行が定格 80% 以上 → 稼働割合 = 0.75
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    cleaned_df = _make_cleaned_df(
        "5630_1",
        [
            "2024-01-01 00:00",
            "2024-01-01 00:30",
            "2024-01-01 01:00",
            "2024-01-01 01:30",
        ],
        [296.0, 296.0, 296.0, 200.0],  # 370 * 0.8 = 296; 最後の1行は閾値未満
        [100.0, 101.0, 102.0, 103.0],
        baseline=200.0,
        gen_no=0,
    )
    result = add_features(daily_df, cleaned_df, rated_kw=370.0)
    val = result.loc[result["date"] == pd.Timestamp("2024-01-01"), "稼働割合"].iloc[0]
    assert abs(val - 0.75) < 1e-9


def test_add_features_累積運転時間() -> None:
    """plug × 日付の累積運転時間最大値が返る。"""
    daily_df = _make_daily_df("5630_1", ["2024-01-01"], [220.0])
    cleaned_df = _make_cleaned_df(
        "5630_1",
        ["2024-01-01 00:30", "2024-01-01 01:00", "2024-01-01 01:30"],
        [300.0, 300.0, 300.0],
        [100.0, 101.0, 102.5],  # 最大値は 102.5
        baseline=200.0,
    )
    result = add_features(daily_df, cleaned_df)
    val = result.loc[result["date"] == pd.Timestamp("2024-01-01"), "累積運転時間"].iloc[0]
    assert val == 102.5
```

### Step 2-2: テストが通ることを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_features_features.py -v
```

期待出力: `7 passed`

もし `test_add_features_lags` が失敗する場合は、`add_features` の lag 計算が `groupby.shift` ではなく単純 `shift` になっていないか確認する。以下のように plug 単位 groupby を使うこと:

```python
result[col] = (
    result.groupby(plug_id_col, sort=True)[daily_max_col]
    .shift(lag)
    .values
)
```

もし `test_add_features_稼働割合` が失敗する場合は、lambda の引数が正しいか確認する:

```python
"稼働割合": (power_col, lambda s: (s >= threshold).sum() / len(s)),
```

### Step 2-3: 全テストスイートが通ることを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/ -v --tb=short -q
```

期待出力: `110 passed`（既存 103 + 新規 7）

### Step 2-4: ruff チェック

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m ruff check src/gene_plug_voltage_predictor/features/features.py tests/test_features_features.py
```

期待出力: エラーなし。E501 が出た場合は長い行を折り返す。

### Step 2-5: コミット

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
git add tests/test_features_features.py src/gene_plug_voltage_predictor/features/features.py
git commit -m "test: add feature engineering tests (lag, 稼働割合, 累積運転時間)"
```

---

## 注意事項

### `groupby` の `agg` で lambda を使う場合の ruff

ruff は `agg` 内の lambda に `E731`（lambda 代入禁止）を出すことがある。
その場合は named function に切り出す:

```python
def _active_ratio(s: pd.Series, threshold: float) -> float:
    return (s >= threshold).sum() / len(s)
```

ただし `threshold` が closure 変数のため、内部関数として定義するか `functools.partial` を使う。
最もシンプルな回避策は、`agg` の外で計算してから join する:

```python
active = (
    tmp.assign(_active=(tmp[power_col] >= threshold).astype(int))
    .groupby([plug_id_col, "_date"])["_active"]
    .mean()
    .rename("稼働割合")
    .reset_index()
    .rename(columns={"_date": date_col})
)
```

ruff E731 が出た場合はこちらの実装に切り替えること。

### `groupby(...).shift` の `.values` 代入

`groupby.shift` は元の DataFrame と同じ index を持つ Series を返すため、
`.values` で代入しても index がずれる心配はない（sort_values 後 reset_index 済みのため）。
ただし `sort=True` で `groupby` していることが前提。
