# future_7d_max 予測ターゲット生成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** クリーニング済み 30 分粒度 DataFrame からプラグ単位の日次最大電圧テーブルを生成し、翌 7 カレンダー日の最大電圧 `future_7d_max` を予測ターゲット列として付与する。

**Architecture:** `features/target.py` に 2 つの純粋関数を実装する。`aggregate_daily_max_voltage` は 30 分粒度データを plug × 日付の日次最大電圧に集約し、`add_future_7day_max_target` は カレンダー日ベースの forward rolling max で `future_7d_max` 列を付与する。cleaning pipeline とは独立したモジュールとして設計する（ADR-013）。

**Tech Stack:** Python 3.13, pandas, pytest

---

## ファイル構成

| 操作 | パス | 役割 |
|------|------|------|
| 新規作成 | `src/gene_plug_voltage_predictor/features/target.py` | 2 関数の実装 |
| 新規作成 | `tests/test_features_target.py` | 全テスト |
| 変更なし | `src/gene_plug_voltage_predictor/features/__init__.py` | 既存（空）、触らない |

---

## 背景知識（ゼロコンテキスト向け）

クリーニング後の DataFrame は **縦持ち**（melt 後）で以下の列を持つ:

```
管理No, dailygraphpt_ptdatetime, target_id, 発電機電力, 累積運転時間,
target_output, mcnkind_id, 要求電圧, プラグNo, 管理No_プラグNo, gen_no, baseline
```

- `管理No_プラグNo`: 機場No_プラグNo の文字列 ID（例: `"5630_1"`）
- `dailygraphpt_ptdatetime`: 30 分粒度の計測日時（文字列 or datetime）
- `要求電圧`: プラグの電圧値（float）
- `発電機電力`: 発電機電力 kW（0 以下は停止中）

**予測ターゲットの定義（ADR-013）:**
- ある日 t の `future_7d_max` = t+1〜t+7 の運転日最大電圧の最大値
- 非運転日（その日に `発電機電力 > 0` の行がない日）は日次テーブルに行なし
- カレンダー 7 日のうち全て非運転日 → `future_7d_max = NaN`

---

## Task 1: `aggregate_daily_max_voltage` のテストと実装

**Files:**
- Create: `src/gene_plug_voltage_predictor/features/target.py`
- Create: `tests/test_features_target.py`

### Step 1-1: 失敗するテストを書く

- [ ] `tests/test_features_target.py` を以下の内容で作成する:

```python
"""tests/test_features_target.py"""
from __future__ import annotations

import pandas as pd
import pytest

from gene_plug_voltage_predictor.features.target import aggregate_daily_max_voltage


def _make_30min_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["dailygraphpt_ptdatetime"] = pd.to_datetime(df["dailygraphpt_ptdatetime"])
    return df


def test_aggregate_daily_max_voltage_basic() -> None:
    """2 plug × 2 日で daily_max が正しく集約される。"""
    df = _make_30min_df([
        # plug A, day 1: 電圧 220/225 → daily_max=225
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-01 00:30", "要求電圧": 220.0, "発電機電力": 300.0},
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-01 01:00", "要求電圧": 225.0, "発電機電力": 300.0},
        # plug A, day 2: 電圧 230 → daily_max=230
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-02 00:30", "要求電圧": 230.0, "発電機電力": 300.0},
        # plug B, day 1: 電圧 210 → daily_max=210
        {"管理No_プラグNo": "5630_2", "dailygraphpt_ptdatetime": "2024-01-01 00:30", "要求電圧": 210.0, "発電機電力": 300.0},
    ])
    result = aggregate_daily_max_voltage(df)
    assert set(result.columns) == {"管理No_プラグNo", "date", "daily_max"}
    a1 = result[(result["管理No_プラグNo"] == "5630_1") & (result["date"] == pd.Timestamp("2024-01-01"))]
    assert a1["daily_max"].iloc[0] == 225.0
    a2 = result[(result["管理No_プラグNo"] == "5630_1") & (result["date"] == pd.Timestamp("2024-01-02"))]
    assert a2["daily_max"].iloc[0] == 230.0
    b1 = result[(result["管理No_プラグNo"] == "5630_2") & (result["date"] == pd.Timestamp("2024-01-01"))]
    assert b1["daily_max"].iloc[0] == 210.0


def test_aggregate_daily_max_voltage_excludes_non_running() -> None:
    """発電機電力 <= 0 の行は集約対象外。"""
    df = _make_30min_df([
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-01 00:30", "要求電圧": 250.0, "発電機電力": 0.0},
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-01 01:00", "要求電圧": 220.0, "発電機電力": 300.0},
    ])
    result = aggregate_daily_max_voltage(df)
    assert result["daily_max"].iloc[0] == 220.0  # 250 は停止中なので除外


def test_aggregate_daily_max_voltage_no_running_day_omitted() -> None:
    """全非運転日（発電機電力 <= 0 のみ）は日次テーブルに行が存在しない。"""
    df = _make_30min_df([
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-01 00:30", "要求電圧": 220.0, "発電機電力": 300.0},
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-02 00:30", "要求電圧": 999.0, "発電機電力": 0.0},
        {"管理No_プラグNo": "5630_1", "dailygraphpt_ptdatetime": "2024-01-03 00:30", "要求電圧": 225.0, "発電機電力": 300.0},
    ])
    result = aggregate_daily_max_voltage(df)
    dates = result["date"].tolist()
    assert pd.Timestamp("2024-01-02") not in dates
    assert pd.Timestamp("2024-01-01") in dates
    assert pd.Timestamp("2024-01-03") in dates
```

### Step 1-2: テストが失敗することを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_features_target.py -v
```

期待出力: `ImportError` または `ModuleNotFoundError: No module named 'gene_plug_voltage_predictor.features.target'`

### Step 1-3: `target.py` を実装する

- [ ] `src/gene_plug_voltage_predictor/features/target.py` を以下の内容で作成する:

```python
"""予測ターゲット生成。ADR-013: future_7d_max = t+1〜t+7 の日次最大電圧の最大。"""
from __future__ import annotations

import pandas as pd


def aggregate_daily_max_voltage(
    df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    datetime_col: str = "dailygraphpt_ptdatetime",
    voltage_col: str = "要求電圧",
    power_col: str = "発電機電力",
) -> pd.DataFrame:
    """30 分粒度 DataFrame → plug × 日付の日次最大電圧テーブルを返す。

    非運転日（その日に発電機電力 > 0 の行が 0 件）は行を生成しない。
    返却カラム: [plug_id_col, "date", "daily_max"]
    """
    running = df.loc[df[power_col] > 0].copy()
    running["date"] = pd.to_datetime(running[datetime_col]).dt.normalize()
    daily = (
        running.groupby([plug_id_col, "date"], sort=True)[voltage_col]
        .max()
        .rename("daily_max")
        .reset_index()
    )
    return daily


def add_future_7day_max_target(
    daily_df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    date_col: str = "date",
    daily_max_col: str = "daily_max",
    horizon: int = 7,
) -> pd.DataFrame:
    """日次最大電圧 DataFrame に future_7d_max 列を付与して返す（copy）。

    翌 7 カレンダー日（t+1〜t+horizon）の daily_max の最大値を計算する。
    非運転日は行なし（カレンダー補完して NaN として扱う）。
    ウィンドウ内が全て NaN → future_7d_max = NaN。
    """
    result = daily_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([plug_id_col, date_col])

    future_maxes: list[pd.Series] = []
    for plug_id, group in result.groupby(plug_id_col, sort=False):
        group = group.set_index(date_col)
        full_idx = pd.date_range(group.index.min(), group.index.max(), freq="D")
        reindexed = group[daily_max_col].reindex(full_idx)

        shifted = reindexed.shift(-1)
        fm = shifted[::-1].rolling(horizon, min_periods=1).max()[::-1]
        # 終端 horizon 行（元データ最後の日から horizon-1 日前まで）を NaN に
        # shift(-1) 後に全て NaN になる範囲は rolling が 0 個のデータを扱うため
        # min_periods=1 では NaN にならない。以下で明示的に NaN を設定する。
        if len(fm) >= 1:
            # t+1〜t+horizon が全て元データの末尾外になる行を特定
            last_valid_date = group.index.max()
            cutoff_date = last_valid_date - pd.Timedelta(days=horizon - 1)
            fm.loc[fm.index > cutoff_date] = float("nan")

        future_maxes.append(fm.reindex(group.index))

    all_future = pd.concat(future_maxes)
    result = result.reset_index(drop=True)
    result["future_7d_max"] = all_future.values
    return result
```

### Step 1-4: テストが通ることを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_features_target.py::test_aggregate_daily_max_voltage_basic tests/test_features_target.py::test_aggregate_daily_max_voltage_excludes_non_running tests/test_features_target.py::test_aggregate_daily_max_voltage_no_running_day_omitted -v
```

期待出力: `3 passed`

### Step 1-5: コミット

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
git add src/gene_plug_voltage_predictor/features/target.py tests/test_features_target.py
git commit -m "feat: add aggregate_daily_max_voltage (ADR-013)"
```

---

## Task 2: `add_future_7day_max_target` のテスト

**Files:**
- Modify: `tests/test_features_target.py`

### Step 2-1: 失敗するテストを追加する

- [ ] `tests/test_features_target.py` の末尾に以下を追記する（import に `add_future_7day_max_target` を追加することも忘れずに）:

```python
from gene_plug_voltage_predictor.features.target import (
    add_future_7day_max_target,
    aggregate_daily_max_voltage,
)
```

（ファイル冒頭の import 行を上記に置き換える）

テストコードを末尾に追記:

```python
def _make_daily_df(plug_id: str, dates: list[str], values: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame({
        "管理No_プラグNo": plug_id,
        "date": pd.to_datetime(dates),
        "daily_max": [float("nan") if v is None else v for v in values],
    })


def test_add_future_7day_max_target_basic() -> None:
    """10 日データで future_7d_max が正しく計算される。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    values = [float(200 + i) for i in range(10)]  # 200, 201, ..., 209
    df = _make_daily_df("5630_1", dates, values)
    result = add_future_7day_max_target(df)
    # day 1 (index 0): t+1〜t+7 = days 2〜8 → max(201..207) = 207
    assert result.loc[result["date"] == pd.Timestamp("2024-01-01"), "future_7d_max"].iloc[0] == 207.0
    # day 3 (index 2): t+1〜t+7 = days 4〜10 → max(203..209) = 209
    assert result.loc[result["date"] == pd.Timestamp("2024-01-03"), "future_7d_max"].iloc[0] == 209.0


def test_add_future_7day_max_target_terminal_nan() -> None:
    """最後の horizon 行は future_7d_max = NaN。"""
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]  # 10 日
    values = [220.0] * 10
    df = _make_daily_df("5630_1", dates, values)
    result = add_future_7day_max_target(df, horizon=7)
    # 最後の 7 日（day 4〜10）は t+1〜t+7 が範囲外 → NaN
    terminal = result[result["date"] >= pd.Timestamp("2024-01-04")]
    assert terminal["future_7d_max"].isna().all()


def test_add_future_7day_max_target_plug_isolation() -> None:
    """plug A の値が plug B の future_7d_max に影響しない。"""
    df_a = _make_daily_df("A_1", ["2024-01-01", "2024-01-02", "2024-01-03"], [999.0, 999.0, 999.0])
    df_b = _make_daily_df("B_1", ["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 100.0, 100.0])
    df = pd.concat([df_a, df_b], ignore_index=True)
    result = add_future_7day_max_target(df, horizon=7)
    b_vals = result[result["管理No_プラグNo"] == "B_1"]["future_7d_max"].dropna()
    assert (b_vals <= 100.0).all()


def test_add_future_7day_max_target_partial_nan() -> None:
    """一部 NaN（非運転日）はスキップして残りの最大を返す。"""
    # day 1 → 220, day 2 → (非運転日=行なし), day 3 → 230
    # future_7d_max of day 0 = max(220, 230) = 230
    df = pd.DataFrame({
        "管理No_プラグNo": ["5630_1", "5630_1", "5630_1"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
        "daily_max": [210.0, 220.0, 230.0],
    })
    result = add_future_7day_max_target(df, horizon=7)
    # day 1 の future_7d_max: t+1〜t+7 = 2〜8日。稼働は 2日(220)と4日(230) → max=230
    val = result.loc[result["date"] == pd.Timestamp("2024-01-01"), "future_7d_max"].iloc[0]
    assert val == 230.0


def test_add_future_7day_max_target_calendar_gap() -> None:
    """非運転日（行なし）を挟んでもカレンダー 7 日で正しく計算される。"""
    # day 1: 220, day 8: 999 → day 1 の t+1〜t+7 (day2〜8) に day8 は含まれる
    # day 1 の t+1〜t+7 (day2〜8) に day8(=999) が入る → future_7d_max=999
    df = pd.DataFrame({
        "管理No_プラグNo": ["5630_1", "5630_1", "5630_1"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-08"]),
        "daily_max": [220.0, 200.0, 999.0],
    })
    result = add_future_7day_max_target(df, horizon=7)
    val = result.loc[result["date"] == pd.Timestamp("2024-01-01"), "future_7d_max"].iloc[0]
    assert val == 999.0
    # day 9 は t+1〜t+7 の範囲外（t+8）なので future_7d_max は 999 を含まない
    df2 = pd.DataFrame({
        "管理No_プラグNo": ["5630_1", "5630_1", "5630_1"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-09"]),
        "daily_max": [220.0, 200.0, 999.0],
    })
    result2 = add_future_7day_max_target(df2, horizon=7)
    val2 = result2.loc[result2["date"] == pd.Timestamp("2024-01-01"), "future_7d_max"].iloc[0]
    assert val2 == 200.0  # day9 は t+8 なのでウィンドウ外
```

### Step 2-2: テストが失敗することを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_features_target.py -v -k "future"
```

期待出力: `ImportError` または一部テストが `FAILED`（実装は Task 1 で書いたが、各テストの境界値が合っているか確認のため）

### Step 2-3: 全テストを実行して通ることを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_features_target.py -v
```

期待出力: `8 passed`

もし `test_add_future_7day_max_target_terminal_nan` が失敗する場合は、`add_future_7day_max_target` の cutoff 計算を以下のように修正する:

```python
# 終端 NaN の設定: データの最後の日より horizon 日以内の日は全て NaN
last_date = group.index.max()
terminal_start = last_date - pd.Timedelta(days=horizon - 1)
fm.loc[fm.index >= terminal_start] = float("nan")
```

`target.py` の該当箇所（`cutoff_date` の行）を上記で置き換える。

### Step 2-4: 全テストスイートが通ることを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/ -v
```

期待出力: `103 passed`（既存 95 + 新規 8）

### Step 2-5: ruff チェック

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m ruff check src/gene_plug_voltage_predictor/features/target.py tests/test_features_target.py
```

期待出力: エラーなし。もし `E501`（行長）が出たら長い行を折り返す。

### Step 2-6: コミット

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
git add tests/test_features_target.py src/gene_plug_voltage_predictor/features/target.py
git commit -m "feat: add add_future_7day_max_target with calendar-day rolling (ADR-013)"
```

---

## 注意事項

### `add_future_7day_max_target` の `values` 割り当てについて

Task 1 の実装で `result["future_7d_max"] = all_future.values` としているが、
`groupby(..., sort=False)` の順序と `result` のソート順が一致している必要がある。
`result = result.sort_values([plug_id_col, date_col])` 後に `groupby` しているため整合しているが、
テスト失敗時は `result` を確認すること。

より安全な代替実装（インデックスベース）:

```python
# plug 単位処理後に pd.concat でインデックス付き Series を結合
records: list[pd.Series] = []
for plug_id, group in result.groupby(plug_id_col, sort=True):
    # ... rolling 計算 ...
    s = pd.Series(fm.reindex(group[date_col].values).values,
                  index=group.index, name="future_7d_max")
    records.append(s)
result["future_7d_max"] = pd.concat(records).sort_index()
```

テスト失敗時はこちらの実装に切り替えること。
