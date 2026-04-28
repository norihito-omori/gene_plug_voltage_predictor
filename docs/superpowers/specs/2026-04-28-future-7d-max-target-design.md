# future_7d_max 予測ターゲット生成 設計仕様

## 概要

クリーニング済み 30 分粒度データから、プラグ単位の日次最大電圧テーブルを生成し、
t+1〜t+7 日の最大値（`future_7d_max`）を予測ターゲット列として付与する（ADR-013）。

---

## スコープ

- **対象:** EP370G / EP400G 共通（クリーニング後の正規化済み列名を前提）
- **入力:** `CleaningPipeline.run()` の出力 DataFrame（30 分粒度、縦持ち）
- **出力:** `(管理No_プラグNo, date, daily_max, future_7d_max)` の日次 DataFrame
- **スコープ外:** 排気温度などセンサー系説明変数（別 ADR で対応）

---

## ファイル構成

```
src/gene_plug_voltage_predictor/
  features/
    __init__.py          # 既存（空）
    target.py            # 新規：aggregate_daily_max_voltage + add_future_7day_max_target

tests/
  test_features_target.py  # 新規
```

---

## インターフェース

### `aggregate_daily_max_voltage`

```python
def aggregate_daily_max_voltage(
    df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    datetime_col: str = "dailygraphpt_ptdatetime",
    voltage_col: str = "要求電圧",
    power_col: str = "発電機電力",
) -> pd.DataFrame:
```

**責務:**
- `df` は 30 分粒度・縦持ち（melt 後）の DataFrame
- `発電機電力 > 0` の行（運転中）のみを対象に、`plug_id_col × 日付` でグループ化
- 各グループの `voltage_col` の最大値を `daily_max` として返す
- 非運転日（その日に running row が 1 件もない日）は行自体が存在しない（NaN 行を生成しない）
- 返却カラム: `[plug_id_col, "date", "daily_max"]`
- `date` は `datetime_col` を `dt.normalize()` で日付に変換したもの

**戻り値の型:**

| 列 | dtype |
|----|-------|
| `管理No_プラグNo` | object (str) |
| `date` | datetime64[ns] |
| `daily_max` | float64 |

---

### `add_future_7day_max_target`

```python
def add_future_7day_max_target(
    daily_df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    date_col: str = "date",
    daily_max_col: str = "daily_max",
    horizon: int = 7,
) -> pd.DataFrame:
```

**責務:**
- `daily_df` は `aggregate_daily_max_voltage` の出力（または同スキーマの DataFrame）
- plug 単位でソート後、`t+1〜t+horizon` の `daily_max` の最大値を `future_7d_max` 列として付与
- ウィンドウ計算は `groupby(plug_id_col).shift(-1).rolling(horizon, min_periods=1).max()` に相当
  - 厳密には: plug 単位で `daily_max` を 1 行シフト（t+1 起算）し、forward rolling max を取る
- ウィンドウ内が全て NaN（非運転日のみ）→ NaN
- 終端 horizon 行分は NaN（データが存在しないため）
- in-place 変更ではなく copy を返す

**実装メモ（rolling forward max）:**

```python
# plug 単位で処理
shifted = group["daily_max"].shift(-1)           # t+1 起算
future_max = shifted[::-1].rolling(horizon, min_periods=1).max()[::-1]
```

ただし `min_periods=1` により一部 NaN の場合は非 NaN 値の最大を返す。
終端の `horizon` 行目以降（シフト後に NaN しかない範囲）は NaN になる。

---

## データフロー

```
cleaned_df
  (管理No_プラグNo, dailygraphpt_ptdatetime, 要求電圧, 発電機電力, gen_no, baseline, ...)
         │
         ▼  aggregate_daily_max_voltage()
daily_df
  (管理No_プラグNo, date, daily_max)
         │
         ▼  add_future_7day_max_target()
daily_df
  (管理No_プラグNo, date, daily_max, future_7d_max)
```

---

## エッジケース仕様

| ケース | 挙動 |
|--------|------|
| その日の running row（発電機電力 > 0）が 0 件 | 日次テーブルに行なし（NaN 行を生成しない） |
| t+1〜t+7 の `daily_max` が全て NaN | `future_7d_max = NaN` |
| t+1〜t+7 の一部が NaN（非運転日）| NaN をスキップして残りの最大を返す |
| plug のデータが horizon 日未満で終端 | 終端行は `future_7d_max = NaN` |
| plug 間でデータが混ざる | `groupby(plug_id_col)` で完全分離 |

---

## テスト方針

**ファイル:** `tests/test_features_target.py`

| テスト名 | 内容 |
|----------|------|
| `test_aggregate_daily_max_voltage_basic` | 2 plug × 14 日で daily_max が正しく集約される |
| `test_aggregate_daily_max_voltage_excludes_non_running` | 発電機電力 <= 0 の行は集約対象外 |
| `test_aggregate_daily_max_voltage_no_running_day_omitted` | 全非運転日は行が存在しない |
| `test_add_future_7day_max_target_basic` | 14 日データで future_7d_max の値が正しい |
| `test_add_future_7day_max_target_terminal_nan` | 最後の horizon 行は NaN |
| `test_add_future_7day_max_target_plug_isolation` | plug A の値が plug B に漏れない |
| `test_add_future_7day_max_target_partial_nan` | 一部 NaN の場合は非 NaN 最大を返す |

---

## 依存関係

- `pandas` （既存依存）
- `numpy` （既存依存）
- 新規依存なし

---

## ADR 参照

- **ADR-013:** 予測ターゲット `future_7d_max` 定義
- **ADR-009:** `melt_voltage_columns` — 縦持ち変換（入力前提）
- **ADR-014:** `gen_no` / `baseline` 付与（入力前提）
