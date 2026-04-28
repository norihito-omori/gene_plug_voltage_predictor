# トレンド・統計特徴量追加 設計仕様

## 概要

Phase 0 説明変数に電圧トレンド・交換経過日数・rolling 統計量を追加する。
`add_features()` の出力（日次粒度）を入力とし、`daily_df` 単体から計算できる
4列を `add_trend_features()` で付与する。

---

## スコープ

- **入力:** `add_features()` 出力済みの `daily_df`（`gen_no` 列を含む）
- **出力:** 4列を追加した `daily_df` の copy
- **スコープ外:** センサー列（P入口空気温 etc.）、機場レベルの集計

---

## ファイル構成

```
src/gene_plug_voltage_predictor/
  features/
    features.py        # 既存（変更なし）
    trend_features.py  # 新規: add_trend_features()

tests/
  test_features_trend.py  # 新規
```

---

## インターフェース

```python
def add_trend_features(
    daily_df: pd.DataFrame,
    *,
    plug_id_col: str = "管理No_プラグNo",
    date_col: str = "date",
    daily_max_col: str = "daily_max",
    gen_no_col: str = "gen_no",
    trend_window: int = 7,
) -> pd.DataFrame:
```

**`daily_df` の期待カラム:**
- `管理No_プラグNo`: plug 複合 ID
- `date`: 日付（datetime64[ns]）
- `daily_max`: 日次最大電圧（float）
- `gen_no`: 世代番号（int）— `add_features()` 後に呼ぶ前提

**戻り値:** `daily_df` の copy に 4列を追加したもの（in-place 変更なし）

---

## 追加される列

| 列名 | 定義 | 計算方法 |
|------|------|---------|
| `voltage_trend_7d` | 直近7日（運転日ベース）の電圧傾き（V/日） | plug単位の `daily_max` に対して7行 rolling で線形回帰の傾きを計算（min_periods=2） |
| `days_since_exchange` | 現世代（gen_no）の初日からの経過日数（0-origin） | plug × gen_no の最小 date を起点に `(date - origin).days` |
| `daily_max_rolling_mean_7d` | 直近7日（運転日ベース）の移動平均 | plug単位の7行 rolling mean（min_periods=1） |
| `daily_max_rolling_std_7d` | 直近7日（運転日ベース）の移動標準偏差 | plug単位の7行 rolling std（min_periods=2、未満はNaN） |

**備考:**
- rolling は非運転日（行なし）をスキップする行ベース。カレンダー日ではない。
- `voltage_trend_7d` の傾き計算は `numpy.polyfit(range(n), values, 1)[0]` を
  `rolling.apply()` で適用する。`min_periods=2` で先頭1行は NaN。
- `days_since_exchange` は plug × gen_no の `date.min()` を起点とするため、
  プラグ交換直後（gen_no が変わった最初の日）は 0。
- rolling 系の先頭 NaN は `build_dataset.py` の `dropna` 対象外（target/baseline のみ）。

---

## パイプライン変更

`build_dataset.py` の `build_dataset()` 関数に1行追加:

```python
daily = aggregate_daily_max_voltage(cleaned_df)
daily = add_features(daily, cleaned_df, rated_kw=rated_kw)
daily = add_trend_features(daily)                           # ← 追加
daily = add_future_7day_max_target(daily, horizon=horizon)
daily = daily.dropna(subset=[target_col, "baseline"]).reset_index(drop=True)
```

---

## エッジケース仕様

| ケース | 挙動 |
|--------|------|
| rolling window 未満（先頭行） | `voltage_trend_7d`=NaN（min_periods=2）、`rolling_mean`=有効値（min_periods=1）、`rolling_std`=NaN（min_periods=2） |
| プラグ交換初日（gen_no変化） | `days_since_exchange`=0 |
| plug A の rolling が plug B に漏れる | `groupby(plug_id_col)` で完全分離 |
| `gen_no` 列が存在しない | `KeyError` がそのまま伝播 |

---

## テスト方針

**ファイル:** `tests/test_features_trend.py`

| テスト名 | 内容 |
|----------|------|
| `test_add_trend_features_columns` | 4列が全て追加されている |
| `test_add_trend_features_returns_copy` | 入力 `daily_df` を変更しない |
| `test_voltage_trend_7d_slope` | 単調増加系列で傾き > 0、単調減少で傾き < 0 |
| `test_days_since_exchange_starts_at_zero` | 各世代の初日が 0 |
| `test_days_since_exchange_increments` | 翌日が 1、その翌日が 2 |
| `test_rolling_mean_7d` | 既知値で rolling mean が正しい |
| `test_rolling_std_7d_nan_for_single_row` | 1行のみの場合 rolling_std が NaN |
| `test_no_cross_plug_leakage` | plug A の rolling が plug B に影響しない |

---

## 依存関係

- `pandas`（既存）
- `numpy`（既存）
- 新規依存なし

---

## ADR 参照

- **ADR-013:** `aggregate_daily_max_voltage` / `add_future_7day_max_target`（入力前提）
- **ADR-014:** `baseline` / `gen_no` 付与（`days_since_exchange` の起点に使用）
