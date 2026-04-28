# 説明変数生成（Feature Engineering）設計仕様

## 概要

クリーニング済み 30 分粒度 DataFrame と日次最大電圧テーブルから、
DataRobot 学習に使う説明変数（X）列を生成する（Phase 0）。

---

## スコープ

- **対象:** EP370G / EP400G 共通
- **入力 1:** `daily_df` — `aggregate_daily_max_voltage` 出力 + baseline/gen_no（30分粒度から日次 join）
- **入力 2:** `cleaned_df` — `CleaningPipeline.run()` の 30 分粒度出力（稼働割合・累積運転時間の計算に使用）
- **出力:** `daily_df` に X 列を追加した copy
- **スコープ外:** センサー列（P入口空気温、給気マニホルド圧、主副室ガス圧）— 別 ADR で対応

---

## ファイル構成

```
src/gene_plug_voltage_predictor/
  features/
    __init__.py        # 既存（空）
    target.py          # 既存（ADR-013）
    features.py        # 新規：add_features()

tests/
  test_features_features.py  # 新規
```

---

## インターフェース

```python
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
```

**`daily_df` の期待カラム:**
- `管理No_プラグNo`: plug 複合 ID
- `date`: 日付（datetime64[ns]）
- `daily_max`: 日次最大電圧（float）

**`cleaned_df` の期待カラム:**
- `管理No_プラグNo`: plug 複合 ID
- `dailygraphpt_ptdatetime`: 30 分粒度日時
- `発電機電力`: kW（運転判定・稼働割合に使用）
- `累積運転時間`: 時間（日次最大値に使用）
- `baseline`: 世代開始 30 日間の日次最大中央値（全行に broadcast 済み）
- `gen_no`: 世代番号（全行に broadcast 済み）

**戻り値:** `daily_df` の copy に X 列を追加したもの（in-place 変更なし）

---

## 追加される列

| 列名 | 定義 | ソース |
|------|------|--------|
| `baseline` | 世代開始 30 日間の日次最大電圧中央値 | `cleaned_df` から plug × 日付の `first()` で join |
| `gen_no` | 世代番号（0-origin） | 同上 |
| `voltage_vs_baseline` | `daily_max - baseline` | `daily_df` + join 後に計算 |
| `daily_max_lag_1` | 1 日前の `daily_max`（plug 単位） | `daily_df` の `groupby.shift(-1)` |
| `daily_max_lag_3` | 3 日前の `daily_max`（plug 単位） | 同上 |
| `daily_max_lag_7` | 7 日前の `daily_max`（plug 単位） | 同上 |
| `稼働割合` | `発電機電力 >= rated_kw × 0.8` の行数 ÷ 全行数（plug × 日付） | `cleaned_df` から計算して join |
| `累積運転時間` | plug × 日付の `累積運転時間` 最大値 | `cleaned_df` から計算して join |

**備考:**
- `baseline` と `gen_no` は `cleaned_df` に 30 分粒度で broadcast されているため、plug × 日付でグループ化して `first()` で日次値を取得し join する。呼び出し側での事前 join は不要。
- lag は plug 単位（`groupby(plug_id_col)` 後に `shift`）。系列先頭は `NaN`。
- `稼働割合` の計算は非運転日（`daily_df` に行なし）も対象とする。`cleaned_df` に行がある日は電力 ≤ 0 で稼働割合 0 となる。`cleaned_df` にも行がない日は `daily_df` に存在しないため計算対象外。

---

## データフロー

```
cleaned_df (30分粒度)
    │
    ├──→ aggregate_daily_max_voltage()   [target.py]
    │         ↓
    │     daily_df (管理No_プラグNo, date, daily_max)
    │         │
    ├──→ add_features(daily_df, cleaned_df)   [features.py]
    │         │
    │     内部処理:
    │     1. cleaned_df から baseline / gen_no / 稼働割合 / 累積運転時間 を日次集約
    │     2. daily_df に left join
    │     3. voltage_vs_baseline 計算
    │     4. lag 特徴量計算（plug 単位 groupby + shift）
    │         ↓
    │     daily_df (+ baseline, gen_no, voltage_vs_baseline,
    │                daily_max_lag_1/3/7, 稼働割合, 累積運転時間)
    │
    └──→ add_future_7day_max_target()   [target.py]
              ↓
          daily_df (+ future_7d_max)   ← 最終データセット
```

---

## エッジケース仕様

| ケース | 挙動 |
|--------|------|
| lag 期間に前データなし（series 先頭） | `NaN` |
| `baseline = NaN`（世代の活動日 7 日未満） | `voltage_vs_baseline = NaN` |
| 非運転日（`daily_df` に行なし、`cleaned_df` には行あり） | `稼働割合 = 0.0` |
| `rated_kw` 未指定 | デフォルト `370.0`（EP370G 基準） |
| plug A と plug B のラグが混在 | `groupby(plug_id_col)` で完全分離 |

---

## テスト方針

**ファイル:** `tests/test_features_features.py`

| テスト名 | 内容 |
|----------|------|
| `test_add_features_voltage_vs_baseline` | `daily_max - baseline` が正しく計算される |
| `test_add_features_lags` | `daily_max_lag_1/3/7` が plug 単位で正しくシフトされる |
| `test_add_features_lag_no_cross_plug` | plug A の lag が plug B に漏れない |
| `test_add_features_稼働割合` | `発電機電力 >= rated_kw × 0.8` の割合が正しい |
| `test_add_features_累積運転時間` | plug × 日付の最大値が返る |
| `test_add_features_returns_copy` | 入力 `daily_df` を変更しない |
| `test_add_features_baseline_nan_propagates` | `baseline=NaN` のとき `voltage_vs_baseline=NaN` |

---

## 依存関係

- `pandas`（既存）
- `numpy`（既存）
- 新規依存なし

---

## ADR 参照

- **ADR-013:** `aggregate_daily_max_voltage` / `add_future_7day_max_target`（入力前提）
- **ADR-014:** `baseline` / `gen_no` 付与（入力前提）
- **ADR-009:** `melt_voltage_columns`（`管理No_プラグNo` 生成）
