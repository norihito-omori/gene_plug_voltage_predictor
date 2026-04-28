# build_dataset CLI 設計仕様

## 概要

クリーニング済み CSV（`clean.py` の出力）から DataRobot 学習用データセット CSV を生成する CLI。
`aggregate_daily_max_voltage` → `add_features` → `add_future_7day_max_target` → NaN 除外 の
パイプラインを一括実行し、日次粒度のデータセットを出力する。

---

## スコープ

- **入力:** `clean.py` が出力した cleaned CSV（30 分粒度、縦持ち）
- **出力:** DataRobot 投入用 CSV（日次粒度、NaN 除外済み）
- **スコープ外:** DataRobot API 呼び出し、クリーニング処理

---

## ファイル構成

```
src/gene_plug_voltage_predictor/
  cli/
    clean.py             # 既存
    build_dataset.py     # 新規

tests/
  test_cli_build_dataset.py  # 新規
```

---

## CLI インターフェース

```bash
python -m gene_plug_voltage_predictor.cli.build_dataset \
  --cleaned-csv outputs/cleaned_ep370g.csv \
  --out outputs/dataset_ep370g.csv \
  [--rated-kw 370.0] \
  [--horizon 7]
```

| 引数 | 型 | 必須 | デフォルト | 説明 |
|------|----|------|-----------|------|
| `--cleaned-csv` | Path | ✓ | — | `clean.py` の出力 CSV（utf-8-sig） |
| `--out` | Path | ✓ | — | DataRobot 投入用 CSV 出力先 |
| `--rated-kw` | float | — | 370.0 | 定格出力 kW（稼働割合の閾値 = rated_kw × 0.8） |
| `--horizon` | int | — | 7 | 予測ホライズン（future_{horizon}d_max 列名に反映） |

---

## コアロジック（`build_dataset` 関数）

CLI から呼ばれる純粋関数として実装する（テスト容易性のため）：

```python
def build_dataset(
    cleaned_df: pd.DataFrame,
    *,
    rated_kw: float = 370.0,
    horizon: int = 7,
) -> pd.DataFrame:
```

**処理順序:**

1. `aggregate_daily_max_voltage(cleaned_df)` → `daily_df`（plug × 日付の日次最大電圧）
2. `add_features(daily_df, cleaned_df, rated_kw=rated_kw)` → 説明変数付与
3. `add_future_7day_max_target(daily_df, horizon=horizon)` → ターゲット列付与
4. `dropna(subset=[f"future_{horizon}d_max", "baseline"])` → 学習不能行を除外
5. 除外後 0 行の場合は警告を出力（例外は raise しない）
6. 結果を返す

---

## 処理フロー

```
--cleaned-csv (CSV ファイル)
        │
        ▼ pd.read_csv(encoding="utf-8-sig")
    cleaned_df (30分粒度)
        │
        ▼ build_dataset(cleaned_df, rated_kw, horizon)
        │
        ├── aggregate_daily_max_voltage()
        │       ↓ daily_df (plug × 日付, daily_max)
        ├── add_features(daily_df, cleaned_df)
        │       ↓ + baseline/gen_no/voltage_vs_baseline/lag×3/稼働割合/累積運転時間
        ├── add_future_7day_max_target(daily_df, horizon)
        │       ↓ + future_{horizon}d_max
        └── dropna(subset=[target_col, "baseline"])
                ↓
            dataset (日次粒度、NaN 除外済み)
        │
        ▼ to_csv(--out, encoding="utf-8-sig", index=False)
    dataset_ep370g.csv
```

---

## 出力 CSV スキーマ

| 列名 | 型 | 説明 |
|------|----|------|
| `管理No_プラグNo` | str | plug 複合 ID（DataRobot id_col 候補） |
| `date` | date | 計測日付 |
| `daily_max` | float | 日次最大電圧 |
| `baseline` | float | 世代 baseline 電圧（NaN 行は除外済み） |
| `gen_no` | int | 世代番号 |
| `voltage_vs_baseline` | float | 日次最大 − baseline |
| `daily_max_lag_1` | float | 1 日前の日次最大 |
| `daily_max_lag_3` | float | 3 日前の日次最大 |
| `daily_max_lag_7` | float | 7 日前の日次最大 |
| `稼働割合` | float | その日の高負荷運転割合（0.0〜1.0） |
| `累積運転時間` | float | その日の累積運転時間最大値 |
| `future_7d_max` | float | **ターゲット**（翌 7 カレンダー日の最大電圧、NaN 行は除外済み） |

---

## エッジケース仕様

| ケース | 挙動 |
|--------|------|
| `--cleaned-csv` が存在しない | `FileNotFoundError` が raise されて終了（argparse 後の `pd.read_csv` が自然に raise） |
| `--out` の親ディレクトリが存在しない | `out.parent.mkdir(parents=True, exist_ok=True)` で自動作成 |
| NaN 除外後に 0 行 | `logging.warning(...)` で警告し、空 CSV（ヘッダーのみ）を書く（例外なし） |
| `cleaned_df` に必要列なし | `KeyError` がそのまま伝播（CLI がエラー終了） |

---

## テスト方針

**ファイル:** `tests/test_cli_build_dataset.py`

テストは CLI の `subprocess` 呼び出しではなく、`build_dataset()` 関数を直接テストする。

| テスト名 | 内容 |
|----------|------|
| `test_build_dataset_produces_expected_columns` | 出力 DataFrame に必要列が全て含まれる |
| `test_build_dataset_drops_nan_target_rows` | `future_7d_max = NaN` の行が除外される |
| `test_build_dataset_drops_nan_baseline_rows` | `baseline = NaN` の行が除外される |
| `test_build_dataset_creates_output_dir` | `main()` 呼び出し時に `--out` の親ディレクトリを自動作成する |

---

## DataRobot 設定との対応

`config/datarobot_ep370g.json` の更新が必要:

| 設定キー | 現在値 | 更新後 |
|----------|--------|--------|
| `target_col` | `"plug_voltage"` | `"future_7d_max"` |
| `id_col` | `"location"` | `"管理No_プラグNo"` |
| `group_col` | `"location"` | `"管理No_プラグNo"` |
| `train_path` | — | `"outputs/dataset_ep370g.csv"` |

---

## 依存関係

- `pandas`（既存）
- `gene_plug_voltage_predictor.features.target`（既存）
- `gene_plug_voltage_predictor.features.features`（既存）
- 新規依存なし

---

## ADR 参照

- **ADR-013:** `aggregate_daily_max_voltage` / `add_future_7day_max_target`
- **ADR-014:** `baseline` / `gen_no`（cleaned_df に含まれる前提）
