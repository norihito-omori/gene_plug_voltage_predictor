# DataRobot TS 移行 設計仕様

## 概要

Group CV（プラグ単位）による `future_7d_max` 予測から、
DataRobot Time Series（Datetime Partitioning + Multiseries）による
`daily_max` の t+1〜t+7 予測へ移行する。

7点予測のmaxを閾値判定に使うことで `future_7d_max`（7日間最大値）と
同等の運用判断が可能になり、かつ時間的リークのない評価が実現できる。

---

## 背景・経緯

| 実験 | 手法 | RMSE | 課題 |
|------|------|------|------|
| 69967d06fb174d799c42bead（2026-02-19） | DataRobot TS t+5〜t+7 3点 | 3.13 | ターゲット=要求電圧(actual)、評価=Validationのみ |
| exp-001〜004（2026-04-28） | Group CV + future_7d_max | 0.599〜0.763 | プラグ間の時間的リークあり（楽観的RMSE） |

Group CV はプラグ間汎化を評価するが、同時期の他プラグの未来データが
学習に混入しうる構造的問題がある。
DataRobot TS の Datetime Partitioning は「T以前で学習→T以降で評価」を
厳密に守るため、実運用に近い評価が得られる。

---

## スコープ

- **新規:** `build_dataset_ts.py`（TS用パイプラインCLI）
- **新規:** `features/ts_dataset.py`（カレンダー展開・forward-fill・特徴量整理）
- **新規:** `config/datarobot_ep370g_ts.json`（TS用DataRobot設定）
- **変更:** `datarobot/partitioning.py`（`use_series_id` / `forecast_window` 対応）
- **変更:** `datarobot/config.py`（TS用フィールド追加）
- **変更なし:** `features/features.py`、`features/trend_features.py`、`cli/build_dataset.py`

---

## ファイル構成

```
src/gene_plug_voltage_predictor/
  features/
    ts_dataset.py        # 新規: build_ts_frame()
  cli/
    build_dataset_ts.py  # 新規: TS用パイプラインCLI
  datarobot/
    partitioning.py      # 変更: datetime_cv に use_series_id / forecast_window 追加
    config.py            # 変更: TS用フィールドをPartitioningConfigに追加

config/
  datarobot_ep370g_ts.json  # 新規

tests/
  test_features_ts_dataset.py  # 新規: 6テスト
```

---

## インターフェース

### `build_ts_frame()`

```python
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
```

**入力:** `add_features()` 出力済みの `daily_df`（運転日のみ行あり）

**戻り値:** カレンダー日ベースに展開した DataFrame（copy）

---

## 処理ステップ詳細

### Step 1: カレンダー展開

各プラグの `min(date)〜max(date)` を `pd.date_range(freq="D")` で生成し、
運転日の行をそこに reindex する。補完された行（非運転日）は各列が NaN になる。

### Step 2: `is_operating` 列付与

reindex 前の日付に存在した行（運転日）を 1、補完行（非運転日）を 0 とする。
後工程で DataRobot TS がターゲット欠損行の予測をスキップする。

### Step 3: forward-fill

| 列 | 補完方法 | 理由 |
|----|----------|------|
| `daily_max` | ffill | 劣化状態は継続する |
| `baseline` | ffill | 交換時にリセットされるが間はそのまま |
| `gen_no` | ffill | 交換世代は変化しない |
| `累積運転時間` | ffill | 非運転日は増加しない |
| `稼働割合` | fillna(0) | 非運転日は稼働していない |
| `voltage_vs_baseline` | ffill | daily_max と baseline から派生 |

### Step 4: `operating_hours_since_exchange` 計算

```python
origin = result.groupby([plug_id_col, gen_no_col])[runtime_col].transform("min")
result["operating_hours_since_exchange"] = result[runtime_col] - origin
```

- plug × gen_no 単位で `累積運転時間` の最小値を起点とする
- 交換直後（gen_no 変化初日）= 0
- `days_since_exchange` との違い: カレンダー日ではなく実運転時間ベース

### Step 5: `管理No` 列追加

```python
result["管理No"] = result[plug_id_col].str.split("_").str[0]
```

DataRobot TS が機場共通パターンを特徴量として学習できるようにするため。

### Step 6: TS自動生成列の除外

以下の列が存在する場合は drop する（DataRobot TS が自動生成するため重複・リーク回避）:

```python
TS_AUTO_COLS = [
    "daily_max_lag_1", "daily_max_lag_3", "daily_max_lag_7",
    "voltage_trend_7d",
    "daily_max_rolling_mean_7d", "daily_max_rolling_std_7d",
    "days_since_exchange",
    "future_7d_max",
]
```

---

## 出力 CSV カラム構成

| カラム | 由来 | DataRobot での役割 |
|--------|------|--------------------|
| `管理No_プラグNo` | そのまま | series_id |
| `管理No` | 新規追加 | 機場共通パターン特徴量 |
| `date` | そのまま | datetime_col（時系列軸） |
| `daily_max` | forward-fill済み | ターゲット列 |
| `baseline` | forward-fill | 説明変数 |
| `gen_no` | forward-fill | 説明変数 |
| `稼働割合` | fillna(0) | 説明変数 |
| `累積運転時間` | forward-fill | 説明変数 |
| `operating_hours_since_exchange` | 新規計算 | 劣化進行度（実運転時間ベース） |
| `is_operating` | 新規追加 | 非運転日フラグ（0/1） |
| `voltage_vs_baseline` | forward-fill | 説明変数 |

---

## DataRobot config（`datarobot_ep370g_ts.json`）

```json
{
  "project": {
    "name_prefix": "gene_plug_voltage_ep370g_ts",
    "endpoint": null
  },
  "data": {
    "train_path": "outputs/dataset_ep370g_ts.csv",
    "test_path": "outputs/dataset_ep370g_ts.csv",
    "id_col": "管理No_プラグNo",
    "target_col": "daily_max"
  },
  "task": {
    "task_type": "timeseries",
    "metric": null,
    "positive_class": null
  },
  "partitioning": {
    "cv_type": "datetime_cv",
    "n_folds": 5,
    "seed": 42,
    "datetime_col": "date",
    "validation_duration": "P7D",
    "use_series_id": "管理No_プラグNo",
    "forecast_window_start": 1,
    "forecast_window_end": 7,
    "group_col": null
  },
  "autopilot": {
    "mode": "quick",
    "worker_count": -1,
    "max_wait_minutes": 360,
    "text_mining": false,
    "exclude_columns": []
  },
  "output": {
    "outputs_dir": "outputs",
    "metrics_dir": "metrics"
  }
}
```

---

## `partitioning.py` の変更点

`build_partition()` の `datetime_cv` 分岐に `use_series_id` と
`forecast_window_start/end` を追加する:

```python
if cv_type == "datetime_cv":
    datetime_col = partitioning_config.get("datetime_col")
    validation_duration = partitioning_config.get("validation_duration")
    use_series_id = partitioning_config.get("use_series_id")
    forecast_window_start = partitioning_config.get("forecast_window_start", 1)
    forecast_window_end = partitioning_config.get("forecast_window_end", 1)
    if datetime_col is None:
        raise ValueError("datetime_col is required for datetime_cv")
    if validation_duration is None:
        raise ValueError("validation_duration is required for datetime_cv")
    spec = dr.DatetimePartitioningSpecification(
        datetime_partition_column=datetime_col,
        number_of_backtests=n_folds,
        validation_duration=validation_duration,
    )
    if use_series_id:
        spec.multiseries_id_columns = [use_series_id]
        spec.forecast_window_start = forecast_window_start
        spec.forecast_window_end = forecast_window_end
    return spec
```

`config.py` の `PartitioningConfig` TypedDict に以下を追加:

```python
use_series_id: str | None
forecast_window_start: int | None
forecast_window_end: int | None
```

---

## テスト方針

**ファイル:** `tests/test_features_ts_dataset.py`

| テスト名 | 内容 |
|----------|------|
| `test_calendar_expansion` | 非運転日の行が補完されてカレンダー日が連続している |
| `test_forward_fill` | daily_max が forward-fill されている（非運転日が前日値を引き継ぐ） |
| `test_is_operating` | 運転日=1、非運転日=0 |
| `test_operating_hours_reset` | gen_no 変化時に operating_hours_since_exchange が 0 にリセット |
| `test_no_ts_auto_columns` | lag/rolling/days_since_exchange/future_7d_max が出力に含まれない |
| `test_kanri_no_column` | 管理No 列が管理No_プラグNo から正しく抽出されている |

---

## エッジケース

| ケース | 挙動 |
|--------|------|
| プラグ交換直後（gen_no変化） | `operating_hours_since_exchange=0` |
| 連続した非運転日 | 全てforward-fill（最後の運転日の値が続く） |
| データ先頭の非運転日（forward-fill不可） | `daily_max=NaN`のまま（DataRobotが欠損として扱う） |
| `管理No_プラグNo` が "_" を含まない形式 | `ValueError` を raise |

---

## ADR 参照

- **ADR-013:** `aggregate_daily_max_voltage` / `add_future_7day_max_target`（既存パイプライン）
- **ADR-014:** `baseline` / `gen_no` 付与（`operating_hours_since_exchange` の起点に使用）
