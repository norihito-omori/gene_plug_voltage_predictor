# DataRobot TS 移行 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group CV + `future_7d_max` パイプラインを DataRobot Time Series（Datetime Partitioning + Multiseries t+1〜t+7）へ移行し、時間的リークのない評価基盤を構築する。

**Architecture:** `build_ts_frame()` がカレンダー展開・forward-fill・TS自動生成列除外を担い、`build_dataset_ts.py` がそれを呼ぶCLI。`partitioning.py` に `use_series_id` / `forecast_window` を追加し、`config.py` の TypedDict を拡張する。既存の Group CV パイプライン（`build_dataset.py`）は変更しない。

**Tech Stack:** Python 3.13, pandas, numpy, datarobot SDK, pytest

---

## ファイル構成

| 操作 | パス | 役割 |
|------|------|------|
| 新規作成 | `src/gene_plug_voltage_predictor/features/ts_dataset.py` | `build_ts_frame()` 実装 |
| 新規作成 | `tests/test_features_ts_dataset.py` | 6テスト |
| 新規作成 | `src/gene_plug_voltage_predictor/cli/build_dataset_ts.py` | TS用パイプラインCLI |
| 新規作成 | `config/datarobot_ep370g_ts.json` | TS用DataRobot設定ファイル |
| 変更 | `src/gene_plug_voltage_predictor/datarobot/partitioning.py` | `datetime_cv` に `use_series_id` / `forecast_window` 追加 |
| 変更 | `src/gene_plug_voltage_predictor/datarobot/config.py` | `PartitioningConfig` に TS用フィールド追加 |
| 変更 | `tests/datarobot/test_partitioning.py` | multiseries テスト追加 |
| 変更 | `tests/datarobot/test_config_validation.py` | TS config バリデーションテスト追加 |

---

## 背景知識（ゼロコンテキスト向け）

### プロジェクト概要

EP370G 発電機プラグの電圧を予測するモデル。30分粒度のセンサーデータを日次集計し、DataRobot で学習する。

### 現在のパイプライン（変更なし）

```
cleaned CSV（30分粒度）
  → aggregate_daily_max_voltage()  → 運転日のみの daily_max テーブル
  → add_features()                 → baseline/gen_no/lag/稼働割合/累積運転時間を追加
  → add_trend_features()           → rolling/trend 特徴量を追加
  → add_future_7day_max_target()   → future_7d_max（7日間max）を追加
  → dataset_ep370g.csv             → DataRobot（Group CV）へ投入
```

### 新しいパイプライン（今回実装）

```
cleaned CSV（30分粒度）
  → aggregate_daily_max_voltage()  → 既存流用
  → add_features()                 → 既存流用
  → build_ts_frame()               ← 新規（カレンダー展開・ffill・TS列整理）
  → dataset_ep370g_ts.csv          → DataRobot TS（Datetime Partitioning）へ投入
```

### `add_features()` の出力スキーマ（`build_ts_frame()` の入力）

```
管理No_プラグNo, date, daily_max, baseline, gen_no, 稼働割合, 累積運転時間,
voltage_vs_baseline, daily_max_lag_1, daily_max_lag_3, daily_max_lag_7
```

（`add_trend_features()` を通した場合はさらに voltage_trend_7d 等が追加されるが、
`build_dataset_ts.py` では `add_trend_features()` は呼ばない）

### TS自動生成列（DataRobot TS が自動生成するため除外する列）

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

## Task 1: `build_ts_frame()` の実装とテスト

**Files:**
- Create: `src/gene_plug_voltage_predictor/features/ts_dataset.py`
- Create: `tests/test_features_ts_dataset.py`

### Step 1-1: テストファイルを作成する

- [ ] `tests/test_features_ts_dataset.py` を以下の内容で作成する:

```python
"""tests/test_features_ts_dataset.py"""
from __future__ import annotations

import pandas as pd
import pytest

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
    """daily_max が非運転日に forward-fill されている。"""
    daily_df = _make_daily_df(
        "5630_1",
        ["2024-01-01", "2024-01-03"],
        [220.0, 225.0],
    )
    result = build_ts_frame(daily_df)
    plug_df = result[result["管理No_プラグNo"] == "5630_1"].sort_values("date").reset_index(drop=True)
    # 2024-01-02 は NaN ではなく 220.0（前日の値）
    assert plug_df.loc[1, "daily_max"] == 220.0


def test_is_operating() -> None:
    """運転日は is_operating=1、補完された非運転日は is_operating=0。"""
    daily_df = _make_daily_df(
        "5630_1",
        ["2024-01-01", "2024-01-03"],
        [220.0, 222.0],
    )
    result = build_ts_frame(daily_df)
    plug_df = result[result["管理No_プラグNo"] == "5630_1"].sort_values("date").reset_index(drop=True)
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
    plug_df = result[result["管理No_プラグNo"] == "5630_1"].sort_values("date").reset_index(drop=True)
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
```

### Step 1-2: テストが失敗することを確認

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_features_ts_dataset.py -v
```

期待出力: `ImportError` または `ModuleNotFoundError`

### Step 1-3: `ts_dataset.py` を実装する

- [ ] `src/gene_plug_voltage_predictor/features/ts_dataset.py` を以下の内容で作成する:

```python
"""DataRobot TS 用データセット変換。カレンダー展開・forward-fill・TS自動生成列除外。"""
from __future__ import annotations

import pandas as pd

TS_AUTO_COLS: list[str] = [
    "daily_max_lag_1",
    "daily_max_lag_3",
    "daily_max_lag_7",
    "voltage_trend_7d",
    "daily_max_rolling_mean_7d",
    "daily_max_rolling_std_7d",
    "days_since_exchange",
    "future_7d_max",
]

_FFILL_COLS = ["daily_max", "baseline", "gen_no", "累積運転時間", "voltage_vs_baseline"]


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
    """add_features() 出力をカレンダー日ベースに展開して DataRobot TS 用に整形する。

    非運転日を forward-fill で補完し、DataRobot TS が自動生成する lag/rolling 列を除外する。
    .copy() で入力を変更しない。
    """
    result = daily_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([plug_id_col, date_col]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for plug_id, group in result.groupby(plug_id_col, sort=True):
        group = group.set_index(date_col)
        full_idx = pd.date_range(group.index.min(), group.index.max(), freq="D")

        # is_operating: 元の運転日=1、補完行=0
        operating_flags = group.index.isin(group.index)
        is_op = pd.Series(1, index=group.index, name="is_operating")

        # カレンダー展開
        expanded = group.reindex(full_idx)
        expanded["is_operating"] = 0
        expanded.loc[is_op.index, "is_operating"] = 1

        # forward-fill
        for col in _FFILL_COLS:
            if col in expanded.columns:
                expanded[col] = expanded[col].ffill()

        # 稼働割合: 非運転日は 0
        if operating_ratio_col in expanded.columns:
            expanded[operating_ratio_col] = expanded[operating_ratio_col].fillna(0.0)

        # plug_id_col を復元
        expanded[plug_id_col] = plug_id
        expanded.index.name = date_col
        expanded = expanded.reset_index()

        frames.append(expanded)

    result = pd.concat(frames, ignore_index=True)

    # gen_no が int になるよう変換（ffill 後は float になることがある）
    if gen_no_col in result.columns:
        result[gen_no_col] = result[gen_no_col].ffill().astype("Int64")

    # operating_hours_since_exchange: plug × gen_no の累積運転時間起点リセット
    origin = result.groupby([plug_id_col, gen_no_col])[runtime_col].transform("min")
    result["operating_hours_since_exchange"] = result[runtime_col] - origin

    # 管理No: plug_id_col を "_" で分割して左側を取得
    if result[plug_id_col].str.contains("_").any():
        result["管理No"] = result[plug_id_col].str.split("_").str[0]
    else:
        raise ValueError(
            f"plug_id_col '{plug_id_col}' does not contain '_': "
            "cannot extract 管理No"
        )

    # TS自動生成列を除外（存在する場合のみ）
    drop_cols = [c for c in TS_AUTO_COLS if c in result.columns]
    if drop_cols:
        result = result.drop(columns=drop_cols)

    return result
```

### Step 1-4: テストが通ることを確認

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_features_ts_dataset.py -v
```

期待出力: `6 passed`

失敗する場合はエラーメッセージを確認して修正する。

### Step 1-5: ruff チェック

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m ruff check src/gene_plug_voltage_predictor/features/ts_dataset.py tests/test_features_ts_dataset.py
```

期待出力: エラーなし（エラーがあれば修正して再確認）

### Step 1-6: コミット

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
git add src/gene_plug_voltage_predictor/features/ts_dataset.py tests/test_features_ts_dataset.py
git commit -m "feat: add build_ts_frame() for DataRobot TS dataset preparation"
```

---

## Task 2: `partitioning.py` / `config.py` の TS 対応

**Files:**
- Modify: `src/gene_plug_voltage_predictor/datarobot/partitioning.py`
- Modify: `src/gene_plug_voltage_predictor/datarobot/config.py`
- Modify: `tests/datarobot/test_partitioning.py`
- Modify: `tests/datarobot/test_config_validation.py`

### Step 2-1: partitioning テストに multiseries テストを追加する

- [ ] `tests/datarobot/test_partitioning.py` の末尾に以下を追加する:

```python
def test_build_partition_datetime_cv_with_multiseries(mocker: MockerFixture) -> None:
    """use_series_id が指定された場合、multiseries_id_columns と forecast_window が設定される。"""
    mock_spec = mocker.MagicMock()
    mocker.patch.object(part_mod.dr, "DatetimePartitioningSpecification", return_value=mock_spec)
    part_mod.build_partition(
        _base_partitioning(
            cv_type="datetime_cv",
            datetime_col="date",
            validation_duration="P7D",
            use_series_id="管理No_プラグNo",
            forecast_window_start=1,
            forecast_window_end=7,
        )
    )
    assert mock_spec.multiseries_id_columns == ["管理No_プラグNo"]
    assert mock_spec.forecast_window_start == 1
    assert mock_spec.forecast_window_end == 7


def test_build_partition_datetime_cv_without_series_id(mocker: MockerFixture) -> None:
    """use_series_id が None の場合、multiseries_id_columns は設定されない。"""
    mock_spec = mocker.MagicMock()
    mocker.patch.object(part_mod.dr, "DatetimePartitioningSpecification", return_value=mock_spec)
    part_mod.build_partition(
        _base_partitioning(
            cv_type="datetime_cv",
            datetime_col="date",
            validation_duration="P7D",
        )
    )
    # use_series_id がないので multiseries_id_columns への代入は呼ばれない
    calls = [str(c) for c in mock_spec.mock_calls]
    assert not any("multiseries_id_columns" in c for c in calls)
```

### Step 2-2: config テストに TS バリデーションテストを追加する

- [ ] `tests/datarobot/test_config_validation.py` の末尾に以下を追加する:

```python
def test_validate_config_accepts_datetime_cv_with_series_id(
    valid_config: dict[str, Any],
) -> None:
    """use_series_id / forecast_window を含む datetime_cv config が受け入れられる。"""
    valid_config["partitioning"]["cv_type"] = "datetime_cv"
    valid_config["partitioning"]["datetime_col"] = "date"
    valid_config["partitioning"]["validation_duration"] = "P7D"
    valid_config["partitioning"]["use_series_id"] = "管理No_プラグNo"
    valid_config["partitioning"]["forecast_window_start"] = 1
    valid_config["partitioning"]["forecast_window_end"] = 7
    config_mod.validate_config(valid_config)  # 例外なし
```

### Step 2-3: テストが失敗することを確認

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/datarobot/test_partitioning.py::test_build_partition_datetime_cv_with_multiseries tests/datarobot/test_config_validation.py::test_validate_config_accepts_datetime_cv_with_series_id -v
```

期待出力: `FAILED`（`AttributeError` または `AssertionError`）

### Step 2-4: `partitioning.py` の `datetime_cv` 分岐を更新する

- [ ] `src/gene_plug_voltage_predictor/datarobot/partitioning.py` の `datetime_cv` ブロック（93〜104行）を以下に置き換える:

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

### Step 2-5: `config.py` の `PartitioningConfig` を更新する

- [ ] `src/gene_plug_voltage_predictor/datarobot/config.py` の `PartitioningConfig` クラス（36〜42行）を以下に置き換える:

```python
class PartitioningConfig(TypedDict, total=False):
    cv_type: CvType
    n_folds: int
    seed: int
    datetime_col: str | None
    validation_duration: str | None
    group_col: str | None
    use_series_id: str | None
    forecast_window_start: int | None
    forecast_window_end: int | None
```

### Step 2-6: テストが通ることを確認

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/datarobot/test_partitioning.py tests/datarobot/test_config_validation.py -v
```

期待出力: 全テスト passed

### Step 2-7: 全テストスイートが通ることを確認

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/ -q --tb=short
```

期待出力: 全テスト passed（既存テストへの影響がないこと）

### Step 2-8: ruff チェック

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m ruff check src/gene_plug_voltage_predictor/datarobot/partitioning.py src/gene_plug_voltage_predictor/datarobot/config.py
```

期待出力: エラーなし

### Step 2-9: コミット

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
git add src/gene_plug_voltage_predictor/datarobot/partitioning.py src/gene_plug_voltage_predictor/datarobot/config.py tests/datarobot/test_partitioning.py tests/datarobot/test_config_validation.py
git commit -m "feat: add multiseries/forecast_window support to datetime_cv partitioning"
```

---

## Task 3: `build_dataset_ts.py` と `datarobot_ep370g_ts.json` の作成

**Files:**
- Create: `src/gene_plug_voltage_predictor/cli/build_dataset_ts.py`
- Create: `config/datarobot_ep370g_ts.json`

### Step 3-1: `build_dataset_ts.py` を作成する

- [ ] `src/gene_plug_voltage_predictor/cli/build_dataset_ts.py` を以下の内容で作成する:

```python
"""DataRobot TS 用データセット組み立て CLI: cleaned CSV → TS 学習用日次 CSV。

処理順:
  aggregate_daily_max_voltage → add_features → build_ts_frame → CSV 出力
  （add_trend_features / add_future_7day_max_target は使わない）
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gene_plug_voltage_predictor.features.features import add_features
from gene_plug_voltage_predictor.features.target import aggregate_daily_max_voltage
from gene_plug_voltage_predictor.features.ts_dataset import build_ts_frame

_logger = logging.getLogger(__name__)


def build_dataset_ts(
    cleaned_df: pd.DataFrame,
    *,
    rated_kw: float = 370.0,
) -> pd.DataFrame:
    """cleaned CSV（30分粒度）から DataRobot TS 用日次データセットを生成する。"""
    daily = aggregate_daily_max_voltage(cleaned_df)
    daily = add_features(daily, cleaned_df, rated_kw=rated_kw)
    daily = build_ts_frame(daily)
    _logger.info("built TS dataset: %d rows, %d columns", len(daily), len(daily.columns))
    return daily


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build DataRobot TS training dataset from cleaned CSV"
    )
    ap.add_argument(
        "--cleaned-csv", required=True, type=Path,
        help="clean.py の出力 CSV（utf-8-sig）",
    )
    ap.add_argument(
        "--out", required=True, type=Path,
        help="DataRobot TS 投入用 CSV の出力先",
    )
    ap.add_argument(
        "--rated-kw", type=float, default=370.0,
        help="定格出力 kW（稼働割合の閾値 = rated_kw × 0.8, default: 370.0）",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cleaned_df = pd.read_csv(args.cleaned_csv, encoding="utf-8-sig")
    dataset = build_dataset_ts(cleaned_df, rated_kw=args.rated_kw)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.out} ({len(dataset)} rows, {len(dataset.columns)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 3-2: `datarobot_ep370g_ts.json` を作成する

- [ ] `config/datarobot_ep370g_ts.json` を以下の内容で作成する:

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

### Step 3-3: `build_dataset_ts.py` が動作することを手動確認

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m gene_plug_voltage_predictor.cli.build_dataset_ts --cleaned-csv outputs/cleaned_ep370g.csv --out outputs/dataset_ep370g_ts.csv --rated-kw 370.0
```

期待出力:
```
[OK] wrote outputs\dataset_ep370g_ts.csv (XXXXX rows, 11 cols)
```

出力された CSV の列を確認:
```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/dataset_ep370g_ts.csv', encoding='utf-8-sig', nrows=1); print(list(df.columns))"
```

期待される列: `['管理No_プラグNo', '管理No', 'date', 'daily_max', 'baseline', 'gen_no', '稼働割合', '累積運転時間', 'voltage_vs_baseline', 'is_operating', 'operating_hours_since_exchange']`

### Step 3-4: 全テストスイートが通ることを確認

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/ -q --tb=short
```

期待出力: 全テスト passed

### Step 3-5: ruff チェック

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m ruff check src/gene_plug_voltage_predictor/cli/build_dataset_ts.py
```

期待出力: エラーなし

### Step 3-6: コミット

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
git add src/gene_plug_voltage_predictor/cli/build_dataset_ts.py config/datarobot_ep370g_ts.json
git commit -m "feat: add build_dataset_ts CLI and datarobot_ep370g_ts.json config"
```

---

## 注意事項

### `PartitioningConfig` の `total=False` と既存フィールドの必須性

Task 2 の Step 2-5 で `TypedDict` に `total=False` を追加している。`total=False` にすると全フィールドが省略可能になり、既存の必須フィールド（`cv_type` 等）も型チェック上は省略可能と見なされる（ただし実行時バリデーションは `validate_config()` が担保する）。

型チェックの厳密性が気になる場合は以下のように基底クラスを分離する方法もある（今回はシンプルさ優先で `total=False` を採用）:

```python
class _PartitioningConfigBase(TypedDict):
    cv_type: CvType  # 必須

class PartitioningConfig(_PartitioningConfigBase, total=False):
    n_folds: int
    use_series_id: str | None
    ...
```

### `gen_no` の型

`groupby` 後に `ffill()` を使うと `gen_no` が `float` になる場合がある（NaN を含むため）。Task 1 の実装で `astype("Int64")`（nullable integer）で変換している。これにより `operating_hours_since_exchange` の `groupby` が正しく動作する。

### `is_operating` フラグと DataRobot TS のターゲット欠損

非運転日の `daily_max` は forward-fill するが、DataRobot TS は `daily_max` が存在するすべての行を学習に使う。`is_operating=0` の行は説明変数として「非運転日状態」を表す特徴量になる。DataRobot TS の target 列に欠損がある場合は自動的にスキップされるが、今回は forward-fill しているためすべての行が有効な target を持つ。これは意図した設計（非運転日の劣化状態を継続値として予測する）。
