# build_dataset CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `cli/build_dataset.py` を実装し、cleaning 済み CSV から DataRobot 学習用日次データセット CSV を一括生成する。

**Architecture:** `build_dataset(cleaned_df, ...)` の純粋関数にコアロジックを集約し、CLI の `main()` はその薄いラッパーとする。コアは `aggregate_daily_max_voltage → add_features → add_future_7day_max_target → NaN 除外` のパイプライン。DataRobot config も合わせて更新する。

**Tech Stack:** Python 3.13, pandas, argparse, pytest

---

## ファイル構成

| 操作 | パス | 役割 |
|------|------|------|
| 新規作成 | `src/gene_plug_voltage_predictor/cli/build_dataset.py` | `build_dataset()` 関数 + `main()` CLI |
| 新規作成 | `tests/test_cli_build_dataset.py` | 全テスト（4 ケース） |
| 更新 | `config/datarobot_ep370g.json` | `target_col`, `id_col`, `group_col`, `train_path` を実際の列名に修正 |

---

## 背景知識（ゼロコンテキスト向け）

### cleaned_df のスキーマ（`clean.py` の出力）

30 分粒度・縦持ちで以下の列を持つ:

```
管理No, dailygraphpt_ptdatetime, target_id, 発電機電力, 累積運転時間,
target_output, mcnkind_id, 要求電圧, プラグNo, 管理No_プラグNo, gen_no, baseline
```

### 使う関数（既存実装）

```python
# features/target.py
aggregate_daily_max_voltage(df) -> DataFrame  # columns: 管理No_プラグNo, date, daily_max
add_future_7day_max_target(daily_df, *, horizon=7) -> DataFrame  # + future_7d_max

# features/features.py
add_features(daily_df, cleaned_df, *, rated_kw=370.0, ...) -> DataFrame
# + baseline, gen_no, voltage_vs_baseline, daily_max_lag_1/3/7, 稼働割合, 累積運転時間
```

### 出力 CSV のスキーマ（NaN 除外後）

`管理No_プラグNo, date, daily_max, baseline, gen_no, voltage_vs_baseline,
daily_max_lag_1, daily_max_lag_3, daily_max_lag_7, 稼働割合, 累積運転時間, future_7d_max`

---

## Task 1: `build_dataset()` 関数の実装とテスト

**Files:**
- Create: `src/gene_plug_voltage_predictor/cli/build_dataset.py`
- Create: `tests/test_cli_build_dataset.py`

### Step 1-1: 失敗するテストを書く

- [ ] `tests/test_cli_build_dataset.py` を以下の内容で作成する:

```python
"""tests/test_cli_build_dataset.py"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

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
    # 1 行だけ baseline を NaN に書き換える
    cleaned_df.loc[cleaned_df["管理No_プラグNo"] == "5630_1", "baseline"] = float("nan")
    result = build_dataset(cleaned_df, rated_kw=370.0, horizon=7)
    assert result["baseline"].notna().all(), "baseline must have no NaN"
```

### Step 1-2: テストが失敗することを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_cli_build_dataset.py -v
```

期待出力: `ImportError` または `ModuleNotFoundError`

### Step 1-3: `build_dataset.py` を実装する

- [ ] `src/gene_plug_voltage_predictor/cli/build_dataset.py` を以下の内容で作成する:

```python
"""データセット組み立て CLI: cleaned CSV → DataRobot 学習用日次 CSV。

処理順:
  aggregate_daily_max_voltage → add_features → add_future_7day_max_target
  → NaN 除外（future_{horizon}d_max / baseline）→ CSV 出力
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

_logger = logging.getLogger(__name__)


def build_dataset(
    cleaned_df: pd.DataFrame,
    *,
    rated_kw: float = 370.0,
    horizon: int = 7,
) -> pd.DataFrame:
    """cleaned_df（30分粒度）から学習用日次データセットを生成して返す。

    NaN を持つターゲット行（末尾 horizon 日）と baseline=NaN 行を除外する。
    """
    daily = aggregate_daily_max_voltage(cleaned_df)
    daily = add_features(daily, cleaned_df, rated_kw=rated_kw)
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

### Step 1-4: テストが通ることを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_cli_build_dataset.py::test_build_dataset_produces_expected_columns tests/test_cli_build_dataset.py::test_build_dataset_drops_nan_target_rows tests/test_cli_build_dataset.py::test_build_dataset_drops_nan_baseline_rows -v
```

期待出力: `3 passed`

もし `test_build_dataset_drops_nan_baseline_rows` が失敗する場合は、`cleaned_df` 側の `baseline` 列を全行 NaN にしていることが原因の可能性がある。`add_features` が `cleaned_df` から `baseline` を join するため、cleaned_df の全行を NaN にしないとテストが通らない場合は以下のように修正:

```python
# cleaned_df の 5630_1 に属する全行の baseline を NaN にする
cleaned_df.loc[cleaned_df["管理No_プラグNo"] == "5630_1", "baseline"] = float("nan")
```

上記はすでにそうなっているので問題ないはず。

### Step 1-5: コミット

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
git add src/gene_plug_voltage_predictor/cli/build_dataset.py tests/test_cli_build_dataset.py
git commit -m "feat: add build_dataset CLI (cleaned CSV → daily training dataset)"
```

---

## Task 2: `--out` 親ディレクトリ自動作成テスト + DataRobot config 更新

**Files:**
- Modify: `tests/test_cli_build_dataset.py`
- Modify: `config/datarobot_ep370g.json`

### Step 2-1: ディレクトリ自動作成テストを追加する

- [ ] `tests/test_cli_build_dataset.py` の末尾に以下を追記する。また冒頭の import に `argparse` と `main` を追加する:

ファイル冒頭の import を以下に更新:

```python
from gene_plug_voltage_predictor.cli.build_dataset import build_dataset, main
```

末尾に追記:

```python
def test_build_dataset_creates_output_dir(tmp_path: Path) -> None:
    """--out の親ディレクトリが存在しなくても自動作成される。"""
    cleaned_df = _make_cleaned_df()
    out_dir = tmp_path / "nested" / "subdir"
    out_file = out_dir / "dataset.csv"
    # out_dir はまだ存在しない
    assert not out_dir.exists()

    cleaned_csv = tmp_path / "cleaned.csv"
    cleaned_df.to_csv(cleaned_csv, index=False, encoding="utf-8-sig")

    import sys
    orig_argv = sys.argv
    sys.argv = [
        "build_dataset",
        "--cleaned-csv", str(cleaned_csv),
        "--out", str(out_file),
    ]
    try:
        ret = main()
    finally:
        sys.argv = orig_argv

    assert ret == 0
    assert out_file.exists()
    result = pd.read_csv(out_file, encoding="utf-8-sig")
    assert len(result) > 0
```

### Step 2-2: テストが通ることを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/test_cli_build_dataset.py -v
```

期待出力: `4 passed`

### Step 2-3: DataRobot config を更新する

- [ ] `config/datarobot_ep370g.json` を読み込み、以下の通り更新する:

**更新前:**
```json
"data": {
    "train_path": "outputs/ep370g_train_20260501.csv",
    "test_path": "outputs/ep370g_train_20260501.csv",
    "id_col": "location",
    "target_col": "plug_voltage"
},
...
"partitioning": {
    "cv_type": "group_cv",
    "n_folds": 5,
    "seed": 42,
    "datetime_col": null,
    "validation_duration": null,
    "group_col": "location"
},
```

**更新後:**
```json
"data": {
    "train_path": "outputs/dataset_ep370g.csv",
    "test_path": "outputs/dataset_ep370g.csv",
    "id_col": "管理No_プラグNo",
    "target_col": "future_7d_max"
},
...
"partitioning": {
    "cv_type": "group_cv",
    "n_folds": 5,
    "seed": 42,
    "datetime_col": null,
    "validation_duration": null,
    "group_col": "管理No_プラグNo"
},
```

### Step 2-4: 全テストスイートが通ることを確認する

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m pytest tests/ -v --tb=short -q
```

期待出力: `114 passed`（既存 110 + 新規 4）

### Step 2-5: ruff チェック

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
python -m ruff check src/gene_plug_voltage_predictor/cli/build_dataset.py tests/test_cli_build_dataset.py
```

期待出力: エラーなし。

### Step 2-6: コミット

- [ ] 以下を実行:

```bash
cd E:/projects/gene_plug_voltage_predictor
git add tests/test_cli_build_dataset.py config/datarobot_ep370g.json
git commit -m "test: add output dir creation test; fix: update datarobot config column names"
```

---

## 注意事項

### `sys.argv` を直接操作するテストについて

`test_build_dataset_creates_output_dir` は `sys.argv` を一時的に書き換えて `main()` を直接呼ぶ方法を使っている。これは `argparse` を使う CLI の最もシンプルなテスト手法。`monkeypatch` fixture（pytest）を使う方法もあるが、`tmp_path` だけで完結するためここでは不要。

`sys.argv` を書き換えた後は `finally` で必ず元に戻すことを確認済み。

### DataRobot config の `group_col` について

spec では `group_col` を `管理No_プラグNo`（プラグ単位）に設定している。これにより DataRobot の Group CV では**プラグ単位**でフォールドが分割される。機場単位（`管理No`）にしたい場合は別途 ADR で変更すること。
