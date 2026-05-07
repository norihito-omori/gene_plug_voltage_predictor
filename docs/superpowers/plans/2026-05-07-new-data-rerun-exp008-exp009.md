# 新データ再投入 + exp-008/exp-009 実験 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 昨日（2026-05-06）までのデータに更新された入力 CSV を使い、クリーニングからやり直して DataRobot TS 新規プロジェクト 2 本（回帰 exp-008・二値分類 exp-009）を実行し、2025-12-15 以降のデータ急減問題が解消されたことを BT 全体の評価で確認する。

**Architecture:** ① `cli.clean` 再実行で交換イベントを再検出しつつ cleaned CSV を再構築 → ② `cli.build_dataset_ts` を 2 回実行して回帰用・二値分類用データセットを生成 → ③ `cli.train` を 2 回実行して DataRobot 新規プロジェクトを投入 → ④ 結果を exp-008/exp-009 ログ（toolbox 側）に記録。プラグ交換の注意点として `exchange_events_ep370g.csv` を更新し直す（2026年の新規交換が `operating_hours_since_exchange` に正しく反映されるようにする）。

**Tech Stack:** Python 3.13, pandas, uv, DataRobot Python SDK, cli.clean / cli.build_dataset_ts / cli.train

---

## ファイル構成

| ファイル | 操作 | 説明 |
|---|---|---|
| `outputs/cleaned_ep370g_v3.csv` | 新規作成 | 新データで再構築した cleaned CSV |
| `outputs/exchange_events_ep370g_v3.csv` | 新規作成 | 再検出した交換イベント CSV |
| `outputs/dataset_ep370g_ts_v3.csv` | 新規作成 | 回帰用 TS データセット |
| `outputs/dataset_ep370g_ts_binary_32kv_v3.csv` | 新規作成 | 二値分類用 TS データセット |
| `config/datarobot_ep370g_ts_v3.json` | 新規作成 | exp-008 用 DataRobot config |
| `config/datarobot_ep370g_ts_binary_32kv_v3.json` | 新規作成 | exp-009 用 DataRobot config |
| `metrics/experiment_008.md` | 新規作成 | exp-008 メトリクスサマリ |
| `metrics/experiment_009.md` | 新規作成 | exp-009 メトリクスサマリ |
| `E:/.../toolbox/.../cleaning/2026-05-07-ep370g-v3.md` | 新規作成 | cleaning ログドラフト（toolbox 側） |
| `E:/.../toolbox/.../experiments/exp-008.md` | 新規作成 | 実験ログドラフト（toolbox 側） |
| `E:/.../toolbox/.../experiments/exp-009.md` | 新規作成 | 実験ログドラフト（toolbox 側） |

---

## Task 1: クリーニング再実行（交換イベント再検出込み）

**目的:** 新データ（〜2026-05-06）を投入し、2026年の新規プラグ交換を自動検出して `operating_hours_since_exchange` に正しく反映させる。

**Files:**
- Create: `outputs/cleaned_ep370g_v3.csv`
- Create: `outputs/exchange_events_ep370g_v3.csv`
- Create: `E:\projects\contact-center-toolbox\60_domains\ress\gene_plug_voltage_predictor\cleaning\2026-05-07-ep370g-v3.md`

- [ ] **Step 1: cleaned_ep370g_v3.csv が存在しないことを確認**

```bash
ls outputs/cleaned_ep370g_v3.csv 2>/dev/null && echo "EXISTS" || echo "NOT EXISTS"
```
Expected: `NOT EXISTS`

- [ ] **Step 2: cli.clean を実行（新データ投入・交換イベント再検出）**

```bash
uv run python -m gene_plug_voltage_predictor.cli.clean \
  --config config/cleaning_ep370g.yaml \
  --out outputs/cleaned_ep370g_v3.csv \
  --events-out outputs/exchange_events_ep370g_v3.csv \
  --cleaning-log "E:/projects/contact-center-toolbox/60_domains/ress/gene_plug_voltage_predictor/cleaning/2026-05-07-ep370g-v3.md" \
  --author 大森
```

Expected: `[OK] wrote outputs/cleaned_ep370g_v3.csv (...rows, sha256:...)` が出力される。エラーがある場合は後述のトラブルシュートを参照。

- [ ] **Step 3: 出力行数・日付範囲を確認**

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('outputs/cleaned_ep370g_v3.csv', encoding='utf-8-sig')
print('rows:', len(df))
dt = pd.to_datetime(df['dailygraphpt_ptdatetime'])
print('date range:', dt.min().date(), '~', dt.max().date())
"
```

Expected:
- rows が旧版（8,272,683 行）より多い
- date range の max が `2026-05-06` 付近

- [ ] **Step 4: 交換イベント再検出結果を確認**

```bash
uv run python -c "
import pandas as pd
ev = pd.read_csv('outputs/exchange_events_ep370g_v3.csv')
ev['exchange_date'] = pd.to_datetime(ev['exchange_date'])
print('total events:', len(ev))
print('2026年以降の新規交換イベント:')
print(ev[ev['exchange_date'] >= '2026-01-01'].to_string())
print('2025-12-15以降（旧データ末尾付近）:')
print(ev[ev['exchange_date'] >= '2025-12-15'].sort_values('exchange_date').to_string())
"
```

Expected: 2026年の交換イベントがあれば表示される。なければ「空」で正常（新規交換がなかった場合）。

**トラブルシュート:**
- `ERROR: target_locations includes EXCLUDED_LOCATIONS` → `config/cleaning_ep370g.yaml` の `target_locations` を確認
- `ERROR: CSV not found for location XXXX` → `E:/gene/input/EP370G_orig/XXXX.csv` の存在を確認
- `ValueError: plug_quorum exceeds voltage column count` → 機場のプラグ数が 3 未満の場合。`constants.py` の `EXCHANGE_DETECTION_DEFAULTS` を確認

- [ ] **Step 5: 旧 exchange_events との差分を確認してメモ**

```bash
uv run python -c "
import pandas as pd
old = pd.read_csv('outputs/exchange_events_ep370g.csv')
new = pd.read_csv('outputs/exchange_events_ep370g_v3.csv')
old['exchange_date'] = pd.to_datetime(old['exchange_date'])
new['exchange_date'] = pd.to_datetime(new['exchange_date'])
print('旧:', len(old), '件, 最新:', old['exchange_date'].max().date())
print('新:', len(new), '件, 最新:', new['exchange_date'].max().date())
new_only = new[new['exchange_date'] >= '2026-01-01']
print('2026年以降の新規:', len(new_only), '件')
if len(new_only):
    print(new_only.to_string())
"
```

---

## Task 2: 回帰用 TS データセット生成（exp-008 用）

**Files:**
- Create: `outputs/dataset_ep370g_ts_v3.csv`

- [ ] **Step 1: build_dataset_ts を実行（回帰用、threshold_kv なし）**

```bash
uv run python -m gene_plug_voltage_predictor.cli.build_dataset_ts \
  --cleaned-csv outputs/cleaned_ep370g_v3.csv \
  --out outputs/dataset_ep370g_ts_v3.csv \
  --rated-kw 370.0
```

Expected: `[OK] wrote outputs/dataset_ep370g_ts_v3.csv (... rows, ... cols)`

- [ ] **Step 2: データセットの内容を確認**

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('outputs/dataset_ep370g_ts_v3.csv')
print('shape:', df.shape)
print('columns:', list(df.columns))
print('date range:', df['date'].min(), '~', df['date'].max())
print('unique series:', df.iloc[:, 1].nunique())  # 管理No_プラグNo

# 週次レコード数（旧データで急減していた期間）
df['date'] = pd.to_datetime(df['date'])
df['week'] = df['date'].dt.to_period('W')
weekly = df.groupby('week').size()
print('\\n=== 週次レコード数 (2025-11 ~ 2026-02) ===')
print(weekly['2025-11':'2026-02'].to_string())
"
```

Expected:
- shape が旧版（177,460行 × 11列）より多い（行数増加）
- 2025-12-15週以降の週次レコード数が旧版（60, 42...）から大幅に増加していること

---

## Task 3: 二値分類用 TS データセット生成（exp-009 用）

**Files:**
- Create: `outputs/dataset_ep370g_ts_binary_32kv_v3.csv`

- [ ] **Step 1: build_dataset_ts を実行（二値分類用、threshold_kv=32.0）**

```bash
uv run python -m gene_plug_voltage_predictor.cli.build_dataset_ts \
  --cleaned-csv outputs/cleaned_ep370g_v3.csv \
  --out outputs/dataset_ep370g_ts_binary_32kv_v3.csv \
  --rated-kw 370.0 \
  --threshold-kv 32.0
```

Expected: `[OK] wrote outputs/dataset_ep370g_ts_binary_32kv_v3.csv (... rows, ... cols)` かつ `added exceeds_threshold (>= 32.0kV): positive rate=X.X%` のログ。

- [ ] **Step 2: 陽性率と期間を確認**

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('outputs/dataset_ep370g_ts_binary_32kv_v3.csv')
print('shape:', df.shape)
print('date range:', df['date'].min(), '~', df['date'].max())
pos_rate = df['exceeds_threshold'].mean()
print(f'positive rate: {pos_rate:.3f} ({pos_rate*100:.1f}%)')
print('exceeds_threshold 分布:')
print(df['exceeds_threshold'].value_counts())
"
```

Expected:
- 陽性率が旧版（6.1%）と大きく変わらないこと（±2%程度）
- date の max が 2026-05-06 付近

---

## Task 4: exp-008 用 DataRobot config 作成

**Files:**
- Create: `config/datarobot_ep370g_ts_v3.json`

- [ ] **Step 1: config ファイルを作成**

```bash
cat > config/datarobot_ep370g_ts_v3.json << 'EOF'
{
  "project": {
    "name_prefix": "gene_plug_voltage_ep370g_ts_v3",
    "endpoint": null
  },
  "data": {
    "train_path": "outputs/dataset_ep370g_ts_v3.csv",
    "test_path": "outputs/dataset_ep370g_ts_v3.csv",
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
EOF
```

- [ ] **Step 2: config 検証**

```bash
uv run python -c "
from pathlib import Path
from gene_plug_voltage_predictor.datarobot.config import load_config
cfg = load_config(Path('config/datarobot_ep370g_ts_v3.json'))
print('OK:', cfg['project']['name_prefix'])
"
```

Expected: `OK: gene_plug_voltage_ep370g_ts_v3`

---

## Task 5: exp-009 用 DataRobot config 作成

**Files:**
- Create: `config/datarobot_ep370g_ts_binary_32kv_v3.json`

- [ ] **Step 1: config ファイルを作成**

```bash
cat > config/datarobot_ep370g_ts_binary_32kv_v3.json << 'EOF'
{
  "project": {
    "name_prefix": "gene_plug_voltage_ep370g_ts_binary_32kv_v3",
    "endpoint": null
  },
  "data": {
    "train_path": "outputs/dataset_ep370g_ts_binary_32kv_v3.csv",
    "test_path": null,
    "id_col": "管理No_プラグNo",
    "target_col": "exceeds_threshold"
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
EOF
```

- [ ] **Step 2: config 検証**

```bash
uv run python -c "
from pathlib import Path
from gene_plug_voltage_predictor.datarobot.config import load_config
cfg = load_config(Path('config/datarobot_ep370g_ts_binary_32kv_v3.json'))
print('OK:', cfg['project']['name_prefix'])
"
```

Expected: `OK: gene_plug_voltage_ep370g_ts_binary_32kv_v3`

---

## Task 6: exp-008 DataRobot Autopilot 実行（回帰）

**Files:**
- Create: `metrics/experiment_008.md`（cli.train が自動生成）
- Create: `outputs/gene_plug_voltage_ep370g_ts_v3_test_pred.csv`（cli.train が自動生成）
- Create: `E:\projects\contact-center-toolbox\60_domains\ress\gene_plug_voltage_predictor\experiments\exp-008.md`

**注意:** DataRobot Autopilot は数十分〜数時間かかる。`max_wait_minutes: 360` で最大 6 時間待つ設定。

- [ ] **Step 1: .env の DATAROBOT_API_TOKEN を確認**

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
token = os.getenv('DATAROBOT_API_TOKEN', '')
print('token set:', bool(token), '(length:', len(token), ')')
"
```

Expected: `token set: True`

- [ ] **Step 2: cli.train を実行（回帰 exp-008）**

```bash
uv run python -m gene_plug_voltage_predictor.cli.train \
  --config config/datarobot_ep370g_ts_v3.json \
  --train outputs/dataset_ep370g_ts_v3.csv \
  --experiment-log "E:/projects/contact-center-toolbox/60_domains/ress/gene_plug_voltage_predictor/experiments/exp-008.md" \
  --model-type EP370G \
  --topic "DataRobot TS 回帰 新データ v3 (exp-008)" \
  --related-cleaning "2026-05-07-ep370g-v3" \
  --related-adrs "decision-001,decision-002,decision-003,decision-009,decision-010,decision-012,decision-014" \
  --experiment-id "exp-008"
```

Expected（数十分〜数時間後）:
- `[OK] best model: <model_id>` が出力される
- `outputs/gene_plug_voltage_ep370g_ts_v3_test_pred.csv` が生成される
- `metrics/experiment_008.md` が生成される
- toolbox 側の `exp-008.md` にドラフトが書き出される

- [ ] **Step 3: exp-008 結果の数値を確認**

```bash
cat metrics/experiment_008.md
```

確認ポイント:
- RMSE が exp-005（0.337）より改善しているか
- バックテスト別メトリクス（BT2〜BT4）が正常値になっているか

---

## Task 7: exp-009 DataRobot Autopilot 実行（二値分類）

**Files:**
- Create: `metrics/experiment_009.md`（cli.train が自動生成）
- Create: `E:\projects\contact-center-toolbox\60_domains\ress\gene_plug_voltage_predictor\experiments\exp-009.md`

**注意:** Task 6 と並行実行可能（DataRobot のプロジェクトは独立）。ただし DataRobot の worker 上限に注意。

- [ ] **Step 1: cli.train を実行（二値分類 exp-009）**

```bash
uv run python -m gene_plug_voltage_predictor.cli.train \
  --config config/datarobot_ep370g_ts_binary_32kv_v3.json \
  --train outputs/dataset_ep370g_ts_binary_32kv_v3.csv \
  --experiment-log "E:/projects/contact-center-toolbox/60_domains/ress/gene_plug_voltage_predictor/experiments/exp-009.md" \
  --model-type EP370G \
  --topic "DataRobot TS 二値分類 32kV 新データ v3 (exp-009)" \
  --related-cleaning "2026-05-07-ep370g-v3" \
  --related-adrs "decision-001,decision-002,decision-003,decision-009,decision-010,decision-012,decision-014" \
  --experiment-id "exp-009"
```

Expected（数十分〜数時間後）:
- `[OK] best model: <model_id>` が出力される
- `metrics/experiment_009.md` が生成される
- toolbox 側の `exp-009.md` にドラフトが書き出される

- [ ] **Step 2: exp-009 結果の数値を確認**

```bash
cat metrics/experiment_009.md
```

確認ポイント:
- BT1 の LogLoss が exp-007（0.253）より改善しているか
- BT2〜BT4 の LogLoss が異常値（6〜12）から正常化しているか（目安: < 1.0）
- AUC が BT1〜BT4 全体で安定して高い値か

---

## Task 8: git コミット（config ファイル）

**注意:** `outputs/` と `metrics/` は `.gitignore` で管理外。コミット対象は config ファイルと toolbox 側ドキュメントのみ。

- [ ] **Step 1: git status で対象を確認**

```bash
git status
```

Expected: `config/datarobot_ep370g_ts_v3.json` と `config/datarobot_ep370g_ts_binary_32kv_v3.json` が untracked として表示される。

- [ ] **Step 2: コミット**

```bash
git add config/datarobot_ep370g_ts_v3.json config/datarobot_ep370g_ts_binary_32kv_v3.json
git commit -m "$(cat <<'EOF'
feat: add DataRobot config v3 for exp-008/exp-009 (new data up to 2026-05-06)

New configs for re-experiment with refreshed data to resolve the
2025-12-15 data cliff issue identified in exp-005/exp-007.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: 結果比較サマリ作成

**Files:**
- Create: `metrics/comparison_exp005_007_vs_008_009.md`

- [ ] **Step 1: 比較表を作成**

Task 6・7 の結果が出たら、以下の比較表を `metrics/comparison_exp005_007_vs_008_009.md` に書く。数値は実際の結果に置き換える。

```markdown
# exp-005/007 vs exp-008/009 比較

| 実験 | モデル | BT0 | BT1 | BT2 | BT3 | BT4 | 備考 |
|---|---|---|---|---|---|---|---|
| exp-005 (回帰) | RMSE | — | — | — | — | — | 旧データ (〜2026-01-20) |
| exp-008 (回帰) | RMSE | — | — | — | — | — | 新データ (〜2026-05-06) |
| exp-007 (二値) | LogLoss | 0.000* | 0.253 | 6.706 | 12.218 | 2.501 | 旧データ、*BT0は不信頼 |
| exp-009 (二値) | LogLoss | — | — | — | — | — | 新データ |

## 評価ポイント

1. BT2〜BT4 の LogLoss が正常化したか（目標: < 1.0）
2. 全 BT で AUC が安定したか（目標: > 0.80）
3. RMSE が exp-005（0.337）より改善したか
4. `operating_hours_since_exchange` の特徴量重要度に変化があるか
```

- [ ] **Step 2: コミット**

```bash
git add metrics/comparison_exp005_007_vs_008_009.md
git commit -m "$(cat <<'EOF'
docs: add comparison table template for exp-008/exp-009 vs exp-005/exp-007

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## 注意事項

### プラグ交換の扱いについて

- `cli.clean` の `detect_exchange_events` が `exchange_events_ep370g_v3.csv` を自動生成する
- この新イベントが `assign_generation` ステップ経由で `gen_no` に反映され、`operating_hours_since_exchange` が正しく再計算される
- Task 1 Step 4 で 2026 年の新規交換イベントが検出された場合、その機場の `operating_hours_since_exchange` が旧データと大きく異なる可能性がある → exp-008/009 の特徴量重要度で確認

### DataRobot 実行順序

- Task 6（exp-008）と Task 7（exp-009）は独立したプロジェクトなので並行実行可能
- ただし DataRobot の同時 worker 数の上限に注意。上限に達すると片方がキューイングされる
- `max_wait_minutes: 360` を超えた場合は CLI がタイムアウトエラーを出す。その場合は DataRobot UI で直接進捗を確認する

### 実験ログの確定

- toolbox 側の `exp-008.md` / `exp-009.md` の「目的」「考察」欄は Claude Code が書かず、`<!-- 人間が記入 -->` プレースホルダのまま出力される（CLAUDE.md ガードレール）
- 数値・Project ID・ハッシュは cli.train が自動記入
- 大森氏がドラフトを確認して `status: draft` → `confirmed` に更新する
