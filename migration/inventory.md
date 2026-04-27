# 既存資産棚卸し

作成日: 2026-04-29
作成者: Claude Code（ドラフト）/ 大森（確定）

## EP-VoltPredictor（`D:\code_commit\EP-VoltPredictor\`）

### 移植／破棄判定

| ファイル | 判定 | 移植先 | 備考 |
|---|---|---|---|
| `ep_voltpredictor/voltage_data_processor4.py`（2026-02-19 10:38, 4.9KB） | 移植 | `src/.../cleaning/steps.py` の元ネタ | 最新版。リファクタ＋型付け＋テスト化 |
| `ep_voltpredictor/voltage_data_processor{,2,2 copy,3}.py` | 破棄 | — | 旧版。4系で統合済み |
| `ep_voltpredictor/make_combine_pred4.py`（2026-02-19 11:58, 1.9KB） | 移植 | `src/.../features/derive.py` の元ネタ | 最新版。派生ウィンドウロジック |
| `ep_voltpredictor/make_combine{,_pred,_pred2,_pred3}.py` | 破棄 | — | 旧版。pred4 で統合済み |
| `ep_voltpredictor/main{,2,2_future,3,3_2,3_future,3_training,4}.py` | 破棄 | — | DB 接続＋処理本体。本プロジェクトは CSV 入力 + CLI 分離設計のため踏襲しない |
| `ep_voltpredictor/make_confusion_matrix{,2,3,4,5,6}.py` | 破棄 | — | 回帰タスクに不要（分類用） |
| `ep_voltpredictor/fix9221.py` | 参照 | — | 機場 9221 プラグ 4 除外の根拠スクリプト。L-03 根拠として ADR に引用 |
| `ep_voltpredictor/add_cumrunhour.py` | 参照 | — | 累積運転時間算出ロジック。M3 で参照 |
| `ep_voltpredictor/combine_csv_files{,2}.py` | 参照 | — | CSV 結合。本プロジェクトは機場単位ファイルなので踏襲しない |
| `ep_voltpredictor/csv_split.py` | 破棄 | — | splitter。DataRobot 側で partition するため不要 |
| `ep_voltpredictor/datarobot2*.py` | 破棄 | — | 旧 DataRobot 連携。Pattern-16 テンプレに置き換わった |
| `ep_voltpredictor/merged_output*.csv` | 参照 | — | 中間データ。必要ならサンプルとして参照のみ |
| `ep_voltpredictor/ep370g_1day.py` / `ep400g_1day.py` / `ep400g_1day2.py` | 参照 | — | 1 日集約。L-07（時刻起点）の根拠を探すのに参照 |
| `ep_voltpredictor/trial.py` / `trial_matrix.py` / `tmp*.py` | 破棄 | — | 試作物 |
| `ep_voltpredictor/plot_*.py` / `generate_voltage_scatter_plots.py` / `make_heatmap*.py` / `hikaku*.py` / `voltage_analysis.py` | 参照 | — | 可視化。本プロジェクト Phase 0 スコープ外 |
| `ep_voltpredictor/EP370G*/` / `ep_voltpredictor/EP400G*/` / `ep_voltpredictor/EP370_400/` | 参照 | — | サブディレクトリ内の成果物。必要に応じて参照 |
| `doc/要求電圧データ処理.md` | 参照 | — | ADR 根拠として引用 |
| `doc/20240509.md` | 参照 | — | 同上 |
| `doc/要求電圧.jpg` | 参照 | — | 必要なら knowledge SoT にコピー |
| `doc/DataRobot20240509.docx` | 参照 | — | DataRobot 側の過去判断。M6 の参照材料 |
| `doc/要求電圧データ処理.docx` | 参照 | — | md と同内容想定 |
| `doc/20260115会議資料.xlsx` | 参照 | — | 中間報告で参照 |

## generator-voltage-alert-poc（`D:\code_commit\generator-voltage-alert-poc\`）

| ファイル | 判定 | 移植先 | 備考 |
|---|---|---|---|
| `cloud-mail-sender/data_processor.py` | 参考 | `src/.../cleaning/steps.py` の参考 | Phase 0 では直接移植しない |
| `aws-pipeline/` | Phase 2 | — | 通知配送路。Phase 0 対象外 |
| `docs/poc-plan.md` | 参照 | — | Phase 2 時の参照材料 |

## 移植済みファイル（`migration/imported/`）

`migration/imported/` は `.gitignore` 済み。バージョン管理せず、M3 で `src/` へ昇格する際の参照元。

| ファイル | 元パス | コピー日 |
|---|---|---|
| `voltage_data_processor4.py` | `D:\code_commit\EP-VoltPredictor\ep_voltpredictor\voltage_data_processor4.py` | 2026-04-29 |
| `make_combine_pred4.py` | `D:\code_commit\EP-VoltPredictor\ep_voltpredictor\make_combine_pred4.py` | 2026-04-29 |

## 次アクション

1. M3（2026-04-30）で `migration/imported/` の最新版を参照しながら `cleaning/steps.py` を実装（現状 3 ステップ実装済み、残り L-05〜L-07 系を検討）
2. 派生ウィンドウ（L-06）は `features/derive.py` として別モジュールへ
3. 可視化スクリプトは Phase 0 スコープ外。Phase 1 以降で検討
