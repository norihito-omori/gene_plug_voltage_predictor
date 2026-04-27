# DataRobot 実行規約（Pattern-16 準拠）

## プロジェクト命名

`{name_prefix}_{YYYYMMDD_HHMMSS JST}`

- name_prefix: `gene_plug_voltage_ep370g` / `gene_plug_voltage_ep400g`

## タスク設定

- task_type: `regression`
- metric: null（task_type の既定 RMSE を使う）
- cv_type: `group_cv`（group_col: location）— 機場単位の漏洩防止

## Autopilot

- Phase 0 は `mode: quick` で回して全体感を把握。
- Phase 1 以降で `mode: full_auto` を検討。

## 出力

- `outputs/{name_prefix}_test_pred.csv`（utf-8-sig）
- `metrics/{name_prefix}_model.json`
- `outputs/run_YYYYMMDD_HHMMSS.log`

## 関連

- `30_patterns/pattern-16-datarobot-analysis.md`
- `90_templates/python/datarobot_pipeline/README.md`
