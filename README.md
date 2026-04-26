# gene_plug_voltage_predictor

EP370G / EP400G のプラグ電圧予測モデル（Phase 0）。

## 知見・判断の SoT

本リポジトリは実装のみを置く。判断記録・実験ログ・ADR は toolbox 側に集約する。

- 知見 SoT: `E:\projects\contact-center-toolbox\60_domains\ress\gene_plug_voltage_predictor\`
- 設計書: `docs/superpowers/specs/2026-04-27-gene-plug-voltage-predictor-design.md`

## セットアップ

```bash
uv sync --dev
cp .env.example .env
# .env を編集して DATAROBOT_API_TOKEN を設定
```

## 主なコマンド（Phase 0）

```bash
# クリーニング（M4）
uv run python -m gene_plug_voltage_predictor.cli.clean --config config/cleaning_ep370g.yaml

# 学習（M6）
uv run python -m gene_plug_voltage_predictor.cli.train --config config/datarobot_ep370g.json
```

## テスト

```bash
uv run pytest
```
