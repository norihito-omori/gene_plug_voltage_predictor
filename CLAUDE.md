# Claude Code ガードレール

## 絶対に守ること

- ADR の「決定」欄は `<!-- 人間が記入 -->` プレースホルダ固定。AI が埋めてはいけない。
- 実験ログの「考察」欄も同様に人間が書く。
- Supersede の判断は人間がする。AI は候補を指摘するまで。
- 数値・ハッシュ・パス・APIレスポンスは AI が埋める。解釈は人間。

## 記録の出力先

- cleaning ログ / experiment ログ / ADR は `E:\projects\contact-center-toolbox\60_domains\ress\gene_plug_voltage_predictor\` 配下に書く。
- `outputs/` / `metrics/` / `logs/` は本リポジトリ内（git 無視）。

## 参照すべき上位ドキュメント

- 設計書: `E:\projects\contact-center-toolbox\docs\superpowers\specs\2026-04-27-gene-plug-voltage-predictor-design.md`
- 実装計画: `E:\projects\contact-center-toolbox\docs\superpowers\plans\2026-04-27-gene-plug-voltage-predictor-phase0.md`
- Pattern-16: `E:\projects\contact-center-toolbox\30_patterns\pattern-16-datarobot-analysis.md`
