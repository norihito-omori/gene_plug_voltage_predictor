"""DataRobot の実行結果から experiment ログ Markdown ドラフトを生成する。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from string import Template


@dataclass(frozen=True)
class ExperimentContext:
    """experiment ログ生成に必要な情報。すべて AI が埋め、人間は「目的」「考察」を書く。"""

    experiment_id: str
    run_date: date
    author: str
    model_type: str
    topic: str
    train_csv: str
    train_csv_hash: str
    related_cleaning: str
    related_adrs: list[str]
    datarobot_project_id: str
    datarobot_project_url: str
    best_model_id: str
    best_blueprint: str
    metric_name: str
    metric_value: float
    test_pred_csv: str
    test_pred_hash: str


_TEMPLATE = Template("""\
---
id: $id
date: $date
author: $author
model_type: $model_type
topic: $topic
status: draft
draft_by: claude-code
confirmed_by: ""
confirmed_at: ""
related_cleaning: $related_cleaning
related_adrs: [$related_adrs]
related_experiments: []
---

# $topic

## 1. 目的(Why)

<!-- 人間が記入 -->

## 2. データ・条件(Setup)

- 学習用 CSV: `$train_csv`
  - SHA256: `$train_csv_hash`
- 関連クリーニング: $related_cleaning
- DataRobot Project ID: `$project_id`
- DataRobot Project URL: $project_url

## 3. 結果(Results)

- 採用モデル ID: `$best_model_id`
- Blueprint: $best_blueprint
- $metric_name: $metric_value
- テスト予測 CSV: `$test_pred_csv`
  - SHA256: `$test_pred_hash`

## 4. 考察(Discussion)

<!-- 人間が記入 -->

## 5. 次アクション(Next)

<!-- Claude Code が候補を列挙、人間が確定 -->
- [ ] (例) 上位 3 モデルのメトリクスを比較して採用モデルを確定
- [ ] (例) 特徴量重要度を確認して説明変数を追加／削除
""")


def render_experiment_log(ctx: ExperimentContext) -> str:
    """ExperimentContext を Markdown ドラフトに変換する。

    `string.Template` を使うことで、任意の文字列フィールドに波括弧が
    含まれていても安全に置換できる（cleaning/reporters.py と同じ方針）。
    """
    return _TEMPLATE.substitute(
        id=ctx.experiment_id,
        date=ctx.run_date.isoformat(),
        author=ctx.author,
        model_type=ctx.model_type,
        topic=ctx.topic,
        train_csv=ctx.train_csv,
        train_csv_hash=ctx.train_csv_hash,
        related_cleaning=ctx.related_cleaning,
        related_adrs=", ".join(ctx.related_adrs),
        project_id=ctx.datarobot_project_id,
        project_url=ctx.datarobot_project_url,
        best_model_id=ctx.best_model_id,
        best_blueprint=ctx.best_blueprint,
        metric_name=ctx.metric_name,
        metric_value=ctx.metric_value,
        test_pred_csv=ctx.test_pred_csv,
        test_pred_hash=ctx.test_pred_hash,
    )
