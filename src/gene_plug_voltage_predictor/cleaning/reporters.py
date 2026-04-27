"""クリーニング履歴から Markdown ドラフトを生成する。"""

from __future__ import annotations

from datetime import date

from .pipeline import CleaningHistory

_TEMPLATE = """\
---
id: clean-{run_date}-{dataset_name}
date: {run_date}
author: {author}
model_type: {model_type}
dataset_name: {dataset_name}
input_rows: {input_rows}
output_rows: {output_rows}
output_path: {output_path}
output_hash: {output_hash}
status: draft
related_adrs: [{related_adrs}]
---

# {dataset_name} クリーニングログ

## 1. 入力データ

- 行数: {input_rows}

## 2. クリーニング手順(実行順)

{steps_block}

## 3. 出力データ

- パス: `{output_path}`
- 行数: {output_rows}
- SHA256: `{output_hash}`

## 4. 検証

<!-- 人間が記入 -->

## 5. 備考・積み残し

<!-- 人間が記入 -->
"""

_STEP_TEMPLATE = """\
- Step {idx}: {name}
  - 根拠: {adr}
  - 備考: {note}
  - 除外: {excluded_rows} 行
  - 残: {rows_after} 行
"""


def render_cleaning_log(
    *,
    history: CleaningHistory,
    dataset_name: str,
    model_type: str,
    author: str,
    run_date: date,
    output_path: str,
    output_hash: str,
    related_adrs: list[str],
) -> str:
    steps_block = "\n".join(
        _STEP_TEMPLATE.format(
            idx=i + 1,
            name=s.name,
            adr=s.adr,
            note=s.note,
            excluded_rows=s.excluded_rows,
            rows_after=s.rows_after,
        )
        for i, s in enumerate(history.steps)
    )
    return _TEMPLATE.format(
        run_date=run_date.isoformat(),
        dataset_name=dataset_name,
        author=author,
        model_type=model_type,
        input_rows=history.input_rows,
        output_rows=history.output_rows,
        output_path=output_path,
        output_hash=output_hash,
        related_adrs=", ".join(related_adrs),
        steps_block=steps_block,
    )
