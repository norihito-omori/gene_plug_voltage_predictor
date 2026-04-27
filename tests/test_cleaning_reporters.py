from __future__ import annotations

from datetime import date
from pathlib import Path

from gene_plug_voltage_predictor.cleaning.pipeline import (
    CleaningHistory,
    StepHistory,
)
from gene_plug_voltage_predictor.cleaning.reporters import render_cleaning_log


def test_render_cleaning_log_includes_frontmatter_and_steps(tmp_path: Path) -> None:
    history = CleaningHistory(
        input_rows=100,
        output_rows=50,
        steps=[
            StepHistory(
                name="exclude_locations",
                adr="decision-004",
                excluded_rows=30,
                rows_before=100,
                rows_after=70,
                note="excluded=[8950]",
            ),
            StepHistory(
                name="filter_cumulative_runtime",
                adr="decision-005",
                excluded_rows=20,
                rows_before=70,
                rows_after=50,
                note="keep rows where cum_runtime_h >= 1000",
            ),
        ],
    )
    md = render_cleaning_log(
        history=history,
        dataset_name="ep370g-baseline",
        model_type="EP370G",
        author="大森",
        run_date=date(2026, 5, 1),
        output_path="outputs/ep370g_train_20260501.csv",
        output_hash="sha256:abc123",
        related_adrs=["decision-001", "decision-004", "decision-005"],
    )
    assert "id: clean-2026-05-01-ep370g-baseline" in md
    assert "model_type: EP370G" in md
    assert "Step 1: exclude_locations" in md
    assert "decision-004" in md
    assert "除外: 30 行" in md
    assert "残: 70 行" in md
    assert "sha256:abc123" in md
