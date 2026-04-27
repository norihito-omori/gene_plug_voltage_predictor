from __future__ import annotations

from datetime import date

from gene_plug_voltage_predictor.datarobot.experiment_reporter import (
    ExperimentContext,
    render_experiment_log,
)


def test_render_experiment_log_includes_frontmatter_and_sections() -> None:
    ctx = ExperimentContext(
        experiment_id="exp-2026-05-07-01",
        run_date=date(2026, 5, 7),
        author="Omori",
        model_type="EP370G",
        topic="ep370g-baseline",
        train_csv="outputs/ep370g_train_20260501.csv",
        train_csv_hash="sha256:aaa",
        related_cleaning="clean-2026-05-01-ep370g-baseline",
        related_adrs=["decision-001", "decision-007"],
        datarobot_project_id="abc123",
        datarobot_project_url="https://app.datarobot.com/projects/abc123",
        best_model_id="model-xyz",
        best_blueprint="eXtreme Gradient Boosting",
        metric_name="RMSE",
        metric_value=12.34,
        test_pred_csv="outputs/ep370g_test_pred.csv",
        test_pred_hash="sha256:bbb",
    )
    md = render_experiment_log(ctx)
    assert "id: exp-2026-05-07-01" in md
    assert "model_type: EP370G" in md
    assert "## 1." in md
    assert "## 2." in md
    assert "sha256:aaa" in md
    assert "RMSE" in md
    assert "12.34" in md
    assert "abc123" in md
    assert "## 4." in md
