"""pipeline.run() のテスト. DataRobot 側は全て mock で差し替える."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pytest_mock import MockerFixture

from gene_plug_voltage_predictor.datarobot import pipeline as pipeline_mod


def _make_config(tmp_path: Path, *, with_test_path: bool) -> dict[str, Any]:
    """共通のテスト用 config. test_path の有無だけを切り替える."""
    outputs_dir = tmp_path / "outputs"
    metrics_dir = tmp_path / "metrics"
    data: dict[str, Any] = {
        "train_path": str(tmp_path / "train.csv"),
        "id_col": "id",
        "target_col": "y",
    }
    if with_test_path:
        data["test_path"] = str(tmp_path / "test.csv")
    return {
        "output": {
            "outputs_dir": str(outputs_dir),
            "metrics_dir": str(metrics_dir),
        },
        "data": data,
        "project": {"name_prefix": "gp"},
        "task": {"task_type": "regression"},
        "partitioning": {"cv_type": "stratified", "holdout_pct": 20, "cv_folds": 5},
    }


def _patch_pipeline_deps(mocker: MockerFixture) -> dict[str, Any]:
    """pipeline モジュールの import バインディングを全て差し替える."""
    best_model = SimpleNamespace(
        id="m-1",
        model_type="XGB",
        metrics={"RMSE": {"crossValidation": 12.5, "validation": 13.0}},
    )
    project = SimpleNamespace(id="proj-123")

    mocks = {
        "setup_logging": mocker.patch(
            "gene_plug_voltage_predictor.datarobot.pipeline.setup_logging"
        ),
        "setup_client": mocker.patch(
            "gene_plug_voltage_predictor.datarobot.pipeline.setup_client"
        ),
        "prepare_train_dataset": mocker.patch(
            "gene_plug_voltage_predictor.datarobot.pipeline.prepare_train_dataset",
            return_value=(Path("train.csv"), None),
        ),
        "create_project": mocker.patch(
            "gene_plug_voltage_predictor.datarobot.pipeline.create_project",
            return_value=project,
        ),
        "build_partition": mocker.patch(
            "gene_plug_voltage_predictor.datarobot.pipeline.build_partition"
        ),
        "configure_and_start": mocker.patch(
            "gene_plug_voltage_predictor.datarobot.pipeline.configure_and_start"
        ),
        "wait_for_autopilot": mocker.patch(
            "gene_plug_voltage_predictor.datarobot.pipeline.wait_for_autopilot"
        ),
        "get_best_model": mocker.patch(
            "gene_plug_voltage_predictor.datarobot.pipeline.get_best_model",
            return_value=(best_model, "RMSE"),
        ),
        "save_model_metrics": mocker.patch(
            "gene_plug_voltage_predictor.datarobot.pipeline.save_model_metrics"
        ),
        "predict_dataset": mocker.patch(
            "gene_plug_voltage_predictor.datarobot.pipeline.predict_dataset"
        ),
    }
    return mocks


def test_run_returns_run_result_with_test_pred_path(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    cfg = _make_config(tmp_path, with_test_path=True)
    mocks = _patch_pipeline_deps(mocker)

    result = pipeline_mod.run(cfg, train_path=tmp_path / "train.csv")

    assert result.project_id == "proj-123"
    assert result.metric_name == "RMSE"
    assert result.metric_value == pytest.approx(12.5)
    assert result.test_pred_csv is not None
    assert str(result.test_pred_csv).endswith("_test_pred.csv")
    mocks["predict_dataset"].assert_called_once()


def test_run_returns_none_test_pred_csv_when_no_test_path(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    cfg = _make_config(tmp_path, with_test_path=False)
    mocks = _patch_pipeline_deps(mocker)

    result = pipeline_mod.run(cfg, train_path=tmp_path / "train.csv")

    assert result.test_pred_csv is None
    mocks["predict_dataset"].assert_not_called()
