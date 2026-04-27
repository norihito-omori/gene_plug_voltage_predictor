"""models.py のテスト."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gene_plug_voltage_predictor.datarobot import models as models_mod


def _make_model(
    model_type: str,
    model_id: str,
    cv: dict[str, float | None],
    val: dict[str, float | None] | None = None,
) -> Any:
    return SimpleNamespace(
        id=model_id,
        model_type=model_type,
        metrics={
            key: {"crossValidation": cv.get(key), "validation": (val or {}).get(key)}
            for key in {*cv, *(val or {})}
        },
    )


def test_get_best_model_binary_prefers_pr_auc() -> None:
    project = SimpleNamespace(
        get_models=lambda: [
            _make_model("ENet", "m1", {"Area Under PR Curve": 0.72, "AUC": 0.81}),
            _make_model("RF", "m2", {"Area Under PR Curve": 0.75, "AUC": 0.80}),
            _make_model("XGB", "m3", {"Area Under PR Curve": 0.70, "AUC": 0.82}),
        ]
    )
    best, metric = models_mod.get_best_model(project, "binary")
    assert best.id == "m2"
    assert metric == "Area Under PR Curve"


def test_get_best_model_binary_falls_back_to_auc_when_pr_null() -> None:
    project = SimpleNamespace(
        get_models=lambda: [
            _make_model("ENet", "m1", {"Area Under PR Curve": None, "AUC": 0.81}),
            _make_model("RF", "m2", {"Area Under PR Curve": None, "AUC": 0.85}),
        ]
    )
    best, metric = models_mod.get_best_model(project, "binary")
    assert best.id == "m2"
    assert metric == "AUC"


def test_get_best_model_deterministic_tie_break() -> None:
    project = SimpleNamespace(
        get_models=lambda: [
            _make_model("RF", "m9", {"Area Under PR Curve": 0.75}),
            _make_model("ENet", "m1", {"Area Under PR Curve": 0.75}),
            _make_model("XGB", "m5", {"Area Under PR Curve": 0.75}),
        ]
    )
    best, metric = models_mod.get_best_model(project, "binary")
    # Tied on score; smallest model.id wins deterministically
    assert best.id == "m1"
    assert metric == "Area Under PR Curve"


def test_get_best_model_regression_prefers_rmse_min() -> None:
    project = SimpleNamespace(
        get_models=lambda: [
            _make_model("LR", "m1", {"RMSE": 1.8}),
            _make_model("GBM", "m2", {"RMSE": 1.2}),
            _make_model("XGB", "m3", {"RMSE": 1.5}),
        ]
    )
    best, metric = models_mod.get_best_model(project, "regression")
    assert best.id == "m2"
    assert metric == "RMSE"


def test_get_best_model_regression_keeps_zero_score() -> None:
    project = SimpleNamespace(
        get_models=lambda: [
            _make_model("LR", "m1", {"RMSE": 1.5}),
            _make_model("GBM", "m2", {"RMSE": 0.0}),
        ]
    )
    best, metric = models_mod.get_best_model(project, "regression")
    assert best.id == "m2"
    assert metric == "RMSE"


def test_get_best_model_timeseries_prefers_rmse_min() -> None:
    project = SimpleNamespace(
        get_models=lambda: [
            _make_model("TS1", "m1", {"RMSE": 2.1}),
            _make_model("TS2", "m2", {"RMSE": 1.7}),
        ]
    )
    best, metric = models_mod.get_best_model(project, "timeseries")
    assert best.id == "m2"
    assert metric == "RMSE"


def test_get_best_model_uses_validation_when_cv_missing() -> None:
    project = SimpleNamespace(
        get_models=lambda: [
            _make_model(
                "RF",
                "m1",
                cv={"Area Under PR Curve": None},
                val={"Area Under PR Curve": 0.60},
            ),
            _make_model(
                "XGB",
                "m2",
                cv={"Area Under PR Curve": None},
                val={"Area Under PR Curve": 0.80},
            ),
        ]
    )
    best, metric = models_mod.get_best_model(project, "binary")
    assert best.id == "m2"
    assert metric == "Area Under PR Curve"


def test_get_best_model_raises_when_all_null() -> None:
    project = SimpleNamespace(
        get_models=lambda: [
            _make_model("X", "m1", cv={"Area Under PR Curve": None, "AUC": None}),
        ]
    )
    with pytest.raises(RuntimeError):
        models_mod.get_best_model(project, "binary")


def test_get_best_model_raises_when_no_models() -> None:
    project = SimpleNamespace(get_models=lambda: [])
    with pytest.raises(RuntimeError):
        models_mod.get_best_model(project, "binary")


def test_save_model_metrics_writes_json(tmp_path: Path) -> None:
    model = SimpleNamespace(
        id="m1",
        model_type="eXtreme Gradient Boosted Trees",
        metrics={
            "AUC": {"crossValidation": 0.82, "validation": 0.80},
            "LogLoss": {"crossValidation": 0.45, "validation": 0.47},
        },
    )
    project = SimpleNamespace(id="p1")
    out = tmp_path / "metrics.json"

    models_mod.save_model_metrics(
        model=model,
        project=project,
        output_path=out,
        selected_by_metric="AUC",
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["project_id"] == "p1"
    assert payload["model_id"] == "m1"
    assert payload["model_type"] == "eXtreme Gradient Boosted Trees"
    assert payload["selected_by_metric"] == "AUC"
    assert payload["cv_scores"]["AUC"] == 0.82
    assert payload["validation_scores"]["AUC"] == 0.80
    assert "project_url" in payload


def test_save_model_metrics_uses_env_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://dr.example.com/api/v2")
    model = SimpleNamespace(
        id="m1",
        model_type="ENet",
        metrics={"AUC": {"crossValidation": 0.7, "validation": 0.7}},
    )
    project = SimpleNamespace(id="p42")
    out = tmp_path / "metrics.json"

    models_mod.save_model_metrics(
        model=model, project=project, output_path=out, selected_by_metric="AUC"
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["project_url"] == "https://dr.example.com/projects/p42/models"
