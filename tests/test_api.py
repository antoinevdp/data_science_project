import importlib
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from predictive_maintenance import config
from predictive_maintenance.api.app import app
from predictive_maintenance.api import schemas as schema_module
from predictive_maintenance.api import routes as route_module
from predictive_maintenance.api import service as service_module
from predictive_maintenance.api.service import TaskArtifactError, build_model_info_payload, load_task_bundle


from predictive_maintenance.api.schemas import ModelInfoComparisonRow, ModelInfoResponse


def test_health_endpoint_reports_degraded_when_some_tasks_are_unavailable(monkeypatch) -> None:
    def fake_build_model_info_payload(task_name: str, artifact_root=None):
        if task_name == "failure_within_24h":
            return schema_module.ModelInfoResponse(
                task_name=task_name,
                primary_metric_name="f1",
                best_model_name="gradient_boosting",
                models_comparison=[],
            )

        raise TaskArtifactError(task_name, "comparison", Path("/missing"), "is missing")

    def fake_load_task_bundle(task_name: str, artifact_root=None):
        if task_name == "failure_within_24h":
            return {"model": object(), "comparison": object(), "importance": object()}

        raise TaskArtifactError(task_name, "importance", Path("/missing"), "is missing")

    monkeypatch.setattr(
        route_module, "build_model_info_payload", fake_build_model_info_payload, raising=False
    )
    monkeypatch.setattr(route_module, "load_task_bundle", fake_load_task_bundle, raising=False)

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "available_tasks": ["failure_within_24h"]}


def test_health_endpoint_reports_degraded_when_model_info_comparison_is_malformed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "failure_within_24h": config.TaskConfig(
                task_name="failure_within_24h",
                target_column="failure_within_24h",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Failure Within 24h",
                model_set="binary",
            )
        },
        raising=False,
    )

    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_failure_within_24h.joblib")
    pd.DataFrame([{"model_name": "gradient_boosting", "f1": 0.81}]).to_csv(
        metrics_dir / "model_comparison_failure_within_24h.csv", index=False
    )
    pd.DataFrame([{"feature": "vibration_rms", "weight": 0.2}]).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )

    monkeypatch.setattr(
        route_module,
        "load_task_bundle",
        lambda task_name, artifact_root=None: service_module.load_task_bundle(
            task_name, tmp_path / "artifacts"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        route_module,
        "build_model_info_payload",
        lambda task_name, artifact_root=None: service_module.build_model_info_payload(
            task_name, tmp_path / "artifacts"
        ),
        raising=False,
    )

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "available_tasks": []}


def test_model_info_comparison_row_supports_best_model_flag() -> None:
    row = ModelInfoComparisonRow(
        model_name="gradient_boosting",
        primary_metric=0.87,
        metric_two=0.83,
        metric_three=0.88,
        is_best_model=True,
    )

    assert row.is_best_model is True


def test_model_info_response_supports_selected_task_comparison_array() -> None:
    response = ModelInfoResponse(
        task_name="failure_within_24h",
        primary_metric_name="f1",
        best_model_name="gradient_boosting",
        models_comparison=[
            ModelInfoComparisonRow(
                model_name="gradient_boosting",
                primary_metric=0.87,
                metric_two=0.83,
                metric_three=0.88,
                is_best_model=True,
            )
        ],
    )

    assert response.task_name == "failure_within_24h"
    assert response.models_comparison[0].model_name == "gradient_boosting"
    assert response.model_dump()["task_name"] == "failure_within_24h"
    assert response.model_dump()["best_model_name"] == "gradient_boosting"
    assert response.model_dump()["models_comparison"][0]["model_name"] == "gradient_boosting"


def test_build_model_info_payload_returns_selected_task_comparison_rows(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"model_name": "gradient_boosting", "f1": 0.87, "recall": 0.83, "precision": 0.88},
            {"model_name": "random_forest", "f1": 0.82, "recall": 0.8, "precision": 0.84},
        ]
    ).to_csv(metrics_dir / "model_comparison_failure_within_24h.csv", index=False)

    payload = build_model_info_payload("failure_within_24h", artifact_root=tmp_path / "artifacts")

    assert payload.task_name == "failure_within_24h"
    assert payload.best_model_name == "gradient_boosting"
    assert [row.is_best_model for row in payload.models_comparison] == [True, False]
    assert [row.model_name for row in payload.models_comparison] == [
        "gradient_boosting",
        "random_forest",
    ]
    assert payload.models_comparison[0].is_best_model is True


def test_build_model_info_payload_marks_best_model_from_metrics_even_when_unsorted(
    tmp_path: Path,
) -> None:
    metrics_dir = tmp_path / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"model_name": "random_forest", "f1": 0.82, "recall": 0.8, "precision": 0.84},
            {"model_name": "gradient_boosting", "f1": 0.87, "recall": 0.83, "precision": 0.88},
        ]
    ).to_csv(metrics_dir / "model_comparison_failure_within_24h.csv", index=False)

    payload = build_model_info_payload("failure_within_24h", artifact_root=tmp_path / "artifacts")

    assert payload.best_model_name == "gradient_boosting"
    assert [row.model_name for row in payload.models_comparison] == [
        "random_forest",
        "gradient_boosting",
    ]
    assert [row.is_best_model for row in payload.models_comparison] == [False, True]


def test_build_model_info_payload_breaks_primary_metric_ties_with_supporting_metrics(
    tmp_path: Path,
) -> None:
    metrics_dir = tmp_path / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"model_name": "random_forest", "f1": 0.87, "recall": 0.8, "precision": 0.9},
            {"model_name": "gradient_boosting", "f1": 0.87, "recall": 0.83, "precision": 0.88},
        ]
    ).to_csv(metrics_dir / "model_comparison_failure_within_24h.csv", index=False)

    payload = build_model_info_payload("failure_within_24h", artifact_root=tmp_path / "artifacts")

    assert payload.best_model_name == "gradient_boosting"
    assert [row.is_best_model for row in payload.models_comparison] == [False, True]


def test_build_model_info_payload_rejects_duplicate_model_names_after_normalization(
    tmp_path: Path,
) -> None:
    metrics_dir = tmp_path / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"model_name": " xgboost ", "f1": 0.87, "recall": 0.83, "precision": 0.88},
            {"model_name": "xgboost", "f1": 0.82, "recall": 0.8, "precision": 0.84},
        ]
    ).to_csv(metrics_dir / "model_comparison_failure_within_24h.csv", index=False)

    with pytest.raises(TaskArtifactError, match="failure_within_24h.*duplicate model names"):
        build_model_info_payload("failure_within_24h", artifact_root=tmp_path / "artifacts")


def test_build_model_info_payload_rejects_malformed_comparison_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "failure_within_24h": config.TaskConfig(
                task_name="failure_within_24h",
                target_column="failure_within_24h",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Failure Within 24h",
                model_set="binary",
            )
        },
        raising=False,
    )

    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_failure_within_24h.joblib")
    pd.DataFrame([{"model_name": "gradient_boosting"}]).to_csv(
        metrics_dir / "model_comparison_failure_within_24h.csv", index=False
    )
    pd.DataFrame([{"feature": "vibration_rms", "weight": 0.2}]).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )

    with pytest.raises(TaskArtifactError, match="failure_within_24h.*comparison"):
        build_model_info_payload("failure_within_24h", artifact_root=tmp_path / "artifacts")


def test_build_model_info_payload_rejects_empty_comparison_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "failure_within_24h": config.TaskConfig(
                task_name="failure_within_24h",
                target_column="failure_within_24h",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Failure Within 24h",
                model_set="binary",
            )
        },
        raising=False,
    )

    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_failure_within_24h.joblib")
    pd.DataFrame(
        columns=["model_name", "f1", "recall", "precision"]
    ).to_csv(metrics_dir / "model_comparison_failure_within_24h.csv", index=False)
    pd.DataFrame([{"feature": "vibration_rms", "weight": 0.2}]).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )

    with pytest.raises(TaskArtifactError, match="failure_within_24h.*comparison"):
        build_model_info_payload("failure_within_24h", artifact_root=tmp_path / "artifacts")


def test_build_model_info_payload_rejects_non_numeric_primary_metric_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "failure_within_24h": config.TaskConfig(
                task_name="failure_within_24h",
                target_column="failure_within_24h",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Failure Within 24h",
                model_set="binary",
            )
        },
        raising=False,
    )

    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_failure_within_24h.joblib")
    pd.DataFrame(
        [
            {"model_name": "gradient_boosting", "f1": "bad", "recall": 0.8, "precision": 0.9},
            {"model_name": "random_forest", "f1": 0.82, "recall": 0.79, "precision": 0.84},
        ]
    ).to_csv(metrics_dir / "model_comparison_failure_within_24h.csv", index=False)
    pd.DataFrame([{"feature": "vibration_rms", "weight": 0.2}]).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )

    with pytest.raises(TaskArtifactError, match="failure_within_24h.*primary metric"):
        build_model_info_payload("failure_within_24h", artifact_root=tmp_path / "artifacts")


def test_build_model_info_payload_rejects_non_numeric_supporting_metric_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "failure_within_24h": config.TaskConfig(
                task_name="failure_within_24h",
                target_column="failure_within_24h",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Failure Within 24h",
                model_set="binary",
            )
        },
        raising=False,
    )

    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_failure_within_24h.joblib")
    pd.DataFrame(
        [
            {"model_name": "gradient_boosting", "f1": 0.87, "recall": "NaN", "precision": 0.9},
            {"model_name": "random_forest", "f1": 0.82, "recall": 0.79, "precision": 0.84},
        ]
    ).to_csv(metrics_dir / "model_comparison_failure_within_24h.csv", index=False)
    pd.DataFrame([{"feature": "vibration_rms", "weight": 0.2}]).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )

    with pytest.raises(TaskArtifactError, match="failure_within_24h.*comparison"):
        build_model_info_payload("failure_within_24h", artifact_root=tmp_path / "artifacts")


def test_build_model_info_payload_rejects_blank_model_names_for_failure_type(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "failure_type": config.TaskConfig(
                task_name="failure_type",
                target_column="failure_type",
                problem_type="multiclass_classification",
                primary_metric="weighted_f1",
                display_name="Failure Type",
                model_set="multiclass",
            )
        },
        raising=False,
    )

    metrics_dir = tmp_path / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "model_name": " ",
                "weighted_f1": 0.74,
                "macro_f1": 0.7,
                "weighted_recall": 0.76,
            },
            {
                "model_name": "xgboost",
                "weighted_f1": 0.72,
                "macro_f1": 0.68,
                "weighted_recall": 0.74,
            },
        ]
    ).to_csv(metrics_dir / "model_comparison_failure_type.csv", index=False)

    with pytest.raises(TaskArtifactError, match="failure_type.*model name"):
        build_model_info_payload("failure_type", artifact_root=tmp_path / "artifacts")


def test_build_model_info_payload_rejects_non_numeric_supporting_metric_values_for_failure_type(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "failure_type": config.TaskConfig(
                task_name="failure_type",
                target_column="failure_type",
                problem_type="multiclass_classification",
                primary_metric="weighted_f1",
                display_name="Failure Type",
                model_set="multiclass",
            )
        },
        raising=False,
    )

    metrics_dir = tmp_path / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "model_name": "xgboost",
                "weighted_f1": 0.74,
                "macro_f1": "bad",
                "weighted_recall": 0.76,
            },
            {
                "model_name": "random_forest",
                "weighted_f1": 0.72,
                "macro_f1": 0.68,
                "weighted_recall": 0.74,
            },
        ]
    ).to_csv(metrics_dir / "model_comparison_failure_type.csv", index=False)

    with pytest.raises(TaskArtifactError, match="failure_type.*comparison"):
        build_model_info_payload("failure_type", artifact_root=tmp_path / "artifacts")


def test_model_info_endpoint_returns_ready_tasks_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "failure_within_24h": config.TaskConfig(
                task_name="failure_within_24h",
                target_column="failure_within_24h",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Failure Within 24h",
                model_set="binary",
            )
        },
        raising=False,
    )

    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_failure_within_24h.joblib")
    pd.DataFrame([{"model_name": "stub", "f1": 0.81, "recall": 0.8, "precision": 0.79}]).to_csv(
        metrics_dir / "model_comparison_failure_within_24h.csv", index=False
    )
    pd.DataFrame([{"feature": "vibration_rms", "weight": 0.2}]).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )

    monkeypatch.setattr(
        route_module,
        "build_model_info_payload",
        lambda task_name: service_module.build_model_info_payload(task_name, tmp_path / "artifacts"),
    )

    client = TestClient(app)

    response = client.get("/model-info", params={"task_name": "failure_within_24h"})

    assert response.status_code == 200
    assert response.json() == {
        "task_name": "failure_within_24h",
        "primary_metric_name": "f1",
        "best_model_name": "stub",
        "models_comparison": [
            {
                "model_name": "stub",
                "primary_metric": 0.81,
                "metric_two": 0.8,
                "metric_three": 0.79,
                "is_best_model": True,
            }
        ],
        "tasks": []
    }


def test_model_info_endpoint_filters_by_task_name(monkeypatch) -> None:
    client = TestClient(app)

    response = client.get("/model-info", params={"task_name": "failure_type"})

    assert response.status_code == 200
    assert response.json()["task_name"] == "failure_type"


def test_model_info_endpoint_accepts_tasks_from_runtime_config(monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "custom_task": config.TaskConfig(
                task_name="custom_task",
                target_column="custom_task",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Custom Task",
                model_set="binary",
            )
        },
        raising=False,
    )
    monkeypatch.setattr(
        route_module,
        "build_model_info_payload",
        lambda task_name: schema_module.ModelInfoResponse(
            task_name=task_name,
            primary_metric_name="f1",
            best_model_name="gradient_boosting",
            models_comparison=[],
        ),
    )

    client = TestClient(app)

    response = client.get("/model-info", params={"task_name": "custom_task"})

    assert response.status_code == 200
    assert response.json()["task_name"] == "custom_task"


def test_model_info_endpoint_defaults_to_primary_task(monkeypatch) -> None:
    client = TestClient(app)

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["task_name"] == config.DEFAULT_TASK_NAME


def test_health_endpoint_requires_full_task_bundle(monkeypatch) -> None:
    monkeypatch.setattr(
        route_module,
        "build_model_info_payload",
        lambda task_name: schema_module.ModelInfoResponse(
            task_name=task_name,
            primary_metric_name="f1",
            best_model_name="gradient_boosting",
            models_comparison=[],
        ),
    )

    def fake_load_task_bundle(task_name: str, artifact_root=None):
        if task_name == "failure_within_24h":
            return {"model": object(), "comparison": object(), "importance": object()}

        raise TaskArtifactError(task_name, "importance", Path("/missing"), "is missing")

    monkeypatch.setattr(route_module, "load_task_bundle", fake_load_task_bundle, raising=False)

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "available_tasks": ["failure_within_24h"]}


def test_build_model_info_payload_returns_task_when_artifacts_are_loadable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "failure_within_24h": config.TaskConfig(
                task_name="failure_within_24h",
                target_column="failure_within_24h",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Failure Within 24h",
                model_set="binary",
            )
        },
        raising=False,
    )

    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_failure_within_24h.joblib")
    pd.DataFrame([{"model_name": "stub", "f1": 0.81}]).to_csv(
        metrics_dir / "model_comparison_failure_within_24h.csv", index=False
    )
    pd.DataFrame([{"feature": "vibration_rms", "weight": 0.2}]).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )

    payload = build_model_info_payload(tmp_path / "artifacts")

    assert len(payload.tasks) == 1
    assert payload.tasks[0].task_name == "failure_within_24h"
    assert payload.tasks[0].model_name == "stub"


def test_build_model_info_payload_includes_partial_tasks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "failure_within_24h": config.TaskConfig(
                task_name="failure_within_24h",
                target_column="failure_within_24h",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Failure Within 24h",
                model_set="binary",
            ),
            "missing_model": config.TaskConfig(
                task_name="missing_model",
                target_column="missing_model",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Missing Model",
                model_set="binary",
            ),
            "corrupted_metrics": config.TaskConfig(
                task_name="corrupted_metrics",
                target_column="corrupted_metrics",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Corrupted Metrics",
                model_set="binary",
            ),
        },
        raising=False,
    )

    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_failure_within_24h.joblib")
    pd.DataFrame([{"model_name": "gradient_boosting", "f1": 0.81}]).to_csv(
        metrics_dir / "model_comparison_failure_within_24h.csv", index=False
    )
    pd.DataFrame([{ "f1": 0.63 }]).to_csv(
        metrics_dir / "model_comparison_missing_model.csv", index=False
    )
    (model_dir / "best_model_missing_model.joblib").unlink(missing_ok=True)
    joblib.dump({"model": "stub"}, model_dir / "best_model_corrupted_metrics.joblib")
    pd.DataFrame([{ "feature": "vibration_rms", "weight": 0.2 }]).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )

    payload = build_model_info_payload(tmp_path / "artifacts")

    assert [task.task_name for task in payload.tasks] == ["failure_within_24h"]
    assert payload.tasks[0].model_name == "gradient_boosting"


def test_build_model_info_payload_skips_task_when_importance_artifact_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "missing_importance": config.TaskConfig(
                task_name="missing_importance",
                target_column="missing_importance",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Missing Importance",
                model_set="binary",
            ),
        },
        raising=False,
    )

    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_missing_importance.joblib")
    pd.DataFrame([{"model_name": "gradient_boosting", "f1": 0.81}]).to_csv(
        metrics_dir / "model_comparison_missing_importance.csv", index=False
    )

    payload = build_model_info_payload(tmp_path / "artifacts")

    assert payload.tasks == []


def test_build_model_info_payload_skips_task_when_importance_artifact_is_corrupt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "corrupt_importance": config.TaskConfig(
                task_name="corrupt_importance",
                target_column="corrupt_importance",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Corrupt Importance",
                model_set="binary",
            ),
        },
        raising=False,
    )

    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_corrupt_importance.joblib")
    pd.DataFrame([{"model_name": "gradient_boosting", "f1": 0.81}]).to_csv(
        metrics_dir / "model_comparison_corrupt_importance.csv", index=False
    )
    (metrics_dir / "feature_importance_corrupt_importance.csv").write_text("")

    payload = build_model_info_payload(tmp_path / "artifacts")

    assert payload.tasks == []


def test_prediction_request_rejects_stringified_numeric_fields() -> None:
    with pytest.raises(ValidationError):
        schema_module.FailureWithin24hPredictionRequest(
            vibration_rms="1.2",
            temperature_motor=60.0,
            rpm=1200.0,
            pressure_level=30.5,
            rul_hours=12.0,
            operating_mode="normal",
        )


def test_build_model_info_payload_skips_task_when_model_artifact_is_corrupt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "corrupt_model": config.TaskConfig(
                task_name="corrupt_model",
                target_column="corrupt_model",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Corrupt Model",
                model_set="binary",
            ),
        },
        raising=False,
    )

    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    (model_dir / "best_model_corrupt_model.joblib").write_text("not-a-joblib-payload")
    pd.DataFrame([{"model_name": "gradient_boosting", "f1": 0.81}]).to_csv(
        metrics_dir / "model_comparison_corrupt_model.csv", index=False
    )
    pd.DataFrame([{"feature": "vibration_rms", "weight": 0.2}]).to_csv(
        metrics_dir / "feature_importance_corrupt_model.csv", index=False
    )

    payload = build_model_info_payload(tmp_path / "artifacts")

    assert payload.tasks == []


def test_load_task_bundle_caches_repeated_loads(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_failure_within_24h.joblib")
    pd.DataFrame([{"model_name": "gradient_boosting", "f1": 0.81}]).to_csv(
        metrics_dir / "model_comparison_failure_within_24h.csv", index=False
    )
    pd.DataFrame([{"feature": "vibration_rms", "weight": 0.2}]).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )

    load_model_calls = 0
    load_frame_calls = 0
    original_load_model = service_module.load_model
    original_load_frame = service_module.load_frame

    if hasattr(service_module, "_load_task_bundle_cached"):
        service_module._load_task_bundle_cached.cache_clear()  # type: ignore[attr-defined]

    def counting_load_model(path: Path):
        nonlocal load_model_calls
        load_model_calls += 1
        return original_load_model(path)

    def counting_load_frame(path: Path):
        nonlocal load_frame_calls
        load_frame_calls += 1
        return original_load_frame(path)

    monkeypatch.setattr(service_module, "load_model", counting_load_model)
    monkeypatch.setattr(service_module, "load_frame", counting_load_frame)

    first_bundle = load_task_bundle("failure_within_24h", tmp_path / "artifacts")
    second_bundle = load_task_bundle("failure_within_24h", tmp_path / "artifacts")

    assert first_bundle is second_bundle
    assert load_model_calls == 1
    assert load_frame_calls == 2


def test_load_task_bundle_invalidates_cache_when_artifacts_change(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_failure_within_24h.joblib")
    pd.DataFrame([{"model_name": "gradient_boosting", "f1": 0.81}]).to_csv(
        metrics_dir / "model_comparison_failure_within_24h.csv", index=False
    )
    pd.DataFrame([{"feature": "vibration_rms", "weight": 0.2}]).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )

    load_model_calls = 0
    load_frame_calls = 0
    original_load_model = service_module.load_model
    original_load_frame = service_module.load_frame

    if hasattr(service_module, "_load_task_bundle_cached"):
        service_module._load_task_bundle_cached.cache_clear()  # type: ignore[attr-defined]

    def counting_load_model(path: Path):
        nonlocal load_model_calls
        load_model_calls += 1
        return original_load_model(path)

    def counting_load_frame(path: Path):
        nonlocal load_frame_calls
        load_frame_calls += 1
        return original_load_frame(path)

    monkeypatch.setattr(service_module, "load_model", counting_load_model)
    monkeypatch.setattr(service_module, "load_frame", counting_load_frame)

    first_bundle = load_task_bundle("failure_within_24h", tmp_path / "artifacts")

    original_stat = (metrics_dir / "feature_importance_failure_within_24h.csv").stat()
    pd.DataFrame(
        [
            {"feature": "vibration_rms", "weight": 0.9},
            {"feature": "temperature_motor", "weight": 0.1},
        ]
    ).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )
    os.utime(
        metrics_dir / "feature_importance_failure_within_24h.csv",
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    second_bundle = load_task_bundle("failure_within_24h", tmp_path / "artifacts")

    assert first_bundle is not second_bundle
    assert load_model_calls == 2
    assert load_frame_calls == 4
    assert second_bundle["importance"].iloc[0]["weight"] == 0.9


def test_load_task_bundle_invalidates_cache_when_artifact_contents_change_without_size_delta(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    joblib.dump({"model": "stub"}, model_dir / "best_model_failure_within_24h.joblib")
    pd.DataFrame([{"model_name": "gradient_boosting", "f1": 0.81}]).to_csv(
        metrics_dir / "model_comparison_failure_within_24h.csv", index=False
    )
    artifact_path = metrics_dir / "feature_importance_failure_within_24h.csv"
    artifact_path.write_text("feature,weight\nvibration_rms,0.2\n")

    load_model_calls = 0
    load_frame_calls = 0
    original_load_model = service_module.load_model
    original_load_frame = service_module.load_frame

    if hasattr(service_module, "_load_task_bundle_cached"):
        service_module._load_task_bundle_cached.cache_clear()  # type: ignore[attr-defined]

    def counting_load_model(path: Path):
        nonlocal load_model_calls
        load_model_calls += 1
        return original_load_model(path)

    def counting_load_frame(path: Path):
        nonlocal load_frame_calls
        load_frame_calls += 1
        return original_load_frame(path)

    monkeypatch.setattr(service_module, "load_model", counting_load_model)
    monkeypatch.setattr(service_module, "load_frame", counting_load_frame)

    first_bundle = load_task_bundle("failure_within_24h", tmp_path / "artifacts")

    original_stat = artifact_path.stat()
    artifact_path.write_text("feature,weight\nvibration_rms,0.9\n")
    os.utime(artifact_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    second_bundle = load_task_bundle("failure_within_24h", tmp_path / "artifacts")

    assert first_bundle is not second_bundle
    assert load_model_calls == 2
    assert load_frame_calls == 4
    assert second_bundle["importance"].iloc[0]["weight"] == 0.9


def test_artifact_signature_includes_inode_and_size(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "metrics" / "feature_importance_failure_within_24h.csv"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("feature,weight\nvibration_rms,0.2\n")

    signature = service_module._artifact_signature(artifact_path)

    assert len(signature) == 4
    assert signature[0] == artifact_path.stat().st_ino
    assert signature[1] == artifact_path.stat().st_ctime_ns
    assert signature[2] == artifact_path.stat().st_mtime_ns
    assert signature[3] == artifact_path.stat().st_size


def test_build_model_info_payload_propagates_unexpected_task_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "broken_task": config.TaskConfig(
                task_name="broken_task",
                target_column="broken_task",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Broken Task",
                model_set="binary",
            )
        },
        raising=False,
    )

    def boom(task_name: str):
        raise RuntimeError(f"boom: {task_name}")

    monkeypatch.setattr(config, "get_task_config", boom, raising=False)

    with pytest.raises(RuntimeError, match="boom: broken_task"):
        build_model_info_payload(tmp_path / "artifacts")


def test_load_task_bundle_raises_task_scoped_error_for_missing_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    pd.DataFrame([{"model_name": "gradient_boosting", "f1": 0.81}]).to_csv(
        metrics_dir / "model_comparison_failure_within_24h.csv", index=False
    )

    with pytest.raises(TaskArtifactError, match="failure_within_24h.*best_model_failure_within_24h.joblib"):
        load_task_bundle("failure_within_24h", tmp_path / "artifacts")


def test_load_task_bundle_wraps_corrupt_model_artifact(tmp_path: Path) -> None:
    model_dir = tmp_path / "artifacts" / "model"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    model_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    (model_dir / "best_model_failure_within_24h.joblib").write_text("not-a-joblib-payload")
    pd.DataFrame([{"model_name": "gradient_boosting", "f1": 0.81}]).to_csv(
        metrics_dir / "model_comparison_failure_within_24h.csv", index=False
    )
    pd.DataFrame([{"feature": "vibration_rms", "weight": 0.2}]).to_csv(
        metrics_dir / "feature_importance_failure_within_24h.csv", index=False
    )

    with pytest.raises(TaskArtifactError, match="failure_within_24h.*could not be read"):
        load_task_bundle("failure_within_24h", tmp_path / "artifacts")

def test_prediction_request_fields_follow_canonical_feature_contract(monkeypatch) -> None:
    original_fields = tuple(config.FEATURE_COLUMNS)

    with monkeypatch.context() as patched:
        patched.setattr(config, "NUMERIC_COLUMNS", ["vibration_rms", "temperature_motor"], raising=False)
        patched.setattr(config, "CATEGORICAL_COLUMNS", ["operating_mode", "shift"], raising=False)
        patched.setattr(
            config,
            "FEATURE_COLUMNS",
            config.NUMERIC_COLUMNS + config.CATEGORICAL_COLUMNS,
            raising=False,
        )

        schemas = importlib.reload(schema_module)

        expected_fields = config.FEATURE_COLUMNS
        assert list(schemas.FailureWithin24hPredictionRequest.model_fields) == expected_fields
        assert list(schemas.FailureTypePredictionRequest.model_fields) == expected_fields

        payload = schemas.FailureWithin24hPredictionRequest(
            vibration_rms=1.2,
            temperature_motor=60.0,
            operating_mode="normal",
            shift="day",
        )

        assert list(payload.model_dump()) == expected_fields

    restored_schemas = importlib.reload(schema_module)
    assert list(restored_schemas.FailureWithin24hPredictionRequest.model_fields) == list(original_fields)


def test_prediction_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        schema_module.FailureWithin24hPredictionRequest(
            vibration_rms=1.2,
            temperature_motor=60.0,
            rpm=1200.0,
            pressure_level=30.5,
            rul_hours=12.0,
            operating_mode="normal",
            unexpected_field="nope",
        )


@pytest.mark.parametrize(
    "path",
    ["/predict/failure-within-24h", "/predict/failure-type"],
)
def test_predict_rejects_missing_field(path: str) -> None:
    client = TestClient(app)

    response = client.post(
        path,
        json={
            "temperature_motor": 60.0,
            "rpm": 1200.0,
            "pressure_level": 30.5,
            "rul_hours": 12.0,
            "operating_mode": "normal",
        },
    )

    assert response.status_code == 422


def test_predict_failure_within_24h_returns_rich_binary_response(monkeypatch) -> None:
    client = TestClient(app)

    response = client.post(
        "/predict/failure-within-24h",
        json={
            "vibration_rms": 1.2,
            "temperature_motor": 60.0,
            "rpm": 1200.0,
            "pressure_level": 30.5,
            "rul_hours": 12.0,
            "operating_mode": "normal",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_name"] == "failure_within_24h"
    assert "probability" in body
    assert "model_name" in body


def test_predict_failure_type_returns_class_probabilities(monkeypatch) -> None:
    client = TestClient(app)

    response = client.post(
        "/predict/failure-type",
        json={
            "vibration_rms": 1.2,
            "temperature_motor": 60.0,
            "rpm": 1200.0,
            "pressure_level": 30.5,
            "rul_hours": 12.0,
            "operating_mode": "stress",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_name"] == "failure_type"
    assert "class_probabilities" in body


@pytest.mark.parametrize(
    ("handler_name", "task_name", "payload"),
    [
        (
            "predict_failure_within_24h",
            "failure_within_24h",
            {
                "vibration_rms": 1.2,
                "temperature_motor": 60.0,
                "rpm": 1200.0,
                "pressure_level": 30.5,
                "rul_hours": 12.0,
                "operating_mode": "normal",
            },
        ),
        (
            "predict_failure_type",
            "failure_type",
            {
                "vibration_rms": 1.2,
                "temperature_motor": 60.0,
                "rpm": 1200.0,
                "pressure_level": 30.5,
                "rul_hours": 12.0,
                "operating_mode": "stress",
            },
        ),
    ],
)
def test_prediction_helpers_raise_task_artifact_error_for_malformed_importance_artifacts(
    monkeypatch, handler_name: str, task_name: str, payload: dict[str, float | str]
) -> None:
    model = type(
        "PredictModel",
        (),
        {
            "classes_": np.array(["A", "B", "C"]),
            "predict_proba": lambda self, frame: np.array([[0.2, 0.5, 0.3]]),
            "predict": lambda self, frame: np.array(["B"]),
        },
    )()
    if handler_name == "predict_failure_within_24h":
        model = type(
            "BinaryModel",
            (),
            {
                "predict_proba": lambda self, frame: np.array([[0.2, 0.8]]),
                "predict": lambda self, frame: np.array([1]),
            },
        )()

    monkeypatch.setattr(
        service_module,
        "load_task_bundle",
        lambda selected_task_name: {
            "model": model,
            "comparison": pd.DataFrame([{"model_name": "gradient_boosting"}]),
            "importance": pd.DataFrame([{"weight": 0.2}]),
        },
    )

    helper = getattr(service_module, handler_name)

    with pytest.raises(TaskArtifactError, match=f"{task_name}.*importance"):
        helper(payload)


@pytest.mark.parametrize(
    ("path", "handler_name", "payload"),
    [
        (
            "/predict/failure-within-24h",
            "predict_failure_within_24h",
            {
                "vibration_rms": 1.2,
                "temperature_motor": 60.0,
                "rpm": 1200.0,
                "pressure_level": 30.5,
                "rul_hours": 12.0,
                "operating_mode": "normal",
            },
        ),
        (
            "/predict/failure-type",
            "predict_failure_type",
            {
                "vibration_rms": 1.2,
                "temperature_motor": 60.0,
                "rpm": 1200.0,
                "pressure_level": 30.5,
                "rul_hours": 12.0,
                "operating_mode": "stress",
            },
        ),
    ],
)
def test_predict_routes_translate_task_artifact_errors_to_http_responses(
    monkeypatch, path: str, handler_name: str, payload: dict[str, float | str]
) -> None:
    monkeypatch.setattr(
        route_module,
        handler_name,
        lambda raw_values: (_ for _ in ()).throw(
            TaskArtifactError("failure_within_24h", "model", Path("/tmp/missing"), "is missing")
        ),
    )

    client = TestClient(app)

    response = client.post(path, json=payload)

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "task_name": "failure_within_24h",
            "artifact_name": "model",
            "reason": "is missing",
        }
    }


def test_build_model_info_payload_evaluates_artifact_root_per_task(tmp_path: Path, monkeypatch) -> None:
    local_root = tmp_path / "worktree" / "artifacts"
    fallback_root = tmp_path / "artifacts"
    for root in (local_root, fallback_root):
        (root / "model").mkdir(parents=True)
        (root / "metrics").mkdir(parents=True)

    monkeypatch.setattr(service_module, "ARTIFACT_ROOT", local_root)
    monkeypatch.setattr(
        config,
        "TASK_CONFIGS",
        {
            "failure_within_24h": config.TaskConfig(
                task_name="failure_within_24h",
                target_column="failure_within_24h",
                problem_type="binary_classification",
                primary_metric="f1",
                display_name="Failure Within 24h",
                model_set="binary",
            ),
            "failure_type": config.TaskConfig(
                task_name="failure_type",
                target_column="failure_type",
                problem_type="multiclass_classification",
                primary_metric="weighted_f1",
                display_name="Failure Type",
                model_set="multiclass",
            ),
        },
        raising=False,
    )

    joblib.dump({"model": "fallback_model"}, fallback_root / "model" / "best_model_failure_within_24h.joblib")
    pd.DataFrame([{"model_name": "fallback_model", "f1": 0.81}]).to_csv(
        fallback_root / "metrics" / "model_comparison_failure_within_24h.csv", index=False
    )
    pd.DataFrame([{"feature": "vibration_rms"}]).to_csv(
        fallback_root / "metrics" / "feature_importance_failure_within_24h.csv", index=False
    )
    joblib.dump({"model": "local_model"}, local_root / "model" / "best_model_failure_type.joblib")
    pd.DataFrame([{"model_name": "local_model", "weighted_f1": 0.72}]).to_csv(
        local_root / "metrics" / "model_comparison_failure_type.csv", index=False
    )
    pd.DataFrame([{"feature": "temperature_motor"}]).to_csv(
        local_root / "metrics" / "feature_importance_failure_type.csv", index=False
    )

    payload = service_module.build_model_info_payload(local_root)

    assert [task.task_name for task in payload.tasks] == ["failure_within_24h", "failure_type"]
    assert payload.tasks[0].model_name == "fallback_model"
    assert payload.tasks[1].model_name == "local_model"
    assert payload.tasks[1].model_name == "local_model"


@pytest.mark.parametrize(
    ("helper_name", "expected_key", "model"),
    [
        (
            "predict_failure_within_24h",
            "predicted_label",
            type(
                "BinaryModel",
                (),
                {
                    "predict_proba": lambda self, frame: np.array([[0.2, 0.8]]),
                    "predict": lambda self, frame: np.array([1]),
                },
            )(),
        ),
        (
            "predict_failure_type",
            "predicted_class",
            type(
                "MulticlassModel",
                (),
                {
                    "classes_": np.array(["A", "B", "C"]),
                    "predict_proba": lambda self, frame: np.array([[0.2, 0.5, 0.3]]),
                    "predict": lambda self, frame: np.array(["B"]),
                },
            )(),
        ),
    ],
)
def test_prediction_helpers_use_safe_model_name_extraction(
    monkeypatch, helper_name: str, expected_key: str, model
) -> None:
    artifact_paths = type("ArtifactPaths", (), {"metrics_path": Path("/tmp/metrics.csv")})()
    bundle = {
        "model": model,
        "comparison": pd.DataFrame([{ "model_name": "legacy" }]),
        "importance": pd.DataFrame([{ "feature": "vibration_rms" }, { "feature": "temperature_motor" }]),
        "artifact_paths": artifact_paths,
    }

    observed_paths: list[Path] = []

    monkeypatch.setattr(service_module, "load_task_bundle", lambda task_name: bundle)

    helper = getattr(service_module, helper_name)
    response = helper(
        {
            "vibration_rms": 1.2,
            "temperature_motor": 60.0,
            "rpm": 1200.0,
            "pressure_level": 30.5,
            "rul_hours": 12.0,
            "operating_mode": "normal",
        }
    )

    assert response.model_name == "legacy"
    assert getattr(response, expected_key)
