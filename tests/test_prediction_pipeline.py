from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.pipeline import Pipeline

from predictive_maintenance.artifacts import load_frame, load_model
from predictive_maintenance.config import FEATURE_COLUMNS
from predictive_maintenance.train import train_and_compare
import predictive_maintenance.train as train_module


def _build_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vibration_rms": [0.8, 1.0, 1.2, 2.0, 2.2, 2.4, 2.6, 2.8],
            "temperature_motor": [45, 48, 50, 70, 72, 74, 76, 78],
            "rpm": [900, 950, 1000, 1500, 1550, 1600, 1650, 1700],
            "pressure_level": [28, 29, 29.5, 34, 34.5, 35, 35.5, 36],
            "rul_hours": [40, 35, 30, 10, 9, 8, 7, 6],
            "operating_mode": ["normal", "normal", "normal", "stress", "stress", "stress", "stress", "stress"],
            "failure_within_24h": [0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


def test_train_and_compare_writes_model_and_metrics(tmp_path: Path) -> None:
    csv_path = tmp_path / "industrial_machine_maintenance.csv"
    dataset = _build_dataset()
    dataset.to_csv(csv_path, index=False)

    artifact_root = tmp_path / "artifacts"
    summary = train_and_compare(csv_path=csv_path, artifact_root=artifact_root)
    comparison = load_frame(summary["comparison_path"])
    importance = load_frame(summary["importance_path"])
    reloaded_model = load_model(summary["model_path"])

    assert summary["comparison_path"].exists()
    assert summary["importance_path"].exists()
    assert summary["model_path"].exists()
    assert summary["best_model_name"] in {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "mlp_classifier",
    }
    assert isinstance(reloaded_model, Pipeline)
    assert not comparison.empty
    assert not importance.empty
    assert set(comparison.columns) == {"model_name", "f1", "precision", "recall", "roc_auc", "pr_auc"}
    assert set(importance.columns) == {"feature", "importance_mean", "importance_std"}
    assert set(importance["feature"]) == set(FEATURE_COLUMNS)
    assert len(importance) == len(FEATURE_COLUMNS)
    assert importance["importance_mean"].is_monotonic_decreasing
    assert summary["best_model_name"] in set(comparison["model_name"])


def test_train_and_compare_supports_failure_type_task(tmp_path: Path) -> None:
    csv_path = tmp_path / "industrial_machine_maintenance.csv"
    dataset = pd.DataFrame(
        {
            "vibration_rms": [0.8, 1.0, 1.2, 1.4, 2.0, 2.2],
            "temperature_motor": [45, 48, 50, 52, 70, 72],
            "rpm": [900, 950, 1000, 1050, 1500, 1550],
            "pressure_level": [28, 29, 29.5, 30.0, 34, 34.5],
            "rul_hours": [40, 35, 30, 25, 10, 9],
            "operating_mode": ["normal", "normal", "stress", "stress", "critical", "critical"],
            "failure_within_24h": [0, 0, 1, 1, 1, 1],
            "failure_type": ["none", "none", "mechanical", "mechanical", "thermal", "thermal"],
        }
    )
    dataset.to_csv(csv_path, index=False)

    summary = train_and_compare(csv_path=csv_path, task_name="failure_type", artifact_root=tmp_path / "artifacts")

    assert summary["task_name"] == "failure_type"
    assert summary["model_path"].name == "best_model_failure_type.joblib"
    assert summary["comparison_path"].name == "model_comparison_failure_type.csv"
    assert summary["importance_path"].name == "feature_importance_failure_type.csv"


def test_train_and_compare_uses_repo_root_artifact_defaults(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "industrial_machine_maintenance.csv"
    dataset = _build_dataset()
    dataset.to_csv(csv_path, index=False)
    isolated_artifact_root = tmp_path / "isolated-artifacts"
    expected_model_path = isolated_artifact_root / "model" / "best_model_failure_within_24h.joblib"
    expected_comparison_path = isolated_artifact_root / "metrics" / "model_comparison_failure_within_24h.csv"
    expected_importance_path = isolated_artifact_root / "metrics" / "feature_importance_failure_within_24h.csv"

    monkeypatch.setattr(
        train_module,
        "build_artifact_paths",
        lambda task_name: SimpleNamespace(
            model_path=expected_model_path,
            metrics_path=expected_comparison_path,
            importance_path=expected_importance_path,
        ),
    )

    summary = train_module.train_and_compare(csv_path=csv_path)

    assert summary["model_path"] == expected_model_path
    assert summary["comparison_path"] == expected_comparison_path
    assert summary["importance_path"] == expected_importance_path
