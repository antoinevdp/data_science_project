from pathlib import Path

import pytest

import predictive_maintenance as package
from predictive_maintenance import config
from predictive_maintenance.config import (
    DEFAULT_TASK_NAME,
    TASK_CONFIGS,
    build_artifact_paths,
)


def test_project_config_matches_mvp_scope() -> None:
    assert package.__all__ == ["FEATURE_COLUMNS", "RANDOM_STATE", "TARGET_COLUMN"]
    assert package.TARGET_COLUMN == "failure_within_24h"
    assert package.FEATURE_COLUMNS == [
        "vibration_rms",
        "temperature_motor",
        "rpm",
        "pressure_level",
        "rul_hours",
        "operating_mode",
    ]
    assert package.RANDOM_STATE == 42
    assert config.REQUIRED_COLUMNS == package.FEATURE_COLUMNS + [package.TARGET_COLUMN]


def test_artifact_and_dataset_paths_are_repo_relative() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert config.DATASET_PATH == repo_root / "data" / "industrial_machine_maintenance.csv"
    assert (
        config.MODEL_ARTIFACT_PATH
        == repo_root / "artifacts" / "model" / "best_model_failure_within_24h.joblib"
    )
    assert (
        config.METRICS_ARTIFACT_PATH
        == repo_root / "artifacts" / "metrics" / "model_comparison_failure_within_24h.csv"
    )
    assert (
        config.IMPORTANCE_ARTIFACT_PATH
        == repo_root / "artifacts" / "metrics" / "feature_importance_failure_within_24h.csv"
    )


def test_task_registry_exposes_binary_and_multiclass_tasks() -> None:
    assert DEFAULT_TASK_NAME == "failure_within_24h"
    assert set(TASK_CONFIGS) == {"failure_within_24h", "failure_type"}
    assert TASK_CONFIGS["failure_type"].target_column == "failure_type"
    assert TASK_CONFIGS["failure_type"].primary_metric == "weighted_f1"


def test_task_registry_keys_match_config_task_names() -> None:
    assert all(task_name == task_config.task_name for task_name, task_config in TASK_CONFIGS.items())


def test_get_task_config_rejects_registry_key_mismatches(monkeypatch) -> None:
    monkeypatch.setitem(
        config.TASK_CONFIGS,
        "drifted_failure_type",
        config.TaskConfig(
            task_name="failure_type",
            target_column="failure_type",
            problem_type="multiclass_classification",
            primary_metric="weighted_f1",
            display_name="Failure Type",
            model_set="multiclass",
        ),
    )

    with pytest.raises(ValueError, match="Registry key/task_name mismatch: drifted_failure_type != failure_type"):
        config.get_task_config("drifted_failure_type")


def test_build_artifact_paths_are_task_specific() -> None:
    artifact_paths = build_artifact_paths("failure_type")

    assert artifact_paths.model_path.name == "best_model_failure_type.joblib"
    assert artifact_paths.metrics_path.name == "model_comparison_failure_type.csv"
    assert artifact_paths.importance_path.name == "feature_importance_failure_type.csv"


def test_build_artifact_paths_reject_unknown_tasks() -> None:
    with pytest.raises(ValueError, match="Unknown task: unknown_task"):
        build_artifact_paths("unknown_task")


def test_train_entrypoint_reports_training_summary(capsys, monkeypatch) -> None:
    pytest.importorskip("sklearn")
    import predictive_maintenance.train as train_module

    monkeypatch.setattr(
        train_module,
        "train_and_compare",
        lambda task_name=DEFAULT_TASK_NAME: {
            "task_name": "failure_within_24h",
            "best_model_name": "random_forest",
            "model_path": Path("artifacts/model/best_model_failure_within_24h.joblib"),
            "comparison_path": Path("artifacts/metrics/model_comparison_failure_within_24h.csv"),
        },
    )

    train_module.main()

    captured = capsys.readouterr()
    assert "Task: failure_within_24h" in captured.out
    assert "Best model: random_forest" in captured.out
    assert "Saved model: artifacts/model/best_model_failure_within_24h.joblib" in captured.out
    assert "Saved metrics: artifacts/metrics/model_comparison_failure_within_24h.csv" in captured.out


def test_train_entrypoint_accepts_task_name_argument(capsys, monkeypatch) -> None:
    pytest.importorskip("sklearn")
    import predictive_maintenance.train as train_module

    observed = {}

    def fake_train_and_compare(task_name=DEFAULT_TASK_NAME):
        observed["task_name"] = task_name
        return {
            "task_name": task_name,
            "best_model_name": "random_forest",
            "model_path": Path(f"artifacts/model/best_model_{task_name}.joblib"),
            "comparison_path": Path(f"artifacts/metrics/model_comparison_{task_name}.csv"),
        }

    monkeypatch.setattr(train_module, "train_and_compare", fake_train_and_compare)

    train_module.main(["--task-name", "failure_type"])

    captured = capsys.readouterr()
    assert observed["task_name"] == "failure_type"
    assert "Task: failure_type" in captured.out
