import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_BASE_URL = os.getenv("PREDICTIVE_MAINTENANCE_API_BASE_URL", "http://127.0.0.1:8000")

RANDOM_STATE = 42
DEFAULT_TASK_NAME = "failure_within_24h"

NUMERIC_COLUMNS = [
    "vibration_rms",
    "temperature_motor",
    "rpm",
    "pressure_level",
    "rul_hours",
]
CATEGORICAL_COLUMNS = ["operating_mode"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


@dataclass(frozen=True)
class TaskConfig:
    task_name: str
    target_column: str
    problem_type: str
    primary_metric: str
    display_name: str
    model_set: str


@dataclass(frozen=True)
class ArtifactPaths:
    model_path: Path
    metrics_path: Path
    importance_path: Path


TASK_CONFIGS = {
    "failure_within_24h": TaskConfig(
        task_name="failure_within_24h",
        target_column="failure_within_24h",
        problem_type="binary_classification",
        primary_metric="f1",
        display_name="Failure Within 24h",
        model_set="binary",
    ),
    "failure_type": TaskConfig(
        task_name="failure_type",
        target_column="failure_type",
        problem_type="multiclass_classification",
        primary_metric="weighted_f1",
        display_name="Failure Type",
        model_set="multiclass",
    ),
}

# Compatibility shims for the existing binary training flow until later tasks
# switch the runtime stack to consume task-aware config directly.
TARGET_COLUMN = TASK_CONFIGS[DEFAULT_TASK_NAME].target_column
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

DATASET_PATH = PROJECT_ROOT / "data" / "industrial_machine_maintenance.csv"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"


def get_task_config(task_name: str) -> TaskConfig:
    try:
        task_config = TASK_CONFIGS[task_name]
    except KeyError as exc:
        raise ValueError(f"Unknown task: {task_name}") from exc

    if task_config.task_name != task_name:
        raise ValueError(
            f"Registry key/task_name mismatch: {task_name} != {task_config.task_name}"
        )

    return task_config


def build_artifact_paths(task_name: str) -> ArtifactPaths:
    task_config = get_task_config(task_name)
    suffix = task_config.task_name
    return ArtifactPaths(
        model_path=ARTIFACT_ROOT / "model" / f"best_model_{suffix}.joblib",
        metrics_path=ARTIFACT_ROOT / "metrics" / f"model_comparison_{suffix}.csv",
        importance_path=ARTIFACT_ROOT / "metrics" / f"feature_importance_{suffix}.csv",
    )


# Compatibility shims for the existing binary artifact names consumed by the
# current runtime path until later tasks switch to build_artifact_paths().
DEFAULT_ARTIFACT_PATHS = build_artifact_paths(DEFAULT_TASK_NAME)
MODEL_ARTIFACT_PATH = DEFAULT_ARTIFACT_PATHS.model_path
METRICS_ARTIFACT_PATH = DEFAULT_ARTIFACT_PATHS.metrics_path
IMPORTANCE_ARTIFACT_PATH = DEFAULT_ARTIFACT_PATHS.importance_path
