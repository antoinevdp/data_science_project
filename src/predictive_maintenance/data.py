from pathlib import Path

import pandas as pd

from predictive_maintenance.config import DEFAULT_TASK_NAME, FEATURE_COLUMNS, get_task_config


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    return pd.read_csv(csv_path)


def validate_required_columns(dataset: pd.DataFrame, task_name: str) -> None:
    task_config = get_task_config(task_name)
    required_columns = FEATURE_COLUMNS + [task_config.target_column]
    missing_columns = sorted(set(required_columns) - set(dataset.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_text}")


def validate_cross_target_consistency(dataset: pd.DataFrame) -> None:
    if "failure_within_24h" not in dataset.columns or "failure_type" not in dataset.columns:
        return

    inconsistent_rows = ((dataset["failure_within_24h"] == 0) & (dataset["failure_type"] != "none")) | (
        (dataset["failure_within_24h"] == 1) & (dataset["failure_type"] == "none")
    )
    if inconsistent_rows.any():
        raise ValueError(
            "Inconsistent failure labels: failure_within_24h and failure_type must agree"
        )


def validate_target_values(target: pd.Series, task_name: str) -> None:
    task_config = get_task_config(task_name)

    if target.isna().any():
        raise ValueError(f"Target column '{task_config.target_column}' must not contain missing values")

    if task_config.problem_type == "binary_classification" and not target.isin([0, 1]).all():
        raise ValueError(
            f"Target column '{task_config.target_column}' must contain only binary values 0 and 1"
        )


def validate_target_trainability(target: pd.Series, task_name: str) -> None:
    task_config = get_task_config(task_name)
    class_counts = target.value_counts().sort_index()

    if len(class_counts) < 2:
        raise ValueError(
            f"Target column '{task_config.target_column}' must contain at least 2 classes"
        )

    if (class_counts < 2).any():
        raise ValueError(
            f"Target column '{task_config.target_column}' must have at least 2 rows per class"
        )


def split_features_and_target(
    dataset: pd.DataFrame, task_name: str = DEFAULT_TASK_NAME
) -> tuple[pd.DataFrame, pd.Series]:
    task_config = get_task_config(task_name)
    validate_required_columns(dataset, task_name)
    validate_cross_target_consistency(dataset)
    target = dataset[task_config.target_column].copy()
    validate_target_values(target, task_name)
    validate_target_trainability(target, task_name)
    features = dataset[FEATURE_COLUMNS].copy()
    return features, target
