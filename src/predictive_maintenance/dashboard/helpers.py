from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from predictive_maintenance.config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    TASK_CONFIGS,
)


HIGH_RISK_LABEL = "High risk within 24h"
LOW_RISK_LABEL = "Low risk within 24h"


def build_input_frame(raw_values: dict[str, float | str]) -> pd.DataFrame:
    return pd.DataFrame(
        [[raw_values[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS
    )


def required_artifacts_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)


def extract_operating_mode_options(model_or_preprocessor: Pipeline | ColumnTransformer) -> list[str]:
    preprocessor = model_or_preprocessor
    if isinstance(model_or_preprocessor, Pipeline):
        preprocessor = model_or_preprocessor.named_steps["preprocessor"]

    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    operating_mode_index = CATEGORICAL_COLUMNS.index("operating_mode")
    return encoder.categories_[operating_mode_index].tolist()


def summarize_probability(
    probability: float, threshold: float = 0.5
) -> tuple[str, str]:
    label = HIGH_RISK_LABEL if probability >= threshold else LOW_RISK_LABEL
    message = f"Predicted failure probability: {probability:.2f}"
    return label, message


def get_prediction_task_options() -> dict[str, str]:
    options: dict[str, str] = {}
    for task in TASK_CONFIGS.values():
        if task.display_name in options:
            raise ValueError(f"Duplicate task display name: {task.display_name}")
        options[task.display_name] = task.task_name
    return options


def format_multiclass_probabilities(
    class_labels: list[str], probabilities: list[float]
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"class_label": class_labels, "probability": probabilities}
    )
    return frame.sort_values(by="probability", ascending=False).reset_index(drop=True)
