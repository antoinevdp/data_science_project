import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from predictive_maintenance.config import DEFAULT_TASK_NAME, get_task_config


def _extract_positive_class_scores(predicted_probabilities) -> np.ndarray:
    scores = np.asarray(predicted_probabilities)

    if scores.ndim == 1:
        return scores

    if scores.ndim == 2 and scores.shape[1] == 2:
        return scores[:, 1]

    raise ValueError(
        "predicted_probabilities must be a 1D positive-class score vector or a 2D array with two class columns"
    )


def evaluate_classifier(
    model_name: str,
    true_values,
    predicted_labels,
    predicted_probabilities,
    task_name: str = DEFAULT_TASK_NAME,
) -> dict[str, float | str | None]:
    task_config = get_task_config(task_name)

    if task_config.problem_type == "binary_classification":
        positive_class_scores = _extract_positive_class_scores(predicted_probabilities)
        metrics: dict[str, float | str | None] = {
            "model_name": model_name,
            "f1": round(f1_score(true_values, predicted_labels, zero_division=0), 4),
            "precision": round(precision_score(true_values, predicted_labels, zero_division=0), 4),
            "recall": round(recall_score(true_values, predicted_labels, zero_division=0), 4),
            "roc_auc": None,
            "pr_auc": None,
        }

        if len(np.unique(true_values)) < 2:
            return metrics

        metrics["roc_auc"] = round(roc_auc_score(true_values, positive_class_scores), 4)
        metrics["pr_auc"] = round(average_precision_score(true_values, positive_class_scores), 4)
        return metrics

    if task_config.problem_type == "multiclass_classification":
        probability_matrix = np.asarray(predicted_probabilities)
        metrics = {
            "model_name": model_name,
            "weighted_f1": round(f1_score(true_values, predicted_labels, average="weighted", zero_division=0), 4),
            "macro_f1": round(f1_score(true_values, predicted_labels, average="macro", zero_division=0), 4),
            "weighted_precision": round(
                precision_score(true_values, predicted_labels, average="weighted", zero_division=0), 4
            ),
            "weighted_recall": round(
                recall_score(true_values, predicted_labels, average="weighted", zero_division=0), 4
            ),
            "class_count": probability_matrix.shape[1],
        }
        return metrics

    raise ValueError(f"Unsupported problem_type: {task_config.problem_type}")


def compare_results(results: list[dict[str, float | str]], task_name: str = DEFAULT_TASK_NAME) -> pd.DataFrame:
    task_config = get_task_config(task_name)
    if task_config.problem_type == "binary_classification":
        sort_columns = ["f1", "recall", "precision"]
        return pd.DataFrame(results).sort_values(by=sort_columns, ascending=False).reset_index(drop=True)

    if task_config.problem_type == "multiclass_classification":
        sort_columns = ["weighted_f1", "macro_f1", "weighted_recall"]
        return pd.DataFrame(results).sort_values(by=sort_columns, ascending=False).reset_index(drop=True)

    raise ValueError(f"Unsupported problem_type: {task_config.problem_type}")


def select_best_model(comparison: pd.DataFrame) -> pd.Series:
    return comparison.iloc[0]
