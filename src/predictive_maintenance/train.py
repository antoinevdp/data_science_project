import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from predictive_maintenance.artifacts import save_frame, save_model
from predictive_maintenance.config import (
    DATASET_PATH,
    DEFAULT_TASK_NAME,
    RANDOM_STATE,
    TASK_CONFIGS,
    build_artifact_paths,
    get_task_config,
)
from predictive_maintenance.data import load_dataset, split_features_and_target
from predictive_maintenance.evaluate import compare_results, evaluate_classifier, select_best_model
from predictive_maintenance.explainability import compute_permutation_importance_frame
from predictive_maintenance.models import build_candidate_models
from predictive_maintenance.preprocessing import build_preprocessor, make_train_test_split


def train_and_compare(
    csv_path: Path = DATASET_PATH,
    task_name: str = DEFAULT_TASK_NAME,
    artifact_root: Path | None = None,
) -> dict[str, object]:
    task_config = get_task_config(task_name)
    dataset = load_dataset(csv_path)
    features, target = split_features_and_target(dataset, task_name=task_name)
    test_size = max(0.2, target.nunique() / len(target))
    x_train, x_test, y_train, y_test = make_train_test_split(features, target, test_size=test_size)

    candidate_models = build_candidate_models(task_name=task_name)
    trained_pipelines: dict[str, Pipeline] = {}
    evaluation_rows: list[dict[str, float | str | None]] = []

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predicted_labels = pipeline.predict(x_test)
        predicted_probabilities = pipeline.predict_proba(x_test)
        trained_pipelines[model_name] = pipeline
        evaluation_rows.append(
            evaluate_classifier(
                task_name=task_name,
                model_name=model_name,
                true_values=y_test,
                predicted_labels=predicted_labels,
                predicted_probabilities=predicted_probabilities,
            )
        )

    comparison = compare_results(task_name=task_name, results=evaluation_rows)
    best_row = select_best_model(comparison)
    best_model_name = str(best_row["model_name"])
    best_pipeline = trained_pipelines[best_model_name]

    artifact_paths = build_artifact_paths(task_name)
    resolved_artifact_root = artifact_root or artifact_paths.model_path.parents[1]
    model_path = resolved_artifact_root / artifact_paths.model_path.relative_to(artifact_paths.model_path.parents[1])
    comparison_path = (
        resolved_artifact_root / artifact_paths.metrics_path.relative_to(artifact_paths.metrics_path.parents[1])
    )
    importance_path = (
        resolved_artifact_root / artifact_paths.importance_path.relative_to(artifact_paths.importance_path.parents[1])
    )
    if task_config.problem_type == "multiclass_classification":
        importance_result = permutation_importance(
            estimator=best_pipeline,
            X=x_test,
            y=y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="f1_weighted",
        )
        importance_frame = pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": importance_result.importances_mean,
                "importance_std": importance_result.importances_std,
            }
        ).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)
    else:
        importance_frame = compute_permutation_importance_frame(best_pipeline, x_test, y_test)

    save_model(best_pipeline, model_path)
    save_frame(comparison, comparison_path)
    save_frame(importance_frame, importance_path)

    return {
        "task_name": task_config.task_name,
        "best_model_name": best_model_name,
        "model_path": model_path,
        "comparison_path": comparison_path,
        "importance_path": importance_path,
        "x_test": x_test,
        "y_test": y_test,
        "best_pipeline": best_pipeline,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", choices=sorted(TASK_CONFIGS), default=DEFAULT_TASK_NAME)
    return parser.parse_args([] if argv is None else argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train_and_compare(task_name=args.task_name)
    print(f"Task: {summary['task_name']}")
    print(f"Best model: {summary['best_model_name']}")
    print(f"Saved model: {summary['model_path']}")
    print(f"Saved metrics: {summary['comparison_path']}")


if __name__ == "__main__":
    main(sys.argv[1:])
