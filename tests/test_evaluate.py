import pytest

from predictive_maintenance.config import DEFAULT_TASK_NAME, TaskConfig
from predictive_maintenance.evaluate import compare_results, evaluate_classifier, select_best_model
from predictive_maintenance.models import build_candidate_models


def test_build_candidate_models_defaults_to_current_training_task() -> None:
    models = build_candidate_models()

    assert set(models) == set(build_candidate_models(task_name=DEFAULT_TASK_NAME))


def test_build_candidate_models_includes_four_required_estimators() -> None:
    models = build_candidate_models(task_name="failure_within_24h")

    assert set(models) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "mlp_classifier",
    }


def test_build_candidate_models_supports_multiclass_task() -> None:
    models = build_candidate_models(task_name="failure_type")

    assert set(models) == {
        "multinomial_logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "mlp_classifier",
    }


def test_build_candidate_models_rejects_unsupported_model_set(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_task_config(task_name: str) -> TaskConfig:
        return TaskConfig(
            task_name=task_name,
            target_column="target",
            problem_type="binary_classification",
            primary_metric="f1",
            display_name="Broken",
            model_set="unsupported",
        )

    monkeypatch.setattr("predictive_maintenance.models.get_task_config", fake_get_task_config)

    with pytest.raises(ValueError, match="Unsupported model_set: unsupported"):
        build_candidate_models(task_name="broken")


def test_select_best_model_uses_highest_f1() -> None:
    comparison = compare_results(
        task_name="failure_within_24h",
        results=[
            {
                "model_name": "a",
                "f1": 0.71,
                "precision": 0.7,
                "recall": 0.72,
                "roc_auc": 0.8,
                "pr_auc": 0.78,
            },
            {
                "model_name": "b",
                "f1": 0.79,
                "precision": 0.77,
                "recall": 0.81,
                "roc_auc": 0.84,
                "pr_auc": 0.82,
            },
        ]
    )

    best_row = select_best_model(comparison)

    assert best_row["model_name"] == "b"
    assert best_row["f1"] == 0.79


def test_compare_results_defaults_to_current_training_task() -> None:
    comparison = compare_results(
        results=[
            {"model_name": "a", "f1": 0.71, "precision": 0.7, "recall": 0.72},
            {"model_name": "b", "f1": 0.79, "precision": 0.77, "recall": 0.81},
        ]
    )

    assert comparison.iloc[0]["model_name"] == "b"


def test_evaluate_classifier_returns_none_auc_metrics_for_single_class_targets() -> None:
    results = evaluate_classifier(
        task_name="failure_within_24h",
        model_name="only_positive",
        true_values=[1, 1, 1],
        predicted_labels=[1, 1, 1],
        predicted_probabilities=[0.92, 0.87, 0.96],
    )

    assert results["model_name"] == "only_positive"
    assert results["f1"] == 1.0
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["roc_auc"] is None
    assert results["pr_auc"] is None


def test_evaluate_classifier_defaults_to_current_training_task() -> None:
    results = evaluate_classifier(
        model_name="default_task",
        true_values=[0, 1, 1, 0],
        predicted_labels=[0, 1, 1, 0],
        predicted_probabilities=[0.2, 0.9, 0.7, 0.1],
    )

    assert results["f1"] == 1.0
    assert results["roc_auc"] == 1.0


def test_evaluate_classifier_accepts_two_column_predict_proba_output() -> None:
    results = evaluate_classifier(
        task_name="failure_within_24h",
        model_name="two_column_scores",
        true_values=[0, 1, 1, 0],
        predicted_labels=[0, 1, 1, 0],
        predicted_probabilities=[
            [0.8, 0.2],
            [0.1, 0.9],
            [0.3, 0.7],
            [0.9, 0.1],
        ],
    )

    assert results == {
        "model_name": "two_column_scores",
        "f1": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "roc_auc": 1.0,
        "pr_auc": 1.0,
    }


def test_evaluate_classifier_supports_weighted_multiclass_metrics() -> None:
    metrics = evaluate_classifier(
        task_name="failure_type",
        model_name="multinomial_logistic_regression",
        true_values=["none", "mechanical", "thermal", "none"],
        predicted_labels=["none", "mechanical", "none", "none"],
        predicted_probabilities=[
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.5, 0.2, 0.3],
            [0.7, 0.2, 0.1],
        ],
    )

    assert metrics == {
        "model_name": "multinomial_logistic_regression",
        "weighted_f1": 0.65,
        "macro_f1": 0.6,
        "weighted_precision": 0.5833,
        "weighted_recall": 0.75,
        "class_count": 3,
    }


def test_evaluate_classifier_rejects_unsupported_problem_type(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_task_config(task_name: str) -> TaskConfig:
        return TaskConfig(
            task_name=task_name,
            target_column="target",
            problem_type="regression",
            primary_metric="rmse",
            display_name="Broken",
            model_set="binary",
        )

    monkeypatch.setattr("predictive_maintenance.evaluate.get_task_config", fake_get_task_config)

    with pytest.raises(ValueError, match="Unsupported problem_type: regression"):
        evaluate_classifier(
            task_name="broken",
            model_name="bad_model",
            true_values=[0, 1],
            predicted_labels=[0, 1],
            predicted_probabilities=[0.2, 0.8],
        )


def test_compare_results_uses_weighted_f1_for_failure_type() -> None:
    comparison = compare_results(
        task_name="failure_type",
        results=[
            {"model_name": "a", "weighted_f1": 0.71, "macro_f1": 0.65, "weighted_recall": 0.70},
            {"model_name": "b", "weighted_f1": 0.79, "macro_f1": 0.60, "weighted_recall": 0.76},
        ],
    )

    assert comparison.iloc[0]["model_name"] == "b"


def test_compare_results_rejects_unsupported_problem_type(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_task_config(task_name: str) -> TaskConfig:
        return TaskConfig(
            task_name=task_name,
            target_column="target",
            problem_type="regression",
            primary_metric="rmse",
            display_name="Broken",
            model_set="binary",
        )

    monkeypatch.setattr("predictive_maintenance.evaluate.get_task_config", fake_get_task_config)

    with pytest.raises(ValueError, match="Unsupported problem_type: regression"):
        compare_results(task_name="broken", results=[{"model_name": "a"}])
