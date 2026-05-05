from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from predictive_maintenance.config import DEFAULT_TASK_NAME, RANDOM_STATE, get_task_config


def build_candidate_models(task_name: str = DEFAULT_TASK_NAME) -> dict[str, object]:
    task_config = get_task_config(task_name)

    if task_config.model_set == "binary":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            "gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "mlp_classifier": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=RANDOM_STATE,
            ),
        }

    if task_config.model_set == "multiclass":
        return {
            "multinomial_logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "mlp_classifier": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=RANDOM_STATE,
            ),
        }

    raise ValueError(f"Unsupported model_set: {task_config.model_set}")
