import pandas as pd
from sklearn.inspection import permutation_importance

from predictive_maintenance.config import RANDOM_STATE


def compute_permutation_importance_frame(model, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    result = permutation_importance(
        estimator=model,
        X=features,
        y=target,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1",
    )
    frame = pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)


def summarize_top_features(importance_frame: pd.DataFrame, top_n: int = 3) -> list[str]:
    ranked_frame = importance_frame.sort_values(by="importance_mean", ascending=False)
    return ranked_frame.head(top_n)["feature"].tolist()
