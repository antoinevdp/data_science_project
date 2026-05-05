import math

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_maintenance.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def validate_stratified_split_capacity(target: pd.Series, test_size: float) -> None:
    class_count = int(target.nunique())
    test_row_count = math.ceil(len(target) * test_size)

    if test_row_count < class_count:
        raise ValueError(
            "Stratified split requires a test fold with at least "
            f"{class_count} rows for {class_count} classes; got {test_row_count} rows "
            f"with test_size={test_size}"
        )


def make_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    validate_stratified_split_capacity(target, test_size)

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=target,
    )
