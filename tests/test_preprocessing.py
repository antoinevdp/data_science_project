import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.preprocessing import build_preprocessor, make_train_test_split


def test_make_train_test_split_preserves_row_count() -> None:
    features = pd.DataFrame(
        {
            "vibration_rms": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "temperature_motor": [50, 55, 60, 65, 70, 75],
            "rpm": [1000, 1100, 1200, 1300, 1400, 1500],
            "pressure_level": [30.0, 30.5, 31.0, 31.5, 32.0, 32.5],
            "rul_hours": [20.0, 18.0, 16.0, 14.0, 12.0, 10.0],
            "operating_mode": ["normal", "normal", "stress", "stress", "normal", "stress"],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1])

    x_train, x_test, y_train, y_test = make_train_test_split(features, target, test_size=0.5)

    assert len(x_train) + len(x_test) == len(features)
    assert len(y_train) + len(y_test) == len(target)
    assert y_train.value_counts().sort_index().tolist() == [1, 2]
    assert y_test.value_counts().sort_index().tolist() == [2, 1]


def test_build_preprocessor_can_fit_and_transform() -> None:
    training_features = pd.DataFrame(
        {
            "vibration_rms": [1.0, None, 2.0, 2.5],
            "temperature_motor": [50, 55, None, 65],
            "rpm": [1000, 1100, 1200, None],
            "pressure_level": [30.0, 30.5, 31.0, None],
            "rul_hours": [20.0, 18.0, None, 10.0],
            "operating_mode": ["normal", None, "stress", "stress"],
        }
    )
    evaluation_features = pd.DataFrame(
        {
            "vibration_rms": [1.25],
            "temperature_motor": [58.0],
            "rpm": [1150],
            "pressure_level": [30.75],
            "rul_hours": [17.0],
            "operating_mode": ["maintenance"],
        }
    )

    preprocessor = build_preprocessor()

    transformed_train = preprocessor.fit_transform(training_features)
    transformed_eval = preprocessor.transform(evaluation_features)

    assert transformed_train.shape == (4, 7)
    assert transformed_eval.shape == (1, 7)
    assert np.isnan(transformed_train).sum() == 0
    assert np.isnan(transformed_eval).sum() == 0
    assert preprocessor.get_feature_names_out().tolist() == [
        "numeric__vibration_rms",
        "numeric__temperature_motor",
        "numeric__rpm",
        "numeric__pressure_level",
        "numeric__rul_hours",
        "categorical__operating_mode_normal",
        "categorical__operating_mode_stress",
    ]
    assert transformed_eval[0, -2:].tolist() == [0.0, 0.0]


def test_make_train_test_split_rejects_test_fold_smaller_than_class_count() -> None:
    features = pd.DataFrame(
        {
            "vibration_rms": [1.0, 1.1, 2.0, 2.1],
            "temperature_motor": [50.0, 51.0, 60.0, 61.0],
            "rpm": [1000.0, 1010.0, 1200.0, 1210.0],
            "pressure_level": [30.0, 30.1, 31.0, 31.1],
            "rul_hours": [20.0, 19.0, 10.0, 9.0],
            "operating_mode": ["normal", "normal", "stress", "stress"],
        }
    )
    target = pd.Series([0, 0, 1, 1])

    with pytest.raises(
        ValueError,
        match="Stratified split requires a test fold with at least 2 rows for 2 classes; got 1 rows with test_size=0.2",
    ):
        make_train_test_split(features, target, test_size=0.2)
