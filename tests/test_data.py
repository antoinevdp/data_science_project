from pathlib import Path

import pandas as pd
import pytest

from predictive_maintenance.data import load_dataset, split_features_and_target


def test_load_dataset_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_dataset(missing_path)


def test_split_features_and_target_returns_expected_shapes(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    frame = pd.DataFrame(
        {
            "vibration_rms": [1.1, 1.4, 2.2, 2.5],
            "temperature_motor": [55.0, 57.0, 65.0, 67.0],
            "rpm": [1200, 1300, 1500, 1600],
            "pressure_level": [30.0, 30.8, 31.5, 32.1],
            "rul_hours": [12.0, 10.0, 8.0, 6.0],
            "operating_mode": ["normal", "normal", "stress", "stress"],
            "failure_within_24h": [0, 0, 1, 1],
        }
    )
    frame.to_csv(csv_path, index=False)

    dataset = load_dataset(csv_path)
    features, target = split_features_and_target(dataset)

    assert list(features.columns) == [
        "vibration_rms",
        "temperature_motor",
        "rpm",
        "pressure_level",
        "rul_hours",
        "operating_mode",
    ]
    assert target.tolist() == [0, 0, 1, 1]


def test_split_features_and_target_rejects_missing_failure_type_labels() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 1.2, 1.4, 1.6],
            "temperature_motor": [50, 55, 60, 65],
            "rpm": [1000, 1050, 1100, 1150],
            "pressure_level": [30.0, 30.2, 30.4, 30.6],
            "rul_hours": [15, 14, 13, 12],
            "operating_mode": ["normal", "stress", "normal", "stress"],
            "failure_within_24h": [0, 1, 1, 0],
            "failure_type": ["none", None, "thermal", "none"],
        }
    )

    with pytest.raises(ValueError, match="Target column 'failure_type' must not contain missing values"):
        split_features_and_target(dataset, task_name="failure_type")


def test_split_features_and_target_raises_for_missing_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing-columns.csv"
    frame = pd.DataFrame(
        {
            "vibration_rms": [1.1],
            "temperature_motor": [55.0],
            "rpm": [1200],
            "pressure_level": [30.0],
            "rul_hours": [12.0],
            "failure_within_24h": [0],
        }
    )
    frame.to_csv(csv_path, index=False)

    dataset = load_dataset(csv_path)

    with pytest.raises(ValueError, match="Dataset is missing required columns: operating_mode"):
        split_features_and_target(dataset, task_name="failure_within_24h")


def test_split_features_and_target_raises_for_missing_feature_columns() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.1],
            "temperature_motor": [55.0],
            "rpm": [1200],
            "pressure_level": [30.0],
            "rul_hours": [12.0],
            "failure_within_24h": [0],
        }
    )

    with pytest.raises(ValueError, match="Dataset is missing required columns: operating_mode"):
        split_features_and_target(dataset, task_name="failure_within_24h")


def test_split_features_and_target_raises_for_invalid_target_values() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.1, 2.2],
            "temperature_motor": [55.0, 65.0],
            "rpm": [1200, 1500],
            "pressure_level": [30.0, 31.5],
            "rul_hours": [12.0, 8.0],
            "operating_mode": ["normal", "stress"],
            "failure_within_24h": [0, 2],
        }
    )

    with pytest.raises(ValueError, match="Target column 'failure_within_24h' must contain only binary values 0 and 1"):
        split_features_and_target(dataset, task_name="failure_within_24h")


def test_split_features_and_target_rejects_single_class_target() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.1, 2.2],
            "temperature_motor": [55.0, 65.0],
            "rpm": [1200, 1500],
            "pressure_level": [30.0, 31.5],
            "rul_hours": [12.0, 8.0],
            "operating_mode": ["normal", "stress"],
            "failure_within_24h": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="Target column 'failure_within_24h' must contain at least 2 classes"):
        split_features_and_target(dataset, task_name="failure_within_24h")


def test_split_features_and_target_rejects_class_with_single_row() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.1, 2.2, 3.3],
            "temperature_motor": [55.0, 65.0, 75.0],
            "rpm": [1200, 1500, 1800],
            "pressure_level": [30.0, 31.5, 33.0],
            "rul_hours": [12.0, 8.0, 4.0],
            "operating_mode": ["normal", "stress", "stress"],
            "failure_within_24h": [0, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="Target column 'failure_within_24h' must have at least 2 rows per class"):
        split_features_and_target(dataset, task_name="failure_within_24h")


def test_split_features_and_target_supports_failure_type() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            "temperature_motor": [50, 55, 60, 65, 70, 75],
            "rpm": [1000, 1050, 1100, 1150, 1200, 1250],
            "pressure_level": [30.0, 30.2, 30.4, 30.6, 30.8, 31.0],
            "rul_hours": [15, 14, 13, 12, 11, 10],
            "operating_mode": ["normal", "stress", "normal", "stress", "normal", "stress"],
            "failure_within_24h": [0, 1, 1, 0, 1, 1],
            "failure_type": ["none", "mechanical", "thermal", "none", "mechanical", "thermal"],
        }
    )

    _, target = split_features_and_target(dataset, task_name="failure_type")

    assert target.tolist() == ["none", "mechanical", "thermal", "none", "mechanical", "thermal"]


def test_split_features_and_target_rejects_inconsistent_failure_labels() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 1.2, 1.4, 1.6],
            "temperature_motor": [50, 55, 60, 65],
            "rpm": [1000, 1050, 1100, 1150],
            "pressure_level": [30.0, 30.2, 30.4, 30.6],
            "rul_hours": [15, 14, 13, 12],
            "operating_mode": ["normal", "stress", "normal", "stress"],
            "failure_within_24h": [0, 1, 1, 0],
            "failure_type": ["none", "mechanical", "thermal", "mechanical"],
        }
    )

    with pytest.raises(ValueError, match="Inconsistent failure labels"):
        split_features_and_target(dataset, task_name="failure_type")


def test_split_features_and_target_rejects_none_failure_type_for_positive_failure_flag() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 1.2, 1.4, 1.6],
            "temperature_motor": [50, 55, 60, 65],
            "rpm": [1000, 1050, 1100, 1150],
            "pressure_level": [30.0, 30.2, 30.4, 30.6],
            "rul_hours": [15, 14, 13, 12],
            "operating_mode": ["normal", "stress", "normal", "stress"],
            "failure_within_24h": [0, 1, 1, 1],
            "failure_type": ["none", "mechanical", "thermal", "none"],
        }
    )

    with pytest.raises(ValueError, match="Inconsistent failure labels"):
        split_features_and_target(dataset, task_name="failure_type")


def test_failure_type_requires_at_least_two_rows_per_class() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 1.2, 1.4],
            "temperature_motor": [50, 55, 60],
            "rpm": [1000, 1050, 1100],
            "pressure_level": [30.0, 30.2, 30.4],
            "rul_hours": [15, 14, 13],
            "operating_mode": ["normal", "stress", "normal"],
            "failure_within_24h": [0, 1, 1],
            "failure_type": ["none", "mechanical", "thermal"],
        }
    )

    with pytest.raises(ValueError, match="at least 2 rows per class"):
        split_features_and_target(dataset, task_name="failure_type")
