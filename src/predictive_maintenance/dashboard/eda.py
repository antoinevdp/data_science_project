from pathlib import Path
from math import ceil

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from predictive_maintenance.config import DATASET_PATH, NUMERIC_COLUMNS


EDA_REQUIRED_COLUMNS = set(
    NUMERIC_COLUMNS + ["operating_mode", "failure_within_24h", "failure_type"]
)


def _coerce_numeric_column(dataset: pd.DataFrame, column: str) -> pd.Series:
    original_values = dataset[column]
    numeric_values = pd.to_numeric(original_values, errors="coerce")

    invalid_mask = numeric_values.isna() & original_values.notna()
    if invalid_mask.any():
        raise ValueError(f"Dataset column '{column}' has invalid numeric values")

    non_finite_mask = numeric_values.notna() & ~np.isfinite(numeric_values.to_numpy())
    if non_finite_mask.any():
        raise ValueError(f"Dataset column '{column}' contains non-finite numeric values")

    return numeric_values


def _coerce_binary_target(dataset: pd.DataFrame) -> pd.Series:
    original_values = dataset["failure_within_24h"]

    if original_values.empty:
        return pd.Series(dtype="int64", index=original_values.index)

    failure_values = pd.to_numeric(original_values, errors="coerce")

    invalid_mask = failure_values.isna() & original_values.notna()
    if invalid_mask.any():
        raise ValueError("Dataset column 'failure_within_24h' must be numeric and binary")

    non_finite_mask = failure_values.notna() & ~np.isfinite(failure_values.to_numpy())
    if non_finite_mask.any():
        raise ValueError("Dataset column 'failure_within_24h' must be numeric and binary")

    if failure_values.isna().any():
        raise ValueError("Dataset column 'failure_within_24h' must not contain missing values")

    if not failure_values.isin([0, 1]).all():
        raise ValueError(
            "Dataset column 'failure_within_24h' must contain only binary values 0 and 1"
        )
    return failure_values.astype(int)


def _sanitize_failure_type(dataset: pd.DataFrame) -> pd.Series:
    failure_type = dataset["failure_type"]

    if failure_type.empty:
        return pd.Series(dtype="string", index=failure_type.index)

    non_null_values = failure_type.dropna()
    if not non_null_values.map(lambda value: isinstance(value, str)).all():
        raise ValueError("Dataset column 'failure_type' must be non-blank text")

    failure_type = failure_type.str.strip()
    if failure_type.isna().any() or (failure_type == "").any():
        raise ValueError("Dataset column 'failure_type' must be non-blank text")
    return failure_type


def _sanitize_required_numeric_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    sanitized = dataset.copy()
    for column in NUMERIC_COLUMNS:
        sanitized[column] = _coerce_numeric_column(sanitized, column)

    return sanitized


def _validate_eda_summary_input(dataset: pd.DataFrame) -> pd.DataFrame:
    sanitized = dataset.copy()
    sanitized["failure_within_24h"] = _coerce_binary_target(sanitized)
    return _sanitize_required_numeric_columns(sanitized)


def load_eda_dataset(csv_path: Path = DATASET_PATH) -> pd.DataFrame:
    dataset = pd.read_csv(csv_path)
    missing_columns = sorted(EDA_REQUIRED_COLUMNS - set(dataset.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required EDA columns: {', '.join(missing_columns)}"
        )

    sanitized = _validate_eda_summary_input(dataset)
    sanitized["failure_type"] = _sanitize_failure_type(dataset)
    return sanitized


def build_eda_summary(dataset: pd.DataFrame) -> dict[str, object]:
    sanitized = _validate_eda_summary_input(dataset)
    missing_summary = {
        column: int(count)
        for column, count in dataset.isna().sum().items()
        if int(count) > 0
    }
    positive_failure_rate = 0.0
    if len(sanitized) > 0:
        positive_failure_rate = round(float(sanitized["failure_within_24h"].mean() * 100), 2)
    return {
        "row_count": int(len(sanitized)),
        "column_count": int(sanitized.shape[1]),
        "missing_summary": missing_summary,
        "positive_failure_rate": positive_failure_rate,
    }


def build_target_distribution_figure(dataset: pd.DataFrame):
    sanitized = dataset.copy()
    sanitized["failure_within_24h"] = _coerce_binary_target(sanitized)
    sanitized["failure_type"] = _sanitize_failure_type(sanitized)
    binary_counts = (
        sanitized["failure_within_24h"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )
    failure_type_counts = sanitized["failure_type"].value_counts()

    figure = make_subplots(rows=1, cols=2, subplot_titles=("failure_within_24h", "failure_type"))
    figure.add_trace(
        go.Bar(x=binary_counts.index.astype(str), y=binary_counts.values, name="failure_within_24h"),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(x=failure_type_counts.index.astype(str), y=failure_type_counts.values, name="failure_type"),
        row=1,
        col=2,
    )
    figure.update_layout(title="Target distributions", showlegend=False)
    return figure


def build_numeric_histogram_figure(dataset: pd.DataFrame):
    sanitized = _sanitize_required_numeric_columns(dataset)
    columns_per_row = max(1, min(3, len(NUMERIC_COLUMNS)))
    rows = ceil(len(NUMERIC_COLUMNS) / columns_per_row)
    figure = make_subplots(rows=rows, cols=columns_per_row, subplot_titles=NUMERIC_COLUMNS)
    for index, column in enumerate(NUMERIC_COLUMNS):
        row = (index // columns_per_row) + 1
        col = (index % columns_per_row) + 1
        figure.add_trace(go.Histogram(x=sanitized[column], name=column), row=row, col=col)

    figure.update_layout(title="Numeric feature distributions", showlegend=False)
    return figure


def build_correlation_heatmap_figure(dataset: pd.DataFrame):
    sanitized = _validate_eda_summary_input(dataset)
    correlation_frame = sanitized[NUMERIC_COLUMNS + ["failure_within_24h"]].corr(numeric_only=True)
    figure = px.imshow(correlation_frame, aspect="auto", color_continuous_scale="Blues")
    figure.update_layout(title="Correlation heatmap")
    return figure


def build_failure_boxplot_figure(dataset: pd.DataFrame):
    sanitized = _validate_eda_summary_input(dataset)
    figure = make_subplots(
        rows=len(NUMERIC_COLUMNS),
        cols=1,
        shared_xaxes=True,
        subplot_titles=NUMERIC_COLUMNS,
        vertical_spacing=0.04,
    )
    for index, feature in enumerate(NUMERIC_COLUMNS, start=1):
        figure.add_trace(
            go.Box(
                x=sanitized["failure_within_24h"].astype(str),
                y=sanitized[feature],
                name=feature,
                boxpoints=False,
                showlegend=False,
            ),
            row=index,
            col=1,
        )
    figure.update_layout(title="Feature spread by failure flag")
    return figure
