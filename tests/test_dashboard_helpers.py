import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import predictive_maintenance.dashboard.helpers as dashboard_helpers
import predictive_maintenance.dashboard.eda as dashboard_eda
import predictive_maintenance.config as config_module
from predictive_maintenance.config import build_artifact_paths
from predictive_maintenance.config import API_BASE_URL
from predictive_maintenance.config import NUMERIC_COLUMNS
from predictive_maintenance.config import TASK_CONFIGS
from predictive_maintenance.dashboard.helpers import (
    HIGH_RISK_LABEL,
    LOW_RISK_LABEL,
    build_input_frame,
    extract_operating_mode_options,
    format_multiclass_probabilities,
    get_prediction_task_options,
    required_artifacts_exist,
    summarize_probability,
)
from predictive_maintenance.dashboard.eda import (
    build_correlation_heatmap_figure,
    build_eda_summary,
    build_failure_boxplot_figure,
    build_numeric_histogram_figure,
    build_target_distribution_figure,
    load_eda_dataset,
)
from predictive_maintenance.preprocessing import build_preprocessor


class _DashboardColumnContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def metric(self, *_args, **_kwargs):
        return None


def _dashboard_columns(spec):
    count = spec if isinstance(spec, int) else len(spec)
    return tuple(_DashboardColumnContext() for _ in range(count))


def _prediction_mode_selectbox(label, options):
    if label == "Dashboard mode":
        return "Prediction"
    if label == "Prediction task":
        return "Failure Type"
    pytest.fail(f"unexpected selectbox label: {label}")


def _eda_mode_selectbox(label, options):
    if label == "Dashboard mode":
        return "EDA"
    pytest.fail(f"unexpected selectbox label: {label}")


def test_api_base_url_default_is_defined() -> None:
    assert API_BASE_URL.startswith("http://")


def test_api_base_url_can_be_overridden_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "PREDICTIVE_MAINTENANCE_API_BASE_URL", "https://api.example.test"
    )
    importlib.reload(config_module)

    try:
        assert config_module.API_BASE_URL == "https://api.example.test"
    finally:
        monkeypatch.delenv("PREDICTIVE_MAINTENANCE_API_BASE_URL", raising=False)
        importlib.reload(config_module)
        assert config_module.API_BASE_URL == "http://127.0.0.1:8000"


def test_dashboard_app_normalizes_api_base_url_with_prefix_and_trailing_slash(
    monkeypatch,
) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    json_payloads: list[dict[str, object]] = []
    captions: list[str] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = "unused"

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, timeout: float, params: dict[str, object] | None = None):
        assert url == "https://api.example.test/prefix/model-info"
        assert timeout == 5
        assert params == {"task_name": "failure_type"}
        return FakeResponse(
            200,
            {
                "primary_metric_name": "weighted_f1",
                "best_model_name": "multiclass-model",
                "models_comparison": [
                    {
                        "model_name": "multiclass-model",
                        "primary_metric_value": 0.91,
                        "secondary_metric_value": 0.13,
                    },
                    {
                        "model_name": "binary-model",
                        "primary_metric_value": 0.84,
                        "secondary_metric_value": 0.2,
                    },
                ]
            },
        )

    def fake_post(url: str, json: dict[str, object], timeout: float):
        assert url == "https://api.example.test/prefix/predict/failure-type"
        assert timeout == 5
        return FakeResponse(
            200,
            {
                "task_name": "failure_type",
                "model_name": "multiclass-model",
                "predicted_class": "mechanical",
                "class_probabilities": [
                    {"class_label": "mechanical", "probability": 0.7},
                    {"class_label": "thermal", "probability": 0.3},
                ],
                "importance_summary": ["rpm", "temperature_motor"],
            },
        )

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=_prediction_mode_selectbox,
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda payload: json_payloads.append(payload),
        caption=lambda message: captions.append(message),
        error=lambda *_: None,
        stop=lambda: pytest.fail("dashboard should not stop on the happy path"),
        info=lambda *_: None,
        write=lambda *_: None,
        dataframe=lambda *_: None,
        bar_chart=lambda *_: None,
    )

    monkeypatch.setattr(config_module, "API_BASE_URL", "https://api.example.test/prefix/")
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(get=fake_get, post=fake_post, RequestException=Exception),
    )
    sys.modules.pop(app_module_name, None)

    importlib.import_module(app_module_name)

    assert json_payloads == [
        {
            "task_name": "failure_type",
            "model_name": "multiclass-model",
            "predicted_class": "mechanical",
            "class_probabilities": [
                {"class_label": "mechanical", "probability": 0.7},
                {"class_label": "thermal", "probability": 0.3},
            ],
            "importance_summary": ["rpm", "temperature_motor"],
        }
    ]
    assert captions == ["Features used: vibration_rms, temperature_motor, rpm, pressure_level, rul_hours, operating_mode"]

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_build_input_frame_preserves_feature_order() -> None:
    raw_values = {
        "vibration_rms": 2.1,
        "temperature_motor": 74.0,
        "rpm": 1550,
        "pressure_level": 34.5,
        "rul_hours": 8.0,
        "operating_mode": "stress",
    }
    frame = build_input_frame(
        raw_values
    )

    assert frame.columns.tolist() == [
        "vibration_rms",
        "temperature_motor",
        "rpm",
        "pressure_level",
        "rul_hours",
        "operating_mode",
    ]
    assert frame.iloc[0].to_dict() == raw_values


def test_load_eda_dataset_reads_csv_and_validates_required_columns(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 2.0],
            "temperature_motor": [50.0, 60.0],
            "rpm": [1000.0, 1200.0],
            "pressure_level": [30.0, 35.0],
            "rul_hours": [10.0, 8.0],
            "operating_mode": ["normal", "peak"],
            "failure_within_24h": [0, 1],
            "failure_type": ["none", "bearing"],
        }
    )
    csv_path = tmp_path / "eda.csv"
    dataset.to_csv(csv_path, index=False)

    loaded = load_eda_dataset(csv_path)

    assert loaded.columns.tolist() == dataset.columns.tolist()
    assert loaded["failure_within_24h"].tolist() == [0, 1]
    assert loaded["failure_type"].tolist() == ["none", "bearing"]
    assert all(pd.api.types.is_numeric_dtype(loaded[column]) for column in NUMERIC_COLUMNS)


def test_load_eda_dataset_coerces_numeric_and_binary_columns(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": ["1.0", "2.0"],
            "temperature_motor": [50, 60],
            "rpm": [1000, 1200],
            "pressure_level": [30.0, 35.0],
            "rul_hours": [10.0, 8.0],
            "operating_mode": ["normal", "peak"],
            "failure_within_24h": ["0", "1"],
            "failure_type": ["none", "bearing"],
        }
    )
    csv_path = tmp_path / "eda.csv"
    dataset.to_csv(csv_path, index=False)

    loaded = load_eda_dataset(csv_path)

    assert loaded["failure_within_24h"].tolist() == [0, 1]
    assert all(pd.api.types.is_numeric_dtype(loaded[column]) for column in NUMERIC_COLUMNS)


def test_load_eda_dataset_preserves_genuine_missing_numeric_values(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, None],
            "temperature_motor": [50.0, 60.0],
            "rpm": [1000.0, 1200.0],
            "pressure_level": [30.0, 35.0],
            "rul_hours": [10.0, 8.0],
            "operating_mode": ["normal", "peak"],
            "failure_within_24h": [0, 1],
            "failure_type": ["none", "bearing"],
        }
    )
    csv_path = tmp_path / "eda.csv"
    dataset.to_csv(csv_path, index=False)

    loaded = load_eda_dataset(csv_path)

    assert pd.isna(loaded.loc[1, "vibration_rms"])
    assert loaded["failure_within_24h"].tolist() == [0, 1]


def test_load_eda_dataset_rejects_missing_required_columns(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0],
            "temperature_motor": [50.0],
            "rpm": [1000.0],
            "pressure_level": [30.0],
            "rul_hours": [10.0],
            "operating_mode": ["normal"],
            "failure_within_24h": [0],
        }
    )
    csv_path = tmp_path / "eda.csv"
    dataset.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="failure_type"):
        load_eda_dataset(csv_path)


def test_load_eda_dataset_rejects_invalid_target_values(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 2.0],
            "temperature_motor": [50.0, 60.0],
            "rpm": [1000.0, 1200.0],
            "pressure_level": [30.0, 35.0],
            "rul_hours": [10.0, 8.0],
            "operating_mode": ["normal", "peak"],
                "failure_within_24h": [0, "bad"],
            "failure_type": ["none", "bearing"],
        }
    )
    csv_path = tmp_path / "eda.csv"
    dataset.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="numeric and binary"):
        load_eda_dataset(csv_path)


def test_load_eda_dataset_strips_and_rejects_failure_type_values(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 2.0],
            "temperature_motor": [50.0, 60.0],
            "rpm": [1000.0, 1200.0],
            "pressure_level": [30.0, 35.0],
            "rul_hours": [10.0, 8.0],
            "operating_mode": ["normal", "peak"],
            "failure_within_24h": [0, 1],
            "failure_type": [" none ", "bearing"],
        }
    )
    csv_path = tmp_path / "eda.csv"
    dataset.to_csv(csv_path, index=False)

    loaded = load_eda_dataset(csv_path)

    assert loaded["failure_type"].tolist() == ["none", "bearing"]


def test_load_eda_dataset_rejects_blank_failure_type(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 2.0],
            "temperature_motor": [50.0, 60.0],
            "rpm": [1000.0, 1200.0],
            "pressure_level": [30.0, 35.0],
            "rul_hours": [10.0, 8.0],
            "operating_mode": ["normal", "peak"],
            "failure_within_24h": [0, 1],
            "failure_type": ["   ", "bearing"],
        }
    )
    csv_path = tmp_path / "eda.csv"
    dataset.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="failure_type"):
        load_eda_dataset(csv_path)


def test_load_eda_dataset_rejects_unusable_numeric_columns(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": ["bad", "worse"],
            "temperature_motor": ["bad", "worse"],
            "rpm": ["bad", "worse"],
            "pressure_level": ["bad", "worse"],
            "rul_hours": ["bad", "worse"],
            "operating_mode": ["normal", "peak"],
            "failure_within_24h": [0, 1],
            "failure_type": ["none", "bearing"],
        }
    )
    csv_path = tmp_path / "eda.csv"
    dataset.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="vibration_rms"):
        load_eda_dataset(csv_path)


def test_load_eda_dataset_rejects_mixed_invalid_numeric_values(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, "bad"],
            "temperature_motor": [50.0, 60.0],
            "rpm": [1000.0, 1200.0],
            "pressure_level": [30.0, 35.0],
            "rul_hours": [10.0, 8.0],
            "operating_mode": ["normal", "peak"],
            "failure_within_24h": [0, 1],
            "failure_type": ["none", "bearing"],
        }
    )
    csv_path = tmp_path / "eda.csv"
    dataset.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="vibration_rms"):
        load_eda_dataset(csv_path)


def test_load_eda_dataset_rejects_non_finite_numeric_values(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, float("inf")],
            "temperature_motor": [50.0, 60.0],
            "rpm": [1000.0, 1200.0],
            "pressure_level": [30.0, 35.0],
            "rul_hours": [10.0, 8.0],
            "operating_mode": ["normal", "peak"],
            "failure_within_24h": [0, 1],
            "failure_type": ["none", "bearing"],
        }
    )
    csv_path = tmp_path / "eda.csv"
    dataset.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="non-finite"):
        load_eda_dataset(csv_path)


def test_load_eda_dataset_accepts_header_only_csv_and_build_eda_summary_handles_it(
    tmp_path: Path,
) -> None:
    dataset = pd.DataFrame(
        columns=[
            "vibration_rms",
            "temperature_motor",
            "rpm",
            "pressure_level",
            "rul_hours",
            "operating_mode",
            "failure_within_24h",
            "failure_type",
        ]
    )
    csv_path = tmp_path / "eda.csv"
    dataset.to_csv(csv_path, index=False)

    loaded = load_eda_dataset(csv_path)
    summary = build_eda_summary(loaded)

    assert loaded.empty
    assert loaded.columns.tolist() == dataset.columns.tolist()
    assert summary == {
        "row_count": 0,
        "column_count": 8,
        "missing_summary": {},
        "positive_failure_rate": 0.0,
    }


def test_build_eda_summary_handles_empty_frame_with_string_target_column() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": pd.Series(dtype=float),
            "temperature_motor": pd.Series(dtype=float),
            "rpm": pd.Series(dtype=float),
            "pressure_level": pd.Series(dtype=float),
            "rul_hours": pd.Series(dtype=float),
            "operating_mode": pd.Series(dtype="string"),
            "failure_within_24h": pd.Series(dtype="string"),
            "failure_type": pd.Series(dtype="string"),
        }
    )

    summary = build_eda_summary(dataset)

    assert summary == {
        "row_count": 0,
        "column_count": 8,
        "missing_summary": {},
        "positive_failure_rate": 0.0,
    }


def test_build_eda_summary_handles_empty_frame_with_malformed_target_column() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": pd.Series(dtype=float),
            "temperature_motor": pd.Series(dtype=float),
            "rpm": pd.Series(dtype=float),
            "pressure_level": pd.Series(dtype=float),
            "rul_hours": pd.Series(dtype=float),
            "operating_mode": pd.Series(dtype="string"),
            "failure_within_24h": pd.Series(dtype="string"),
            "failure_type": pd.Series(dtype="string"),
        }
    )

    summary = build_eda_summary(dataset)

    assert summary == {
        "row_count": 0,
        "column_count": 8,
        "missing_summary": {},
        "positive_failure_rate": 0.0,
    }


def test_build_eda_summary_reports_dataset_overview() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, None],
            "temperature_motor": [50.0, 60.0],
            "rpm": [1000.0, 1200.0],
            "pressure_level": [30.0, 35.0],
            "rul_hours": [10.0, 8.0],
            "operating_mode": ["normal", "peak"],
            "failure_within_24h": [0, 1],
            "failure_type": ["none", "bearing"],
        }
    )

    summary = build_eda_summary(dataset)

    assert summary == {
        "row_count": 2,
        "column_count": 8,
        "missing_summary": {"vibration_rms": 1},
        "positive_failure_rate": 50.0,
    }


def test_build_eda_summary_handles_empty_dataset() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": pd.Series(dtype=float),
            "temperature_motor": pd.Series(dtype=float),
            "rpm": pd.Series(dtype=float),
            "pressure_level": pd.Series(dtype=float),
            "rul_hours": pd.Series(dtype=float),
            "operating_mode": pd.Series(dtype="string"),
            "failure_within_24h": pd.Series(dtype=float),
            "failure_type": pd.Series(dtype="string"),
        }
    )

    summary = build_eda_summary(dataset)

    assert summary["row_count"] == 0
    assert summary["column_count"] == 8
    assert summary["missing_summary"] == {}
    assert summary["positive_failure_rate"] == 0.0


def test_build_target_distribution_figure_contains_binary_and_multiclass_bars() -> None:
    dataset = pd.DataFrame(
        {
            "failure_within_24h": [0, 0, 1],
            "failure_type": ["none", "bearing", "bearing"],
        }
    )

    figure = build_target_distribution_figure(dataset)

    assert len(figure.data) == 2
    assert figure.layout.title.text == "Target distributions"


def test_build_target_distribution_figure_requires_valid_target_column() -> None:
    dataset = pd.DataFrame(
        {
            "failure_within_24h": [0, "bad"],
            "failure_type": ["none", "bearing"],
        }
    )

    with pytest.raises(ValueError, match="failure_within_24h"):
        build_target_distribution_figure(dataset)


@pytest.mark.parametrize("failure_type_value", [1, True])
def test_build_target_distribution_figure_rejects_non_text_failure_type_labels(
    failure_type_value,
) -> None:
    dataset = pd.DataFrame(
        {
            "failure_within_24h": [0, 1],
            "failure_type": [failure_type_value, "bearing"],
        }
    )

    with pytest.raises(ValueError, match="failure_type"):
        build_target_distribution_figure(dataset)


def test_build_target_distribution_figure_shows_missing_binary_class() -> None:
    dataset = pd.DataFrame(
        {
            "failure_within_24h": [1, 1, 1],
            "failure_type": ["none", "bearing", "bearing"],
        }
    )

    figure = build_target_distribution_figure(dataset)

    assert list(figure.data[0].x) == ["0", "1"]
    assert list(figure.data[0].y) == [0, 3]


def test_build_target_distribution_figure_rejects_malformed_target_input() -> None:
    dataset = pd.DataFrame(
        {
            "failure_within_24h": [0, "bad"],
            "failure_type": ["none", "bearing"],
        }
    )

    with pytest.raises(ValueError, match="failure_within_24h"):
        build_target_distribution_figure(dataset)


def test_build_numeric_histogram_figure_contains_one_trace_per_numeric_column() -> None:
    dataset = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in NUMERIC_COLUMNS})

    figure = build_numeric_histogram_figure(dataset)

    assert len(figure.data) == len(NUMERIC_COLUMNS)
    assert figure.layout.title.text == "Numeric feature distributions"


def test_build_numeric_histogram_figure_keeps_numeric_columns_in_order() -> None:
    dataset = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in NUMERIC_COLUMNS})

    figure = build_numeric_histogram_figure(dataset)

    assert [trace.name for trace in figure.data] == list(NUMERIC_COLUMNS)
    assert [list(trace.x) for trace in figure.data] == [
        dataset[column].tolist() for column in NUMERIC_COLUMNS
    ]


def test_build_correlation_heatmap_figure_uses_numeric_columns_and_target() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 2.0, 3.0],
            "temperature_motor": [40.0, 50.0, 60.0],
            "rpm": [900.0, 1000.0, 1100.0],
            "pressure_level": [20.0, 25.0, 30.0],
            "rul_hours": [12.0, 10.0, 8.0],
            "failure_within_24h": [0, 0, 1],
        }
    )

    figure = build_correlation_heatmap_figure(dataset)

    assert len(figure.data) == 1
    assert figure.layout.title.text == "Correlation heatmap"


def test_build_failure_boxplot_figure_facets_by_feature_with_separate_scales() -> None:
    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 2.0, 3.0],
            "temperature_motor": [40.0, 50.0, 60.0],
            "rpm": [900.0, 1000.0, 1100.0],
            "pressure_level": [20.0, 25.0, 30.0],
            "rul_hours": [12.0, 10.0, 8.0],
            "failure_within_24h": [0, 0, 1],
        }
    )

    figure = build_failure_boxplot_figure(dataset)

    assert figure.layout.title.text == "Feature spread by failure flag"
    assert len({trace.yaxis for trace in figure.data}) == len(NUMERIC_COLUMNS)
    assert tuple(annotation.text for annotation in figure.layout.annotations) == tuple(NUMERIC_COLUMNS)
    assert all({str(value) for value in trace.x} == {"0", "1"} for trace in figure.data)


def test_summarize_probability_flags_high_risk() -> None:
    label, message = summarize_probability(0.82)

    assert label == HIGH_RISK_LABEL
    assert "0.82" in message


def test_summarize_probability_marks_low_risk_as_readable() -> None:
    label, message = summarize_probability(0.18)

    assert label == LOW_RISK_LABEL
    assert "0.18" in message


def test_get_prediction_task_options_lists_both_tasks() -> None:
    options = get_prediction_task_options()

    assert options == {
        task.display_name: task.task_name for task in TASK_CONFIGS.values()
    }


def test_get_prediction_task_options_rejects_duplicate_display_names(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dashboard_helpers,
        "TASK_CONFIGS",
        {
            "failure_within_24h": SimpleNamespace(
                display_name="Failure", task_name="failure_within_24h"
            ),
            "failure_type": SimpleNamespace(
                display_name="Failure", task_name="failure_type"
            ),
        },
    )

    with pytest.raises(ValueError, match="Duplicate task display name: Failure"):
        get_prediction_task_options()


def test_build_artifact_paths_match_failure_type_dashboard_loading() -> None:
    artifact_paths = build_artifact_paths("failure_type")

    assert artifact_paths.model_path.name.endswith("failure_type.joblib")
    assert artifact_paths.metrics_path.name.endswith("failure_type.csv")
    assert artifact_paths.importance_path.name.endswith("failure_type.csv")


def test_format_multiclass_probabilities_returns_ranked_table() -> None:
    probability_frame = format_multiclass_probabilities(
        class_labels=["mechanical", "none", "thermal"],
        probabilities=[0.3, 0.6, 0.1],
    )

    assert probability_frame["class_label"].tolist() == [
        "none",
        "mechanical",
        "thermal",
    ]
    assert probability_frame["probability"].tolist() == [0.6, 0.3, 0.1]


def test_required_artifacts_exist_requires_all_task_artifacts(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "best_model.joblib"
    metrics_path = tmp_path / "model_comparison.csv"
    importance_path = tmp_path / "feature_importance.csv"

    model_path.write_text("model")
    metrics_path.write_text("metrics")

    assert required_artifacts_exist(
        [model_path, metrics_path, importance_path]
    ) is False


def test_required_artifacts_exist_accepts_all_present_task_artifacts(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "best_model.joblib"
    metrics_path = tmp_path / "model_comparison.csv"
    importance_path = tmp_path / "feature_importance.csv"

    model_path.write_text("model")
    metrics_path.write_text("metrics")
    importance_path.write_text("importance")

    assert required_artifacts_exist(
        [model_path, metrics_path, importance_path]
    ) is True


def test_required_artifacts_exist_rejects_missing_required_artifacts(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "best_model.joblib"
    metrics_path = tmp_path / "model_comparison.csv"

    model_path.write_text("model")

    assert required_artifacts_exist([model_path, metrics_path]) is False


def test_extract_operating_mode_options_reads_fitted_pipeline_categories() -> None:
    features = pd.DataFrame(
        {
            "vibration_rms": [1.0, 1.5, 2.0, 2.5],
            "temperature_motor": [50.0, 55.0, 60.0, 65.0],
            "rpm": [1000.0, 1100.0, 1200.0, 1300.0],
            "pressure_level": [30.0, 31.0, 32.0, 33.0],
            "rul_hours": [20.0, 18.0, 16.0, 14.0],
            "operating_mode": ["stress", "normal", "maintenance", "stress"],
        }
    )
    target = pd.Series([0, 0, 1, 1])
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression()),
        ]
    )

    model.fit(features, target)

    assert extract_operating_mode_options(model) == ["maintenance", "normal", "stress"]


def test_extract_operating_mode_options_reads_fitted_preprocessor_categories() -> None:
    features = pd.DataFrame(
        {
            "vibration_rms": [1.0, 1.5, 2.0],
            "temperature_motor": [50.0, 55.0, 60.0],
            "rpm": [1000.0, 1100.0, 1200.0],
            "pressure_level": [30.0, 31.0, 32.0],
            "rul_hours": [20.0, 18.0, 16.0],
            "operating_mode": ["critical", "normal", "critical"],
        }
    )
    preprocessor = build_preprocessor()

    preprocessor.fit(features)

    assert extract_operating_mode_options(preprocessor) == ["critical", "normal"]


def test_dashboard_app_renders_selected_task_model_comparison(monkeypatch) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"

    selected_tasks: list[str] = []
    request_params: list[dict[str, object] | None] = []
    json_payloads: list[dict[str, object]] = []
    captions: list[str] = []
    errors: list[str] = []
    post_calls: list[tuple[str, dict[str, object], float]] = []
    dataframe_payloads: list[object] = []
    metric_calls: list[tuple[str, object]] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object], text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, timeout: float, params: dict[str, object] | None = None):
        assert url == "http://127.0.0.1:8000/model-info"
        assert timeout == 5
        request_params.append(params)
        assert params == {"task_name": "failure_type"}
        return FakeResponse(
            200,
            {
                "primary_metric_name": "weighted_f1",
                "best_model_name": "multiclass-model",
                "models_comparison": [
                    {
                        "model_name": "multiclass-model",
                        "primary_metric_value": 0.91,
                        "secondary_metric_value": 0.13,
                    },
                    {
                        "model_name": "binary-model",
                        "primary_metric_value": 0.84,
                        "secondary_metric_value": 0.2,
                    },
                ]
            },
        )

    def fake_post(url: str, json: dict[str, object], timeout: float):
        post_calls.append((url, json, timeout))
        assert timeout == 5
        if url.endswith("/predict/failure-type"):
            return FakeResponse(
                200,
                {
                    "task_name": "failure_type",
                    "model_name": "multiclass-model",
                    "predicted_class": "mechanical",
                    "class_probabilities": [
                        {"class_label": "mechanical", "probability": 0.7},
                        {"class_label": "thermal", "probability": 0.3},
                    ],
                    "importance_summary": ["rpm", "temperature_motor"],
                },
            )
        pytest.fail(f"unexpected endpoint: {url}")

    class _ContextManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _MetricColumn:
        def metric(self, label, value):
            metric_calls.append((label, value))

    def fake_columns(arg):
        if isinstance(arg, int):
            return tuple(_MetricColumn() for _ in range(arg))
        return (_ContextManager(), _ContextManager())

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=lambda label, options: selected_tasks.append(label) or (
            "Prediction" if label == "Dashboard mode" else "Failure Type"
        ),
        columns=fake_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda payload: json_payloads.append(payload),
        caption=lambda message: captions.append(message),
        error=lambda message: errors.append(message),
        stop=lambda: pytest.fail("dashboard should not stop on the happy path"),
        info=lambda *_: None,
        write=lambda *_: None,
        dataframe=lambda payload, **_kwargs: dataframe_payloads.append(payload),
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fake_get, post=fake_post))
    sys.modules.pop(app_module_name, None)

    importlib.import_module(app_module_name)

    assert selected_tasks == ["Dashboard mode", "Prediction task"]
    assert request_params == [{"task_name": "failure_type"}]
    assert metric_calls == [
        ("Primary Metric", "weighted_f1"),
        ("Best Model", "multiclass-model"),
        ("Compared Models", "2"),
    ]
    assert len(dataframe_payloads) == 1
    assert post_calls == [
        (
            "http://127.0.0.1:8000/predict/failure-type",
            {
                "vibration_rms": 1.0,
                "temperature_motor": 1.0,
                "rpm": 1.0,
                "pressure_level": 1.0,
                "rul_hours": 1.0,
                "operating_mode": "normal",
            },
            5,
        )
    ]
    assert json_payloads == [
        {
            "task_name": "failure_type",
            "model_name": "multiclass-model",
            "predicted_class": "mechanical",
            "class_probabilities": [
                {"class_label": "mechanical", "probability": 0.7},
                {"class_label": "thermal", "probability": 0.3},
            ],
            "importance_summary": ["rpm", "temperature_motor"],
        }
    ]
    assert list(dataframe_payloads[0]["model_name"]) == ["multiclass-model", "binary-model"]
    assert list(dataframe_payloads[0]["primary_metric_value"]) == [0.91, 0.84]
    assert captions == ["Features used: vibration_rms, temperature_motor, rpm, pressure_level, rul_hours, operating_mode"]
    assert errors == []

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_dashboard_app_shows_model_info_error_without_blocking_prediction_panel(
    monkeypatch,
) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    json_payloads: list[dict[str, object]] = []
    post_calls: list[tuple[str, dict[str, object], float]] = []
    errors: list[str] = []

    class RequestException(Exception):
        pass

    def fake_get(*_args, **_kwargs):
        raise RequestException("connection refused")

    def fake_post(url: str, json: dict[str, object], timeout: float):
        post_calls.append((url, json, timeout))
        assert url == "http://127.0.0.1:8000/predict/failure-type"
        assert timeout == 5
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "task_name": "failure_type",
                "model_name": "multiclass-model",
                "predicted_class": "mechanical",
                "class_probabilities": [
                    {"class_label": "mechanical", "probability": 0.7},
                    {"class_label": "thermal", "probability": 0.3},
                ],
                "importance_summary": ["rpm", "temperature_motor"],
            },
        )

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=_prediction_mode_selectbox,
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda payload: json_payloads.append(payload),
        caption=lambda *_: None,
        error=lambda message: errors.append(message),
        stop=lambda: pytest.fail("dashboard should not stop when model info fails"),
        info=lambda *_: None,
        write=lambda *_: None,
        dataframe=lambda *_: None,
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(get=fake_get, post=fake_post, RequestException=RequestException),
    )
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    sys.modules.pop(app_module_name, None)

    importlib.import_module(app_module_name)

    assert errors == ["Unable to load model info from API."]
    assert post_calls == [
        (
            "http://127.0.0.1:8000/predict/failure-type",
            {
                "vibration_rms": 1.0,
                "temperature_motor": 1.0,
                "rpm": 1.0,
                "pressure_level": 1.0,
                "rul_hours": 1.0,
                "operating_mode": "normal",
            },
            5,
        )
    ]
    assert json_payloads == [
        {
            "task_name": "failure_type",
            "model_name": "multiclass-model",
            "predicted_class": "mechanical",
            "class_probabilities": [
                {"class_label": "mechanical", "probability": 0.7},
                {"class_label": "thermal", "probability": 0.3},
            ],
            "importance_summary": ["rpm", "temperature_motor"],
        }
    ]

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_dashboard_app_renders_eda_mode(monkeypatch) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    plotly_charts: list[object] = []
    metrics: list[tuple[str, object]] = []
    dataframes: list[object] = []
    write_calls: list[object] = []
    selected_labels: list[str] = []
    loaded_paths: list[Path] = []

    dataset = pd.DataFrame(
        {
            "vibration_rms": [1.0, 2.0],
            "temperature_motor": [50.0, 60.0],
            "rpm": [1000.0, 1200.0],
            "pressure_level": [30.0, 35.0],
            "rul_hours": [10.0, 8.0],
            "operating_mode": ["normal", "peak"],
            "failure_within_24h": [0, 1],
            "failure_type": ["none", "bearing"],
        }
    )

    def fake_load_eda_dataset(csv_path: Path):
        loaded_paths.append(csv_path)
        return dataset

    def fake_build_eda_summary(frame: pd.DataFrame):
        assert frame.equals(dataset)
        return {
            "row_count": 2,
            "column_count": 8,
            "missing_summary": {"vibration_rms": 1},
            "positive_failure_rate": 50.0,
        }

    def fake_build_target_distribution_figure(frame: pd.DataFrame):
        assert frame.equals(dataset)
        return "target-distribution-figure"

    def fake_build_numeric_histogram_figure(frame: pd.DataFrame):
        assert frame.equals(dataset)
        return "numeric-histogram-figure"

    def fake_build_correlation_heatmap_figure(frame: pd.DataFrame):
        assert frame.equals(dataset)
        return "correlation-heatmap-figure"

    def fake_build_failure_boxplot_figure(frame: pd.DataFrame):
        assert frame.equals(dataset)
        return "failure-boxplot-figure"

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=lambda label, options: selected_labels.append(label) or ("EDA" if label == "Dashboard mode" else pytest.fail(f"unexpected selectbox label: {label}")),
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda *_: None,
        caption=lambda *_: None,
        error=lambda *_: pytest.fail("EDA mode should not show an error on the happy path"),
        stop=lambda: pytest.fail("EDA mode should not stop on the happy path"),
        info=lambda *_: None,
        write=lambda value=None, **_: write_calls.append(value),
        dataframe=lambda payload, **_: dataframes.append(payload),
        metric=lambda label, value: metrics.append((label, value)),
        plotly_chart=lambda figure, **_: plotly_charts.append(figure),
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(
            get=lambda *_args, **_kwargs: pytest.fail("EDA mode should not call the API"),
            post=lambda *_args, **_kwargs: pytest.fail("EDA mode should not call the API"),
            RequestException=Exception,
        ),
    )
    monkeypatch.setattr(dashboard_eda, "load_eda_dataset", fake_load_eda_dataset)
    monkeypatch.setattr(dashboard_eda, "build_eda_summary", fake_build_eda_summary)
    monkeypatch.setattr(
        dashboard_eda,
        "build_target_distribution_figure",
        fake_build_target_distribution_figure,
    )
    monkeypatch.setattr(
        dashboard_eda,
        "build_numeric_histogram_figure",
        fake_build_numeric_histogram_figure,
    )
    monkeypatch.setattr(
        dashboard_eda,
        "build_correlation_heatmap_figure",
        fake_build_correlation_heatmap_figure,
    )
    monkeypatch.setattr(
        dashboard_eda,
        "build_failure_boxplot_figure",
        fake_build_failure_boxplot_figure,
    )
    sys.modules.pop(app_module_name, None)

    importlib.import_module(app_module_name)

    assert selected_labels == ["Dashboard mode"]
    assert loaded_paths == [config_module.DATASET_PATH]
    assert metrics == [
        ("Rows", 2),
        ("Columns", 8),
        ("Positive failure rate", "50.0%"),
    ]
    assert dataframes == [{"vibration_rms": 1}]
    assert write_calls == []
    assert plotly_charts == [
        "target-distribution-figure",
        "numeric-histogram-figure",
        "correlation-heatmap-figure",
        "failure-boxplot-figure",
    ]

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_dashboard_app_shows_eda_dataset_error(monkeypatch) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    errors: list[str] = []
    plotly_charts: list[object] = []

    def fake_load_eda_dataset(_csv_path: Path):
        raise ValueError("Dataset is missing required EDA columns: failure_type")

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=lambda label, options: "EDA" if label == "Dashboard mode" else pytest.fail(
            f"unexpected selectbox label: {label}"
        ),
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda *_: None,
        caption=lambda *_: None,
        error=lambda message: errors.append(message),
        stop=lambda: pytest.fail("EDA dataset errors should not stop the app"),
        info=lambda *_: None,
        write=lambda *_: None,
        dataframe=lambda *_: None,
        metric=lambda *_: None,
        plotly_chart=lambda figure, **_: plotly_charts.append(figure),
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(
            get=lambda *_args, **_kwargs: pytest.fail("EDA mode should not call the API"),
            post=lambda *_args, **_kwargs: pytest.fail("EDA mode should not call the API"),
            RequestException=Exception,
        ),
    )
    monkeypatch.setattr(dashboard_eda, "load_eda_dataset", fake_load_eda_dataset)
    sys.modules.pop(app_module_name, None)

    importlib.import_module(app_module_name)

    assert errors == ["Dataset is missing required EDA columns: failure_type"]
    assert plotly_charts == []

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_dashboard_app_shows_generic_eda_dataset_load_error(monkeypatch) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    errors: list[str] = []
    plotly_charts: list[object] = []

    def fake_load_eda_dataset(_csv_path: Path):
        raise pd.errors.EmptyDataError("No columns to parse from file")

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=lambda label, options: "EDA" if label == "Dashboard mode" else pytest.fail(
            f"unexpected selectbox label: {label}"
        ),
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda *_: None,
        caption=lambda *_: None,
        error=lambda message: errors.append(message),
        stop=lambda: pytest.fail("EDA dataset errors should not stop the app"),
        info=lambda *_: None,
        write=lambda *_: None,
        dataframe=lambda *_: None,
        metric=lambda *_: None,
        plotly_chart=lambda figure, **_: plotly_charts.append(figure),
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(
            get=lambda *_args, **_kwargs: pytest.fail("EDA mode should not call the API"),
            post=lambda *_args, **_kwargs: pytest.fail("EDA mode should not call the API"),
            RequestException=Exception,
        ),
    )
    monkeypatch.setattr(dashboard_eda, "load_eda_dataset", fake_load_eda_dataset)
    sys.modules.pop(app_module_name, None)

    importlib.import_module(app_module_name)

    assert errors == ["Unable to load the dataset for EDA mode."]
    assert plotly_charts == []

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_dashboard_app_shows_generic_eda_file_access_error(monkeypatch) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    errors: list[str] = []
    plotly_charts: list[object] = []

    def fake_load_eda_dataset(_csv_path: Path):
        raise PermissionError("permission denied")

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=lambda label, options: "EDA" if label == "Dashboard mode" else pytest.fail(
            f"unexpected selectbox label: {label}"
        ),
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda *_: None,
        caption=lambda *_: None,
        error=lambda message: errors.append(message),
        stop=lambda: pytest.fail("EDA dataset errors should not stop the app"),
        info=lambda *_: None,
        write=lambda *_: None,
        dataframe=lambda *_: None,
        metric=lambda *_: None,
        plotly_chart=lambda figure, **_: plotly_charts.append(figure),
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(
            get=lambda *_args, **_kwargs: pytest.fail("EDA mode should not call the API"),
            post=lambda *_args, **_kwargs: pytest.fail("EDA mode should not call the API"),
            RequestException=Exception,
        ),
    )
    monkeypatch.setattr(dashboard_eda, "load_eda_dataset", fake_load_eda_dataset)
    sys.modules.pop(app_module_name, None)

    importlib.import_module(app_module_name)

    assert errors == ["Unable to load the dataset for EDA mode."]
    assert plotly_charts == []

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_dashboard_app_handles_empty_eda_dataset_without_plotting(monkeypatch) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    info_messages: list[str] = []
    metrics: list[tuple[str, object]] = []
    plotly_charts: list[object] = []
    errors: list[str] = []

    empty_dataset = pd.DataFrame(
        {
            "vibration_rms": pd.Series(dtype=float),
            "temperature_motor": pd.Series(dtype=float),
            "rpm": pd.Series(dtype=float),
            "pressure_level": pd.Series(dtype=float),
            "rul_hours": pd.Series(dtype=float),
            "operating_mode": pd.Series(dtype="string"),
            "failure_within_24h": pd.Series(dtype=float),
            "failure_type": pd.Series(dtype="string"),
        }
    )

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=lambda label, options: "EDA" if label == "Dashboard mode" else pytest.fail(
            f"unexpected selectbox label: {label}"
        ),
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda *_: None,
        caption=lambda *_: None,
        error=lambda message: errors.append(message),
        stop=lambda: pytest.fail("EDA empty dataset should not stop the app"),
        info=lambda message: info_messages.append(message),
        write=lambda *_: None,
        dataframe=lambda *_: None,
        metric=lambda label, value: metrics.append((label, value)),
        plotly_chart=lambda figure, **_: plotly_charts.append(figure),
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(
            get=lambda *_args, **_kwargs: pytest.fail("EDA mode should not call the API"),
            post=lambda *_args, **_kwargs: pytest.fail("EDA mode should not call the API"),
            RequestException=Exception,
        ),
    )
    monkeypatch.setattr(dashboard_eda, "load_eda_dataset", lambda _csv_path: empty_dataset)
    sys.modules.pop(app_module_name, None)

    importlib.import_module(app_module_name)

    assert errors == []
    assert metrics == [
        ("Rows", 0),
        ("Columns", 8),
        ("Positive failure rate", "0.0%"),
    ]
    assert info_messages == ["EDA dataset is empty."]
    assert plotly_charts == []

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_dashboard_app_handles_prediction_request_exception(
    monkeypatch,
) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    json_payloads: list[dict[str, object]] = []
    errors: list[str] = []

    class DashboardStopped(Exception):
        pass

    class RequestException(Exception):
        pass

    def fake_get(url: str, timeout: float, params: dict[str, object] | None = None):
        assert url == "http://127.0.0.1:8000/model-info"
        assert timeout == 5
        assert params == {"task_name": "failure_type"}
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "primary_metric_name": "weighted_f1",
                "best_model_name": "multiclass-model",
                "models_comparison": [
                    {
                        "model_name": "multiclass-model",
                        "primary_metric_value": 0.91,
                        "secondary_metric_value": 0.13,
                    },
                    {
                        "model_name": "binary-model",
                        "primary_metric_value": 0.84,
                        "secondary_metric_value": 0.2,
                    },
                ]
            },
        )

    def fake_post(*_args, **_kwargs):
        raise RequestException("connection refused")

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=_prediction_mode_selectbox,
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda payload: json_payloads.append(payload),
        caption=lambda *_: None,
        error=lambda message: errors.append(message),
        stop=lambda: (_ for _ in ()).throw(DashboardStopped()),
        info=lambda *_: None,
        write=lambda *_: None,
        dataframe=lambda *_: None,
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(get=fake_get, post=fake_post, RequestException=RequestException),
    )
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    sys.modules.pop(app_module_name, None)

    with pytest.raises(DashboardStopped):
        importlib.import_module(app_module_name)

    assert json_payloads == []
    assert errors == ["Unable to get prediction from API."]

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_dashboard_app_falls_back_to_default_message_for_non_json_prediction_error(
    monkeypatch,
) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    errors: list[str] = []

    class DashboardStopped(Exception):
        pass

    def fake_get(url: str, timeout: float, params: dict[str, object] | None = None):
        assert url == "http://127.0.0.1:8000/model-info"
        assert timeout == 5
        assert params == {"task_name": "failure_type"}
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "primary_metric_name": "weighted_f1",
                "best_model_name": "multiclass-model",
                "models_comparison": [
                    {
                        "model_name": "multiclass-model",
                        "primary_metric_value": 0.91,
                        "secondary_metric_value": 0.13,
                    },
                    {
                        "model_name": "binary-model",
                        "primary_metric_value": 0.84,
                        "secondary_metric_value": 0.2,
                    },
                ]
            },
        )

    def fake_post(*_args, **_kwargs):
        return SimpleNamespace(
            status_code=503,
            text="service unavailable",
            json=lambda: (_ for _ in ()).throw(ValueError("bad json")),
        )

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=_prediction_mode_selectbox,
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda *_: None,
        caption=lambda *_: None,
        error=lambda message: errors.append(message),
        stop=lambda: (_ for _ in ()).throw(DashboardStopped()),
        info=lambda *_: None,
        write=lambda *_: None,
        dataframe=lambda *_: None,
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fake_get, post=fake_post))
    sys.modules.pop(app_module_name, None)

    with pytest.raises(DashboardStopped):
        importlib.import_module(app_module_name)

    assert errors == ["Unable to get prediction from API."]

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_dashboard_app_prefers_detail_field_for_non_200_prediction_response(
    monkeypatch,
) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    errors: list[str] = []

    class DashboardStopped(Exception):
        pass

    def fake_get(url: str, timeout: float, params: dict[str, object] | None = None):
        assert url == "http://127.0.0.1:8000/model-info"
        assert timeout == 5
        assert params == {"task_name": "failure_type"}
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "primary_metric_name": "weighted_f1",
                "best_model_name": "multiclass-model",
                "models_comparison": [
                    {
                        "model_name": "multiclass-model",
                        "primary_metric_value": 0.91,
                        "secondary_metric_value": 0.13,
                    },
                    {
                        "model_name": "binary-model",
                        "primary_metric_value": 0.84,
                        "secondary_metric_value": 0.2,
                    },
                ]
            },
        )

    def fake_post(*_args, **_kwargs):
        return SimpleNamespace(
            status_code=503,
            text='{"detail":"prediction unavailable"}',
            json=lambda: {"detail": "prediction unavailable"},
        )

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=_prediction_mode_selectbox,
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda *_: None,
        caption=lambda *_: None,
        error=lambda message: errors.append(message),
        stop=lambda: (_ for _ in ()).throw(DashboardStopped()),
        info=lambda *_: None,
        write=lambda *_: None,
        dataframe=lambda *_: None,
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fake_get, post=fake_post))
    sys.modules.pop(app_module_name, None)

    with pytest.raises(DashboardStopped):
        importlib.import_module(app_module_name)

    assert errors == ["prediction unavailable"]

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_dashboard_app_formats_dict_detail_for_non_200_prediction_response(
    monkeypatch,
) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    errors: list[str] = []

    class DashboardStopped(Exception):
        pass

    def fake_get(url: str, timeout: float, params: dict[str, object] | None = None):
        assert url == "http://127.0.0.1:8000/model-info"
        assert timeout == 5
        assert params == {"task_name": "failure_type"}
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "primary_metric_name": "weighted_f1",
                "best_model_name": "multiclass-model",
                "models_comparison": [
                    {
                        "model_name": "multiclass-model",
                        "primary_metric_value": 0.91,
                        "secondary_metric_value": 0.13,
                    },
                    {
                        "model_name": "binary-model",
                        "primary_metric_value": 0.84,
                        "secondary_metric_value": 0.2,
                    },
                ]
            },
        )

    def fake_post(*_args, **_kwargs):
        return SimpleNamespace(
            status_code=503,
            text='{"detail":{"task_name":"failure_type","artifact_name":"model","reason":"could not be read"}}',
            json=lambda: {
                "detail": {
                    "task_name": "failure_type",
                    "artifact_name": "model",
                    "reason": "could not be read",
                }
            },
        )

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=_prediction_mode_selectbox,
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda *_: None,
        caption=lambda *_: None,
        error=lambda message: errors.append(message),
        stop=lambda: (_ for _ in ()).throw(DashboardStopped()),
        info=lambda *_: None,
        write=lambda *_: None,
        dataframe=lambda *_: None,
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fake_get, post=fake_post))
    sys.modules.pop(app_module_name, None)

    with pytest.raises(DashboardStopped):
        importlib.import_module(app_module_name)

    assert errors == ["failure_type: model: could not be read"]

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)


def test_dashboard_app_handles_malformed_prediction_json(
    monkeypatch,
) -> None:
    app_module_name = "predictive_maintenance.dashboard.app"
    errors: list[str] = []

    class DashboardStopped(Exception):
        pass

    def fake_get(url: str, timeout: float, params: dict[str, object] | None = None):
        assert url == "http://127.0.0.1:8000/model-info"
        assert timeout == 5
        assert params == {"task_name": "failure_type"}
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "primary_metric_name": "weighted_f1",
                "best_model_name": "multiclass-model",
                "models_comparison": [
                    {
                        "model_name": "multiclass-model",
                        "primary_metric_value": 0.91,
                        "secondary_metric_value": 0.13,
                    },
                    {
                        "model_name": "binary-model",
                        "primary_metric_value": 0.84,
                        "secondary_metric_value": 0.2,
                    },
                ]
            },
        )

    def fake_post(*_args, **_kwargs):
        return SimpleNamespace(status_code=200, json=lambda: (_ for _ in ()).throw(ValueError("bad json")))

    streamlit_stub = SimpleNamespace(
        set_page_config=lambda **_: None,
        title=lambda *_: None,
        selectbox=_prediction_mode_selectbox,
        columns=_dashboard_columns,
        subheader=lambda *_: None,
        form=lambda *_: _DashboardColumnContext(),
        number_input=lambda *_args, **_kwargs: 1.0,
        text_input=lambda *_args, **_kwargs: "normal",
        form_submit_button=lambda *_args, **_kwargs: True,
        json=lambda *_: None,
        caption=lambda *_: None,
        error=lambda message: errors.append(message),
        stop=lambda: (_ for _ in ()).throw(DashboardStopped()),
        info=lambda *_: None,
        write=lambda *_: None,
        dataframe=lambda *_: None,
        bar_chart=lambda *_: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fake_get, post=fake_post))
    sys.modules.pop(app_module_name, None)

    with pytest.raises(DashboardStopped):
        importlib.import_module(app_module_name)

    assert errors == ["Unable to read prediction response from API."]

    sys.modules.pop(app_module_name, None)
    sys.modules.pop("streamlit", None)
    sys.modules.pop("requests", None)
