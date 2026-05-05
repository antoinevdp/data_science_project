import requests
import streamlit as st
import pandas as pd

from predictive_maintenance.config import API_BASE_URL, FEATURE_COLUMNS
from predictive_maintenance.config import DATASET_PATH
from predictive_maintenance.dashboard import eda as dashboard_eda
from predictive_maintenance.dashboard.helpers import get_prediction_task_options


def build_api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}/{path.lstrip('/')}"


def extract_error_message(response: object, fallback_message: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return fallback_message

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail
        if isinstance(detail, dict):
            parts: list[str] = []
            task_name = detail.get("task_name")
            artifact_name = detail.get("artifact_name")
            reason = detail.get("reason")
            if isinstance(task_name, str) and task_name.strip():
                parts.append(task_name)
            if isinstance(artifact_name, str) and artifact_name.strip():
                parts.append(artifact_name)
            if isinstance(reason, str) and reason.strip():
                parts.append(reason)
            if parts:
                return ": ".join(parts)
            message = detail.get("message")
            if isinstance(message, str) and message.strip():
                return message

    return fallback_message


def render_eda_dashboard() -> None:
    st.subheader("EDA overview")

    try:
        dataset = dashboard_eda.load_eda_dataset(DATASET_PATH)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        st.error("Unable to load the dataset for EDA mode.")
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    summary = dashboard_eda.build_eda_summary(dataset)

    st.metric("Rows", summary["row_count"])
    st.metric("Columns", summary["column_count"])
    st.metric("Positive failure rate", f"{summary['positive_failure_rate']}%")

    if dataset.empty:
        st.info("EDA dataset is empty.")
        return

    st.subheader("Missing values")
    st.dataframe(summary["missing_summary"])

    st.subheader("Target distributions")
    st.plotly_chart(dashboard_eda.build_target_distribution_figure(dataset))

    st.subheader("Numeric feature distributions")
    st.plotly_chart(dashboard_eda.build_numeric_histogram_figure(dataset))

    st.subheader("Correlation heatmap")
    st.plotly_chart(dashboard_eda.build_correlation_heatmap_figure(dataset))

    st.subheader("Feature spread by failure flag")
    st.plotly_chart(dashboard_eda.build_failure_boxplot_figure(dataset))


def render_prediction_dashboard() -> None:
    task_options = get_prediction_task_options()
    selected_label = st.selectbox("Prediction task", options=list(task_options.keys()))
    task_name = task_options[selected_label]

    model_info: dict[str, object] | None = None
    model_info_error: str | None = None

    try:
        model_info_response = requests.get(
            build_api_url("model-info"),
            params={"task_name": task_name},
            timeout=5,
        )
    except getattr(requests, "RequestException", Exception):
        model_info_error = "Unable to load model info from API."
    else:
        if model_info_response.status_code != 200:
            model_info_error = extract_error_message(
                model_info_response, "Unable to load model info from API."
            )
        else:
            try:
                model_info = model_info_response.json()
            except ValueError:
                model_info_error = "Unable to read model info response from API."

    left_column, right_column = st.columns([1.2, 1])

    with left_column:
        st.subheader("Model information")
        if model_info_error is not None:
            st.error(model_info_error)
        elif model_info is not None:
            metric_columns = st.columns(3)
            metric_columns[0].metric("Primary Metric", f"{model_info['primary_metric_name']}")
            metric_columns[1].metric("Best Model", model_info["best_model_name"])
            metric_columns[2].metric("Compared Models", str(len(model_info["models_comparison"])))

            st.subheader("Model comparison")
            st.dataframe(pd.DataFrame(model_info["models_comparison"]))

    with right_column:
        st.subheader("Predict a machine scenario")
        with st.form("prediction_form"):
            vibration_rms = st.number_input("vibration_rms", min_value=0.0, value=1.5)
            temperature_motor = st.number_input("temperature_motor", min_value=0.0, value=60.0)
            rpm = st.number_input("rpm", min_value=0.0, value=1200.0)
            pressure_level = st.number_input("pressure_level", min_value=0.0, value=30.0)
            rul_hours = st.number_input("rul_hours", min_value=0.0, value=12.0)
            operating_mode = st.text_input("operating_mode", value="normal")
            submitted = st.form_submit_button("Predict")

        if submitted:
            endpoint = (
                build_api_url("predict/failure-within-24h")
                if task_name == "failure_within_24h"
                else build_api_url("predict/failure-type")
            )
            try:
                response = requests.post(
                    endpoint,
                    json={
                        "vibration_rms": vibration_rms,
                        "temperature_motor": temperature_motor,
                        "rpm": rpm,
                        "pressure_level": pressure_level,
                        "rul_hours": rul_hours,
                        "operating_mode": operating_mode,
                    },
                    timeout=5,
                )
            except getattr(requests, "RequestException", Exception):
                st.error("Unable to get prediction from API.")
                st.stop()

            if response.status_code != 200:
                st.error(
                    extract_error_message(response, "Unable to get prediction from API.")
                )
                st.stop()

            try:
                prediction_payload = response.json()
            except ValueError:
                st.error("Unable to read prediction response from API.")
                st.stop()

            st.json(prediction_payload)
            st.caption(f"Features used: {', '.join(FEATURE_COLUMNS)}")


st.set_page_config(page_title="Predictive Maintenance MVP", layout="wide")
st.title("Predictive Maintenance Dashboard")

mode = st.selectbox("Dashboard mode", options=["Prediction", "EDA"])

if mode == "EDA":
    render_eda_dashboard()
else:
    render_prediction_dashboard()
