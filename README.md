# Predictive Maintenance MVP

## Setup

1. Use Python 3.11.
2. Install runtime and dev dependencies with `uv`.

```text
uv sync
```

## Local workflow

1. Put `industrial_machine_maintenance.csv` in `data/`.
2. Run the full test suite.
3. Run the training pipeline.
4. Start the FastAPI API.
5. Open the Streamlit dashboard.
6. Use the notebook for EDA and model-comparison discussion.

Open `notebooks/01_eda_and_model_comparison.ipynb` in your preferred notebook environment if you have one installed separately.

```text
uv run pytest
uv run python -m predictive_maintenance.train
uv run python -m predictive_maintenance.train --task-name failure_within_24h
uv run python -m predictive_maintenance.train --task-name failure_type
uv run uvicorn predictive_maintenance.api.app:app --reload
uv run streamlit run src/predictive_maintenance/dashboard/app.py
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
```
