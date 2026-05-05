from pathlib import Path

import joblib
import pandas as pd

from predictive_maintenance.config import build_artifact_paths


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_model(model, output_path: Path) -> Path:
    ensure_parent_directory(output_path)
    joblib.dump(model, output_path)
    return output_path


def save_frame(frame: pd.DataFrame, output_path: Path) -> Path:
    ensure_parent_directory(output_path)
    frame.to_csv(output_path, index=False)
    return output_path


def load_model(model_path: Path):
    return joblib.load(model_path)


def load_frame(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def resolve_artifact_paths(task_name: str, artifact_root: Path | None = None):
    artifact_paths = build_artifact_paths(task_name)
    if artifact_root is None:
        return artifact_paths
    return type(artifact_paths)(
        model_path=artifact_root / artifact_paths.model_path.relative_to(artifact_paths.model_path.parents[1]),
        metrics_path=artifact_root / artifact_paths.metrics_path.relative_to(artifact_paths.metrics_path.parents[1]),
        importance_path=artifact_root / artifact_paths.importance_path.relative_to(artifact_paths.importance_path.parents[1]),
    )
