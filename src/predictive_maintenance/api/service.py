from pathlib import Path
from functools import lru_cache
import math

import pandas as pd

from predictive_maintenance import config
from predictive_maintenance.api.schemas import (
    ClassProbability,
    FailureTypePredictionResponse,
    FailureWithin24hPredictionResponse,
    ModelInfoComparisonRow,
    ModelInfoResponse,
    ModelInfoTask,
)
from predictive_maintenance.artifacts import load_frame, load_model, resolve_artifact_paths
from predictive_maintenance.config import ARTIFACT_ROOT
from predictive_maintenance.dashboard.helpers import (
    build_input_frame,
    format_multiclass_probabilities,
    summarize_probability,
)


TASK_BUNDLE_CACHE_MAXSIZE = 64


class TaskArtifactError(RuntimeError):
    def __init__(self, task_name: str, artifact_name: str, artifact_path: Path, reason: str) -> None:
        super().__init__(f"{task_name}: {artifact_name} artifact {reason} at {artifact_path}")
        self.task_name = task_name
        self.artifact_name = artifact_name
        self.artifact_path = artifact_path
        self.reason = reason


def _extract_model_name(comparison: pd.DataFrame) -> str | None:
    if comparison.empty or "model_name" not in comparison.columns:
        return None

    try:
        model_name = comparison.iloc[0]["model_name"]
    except (IndexError, KeyError):
        return None

    if pd.isna(model_name):
        return None

    model_name_text = str(model_name).strip()
    return model_name_text or None


def _read_model_name(metrics_path: Path) -> str | None:
    if not metrics_path.exists():
        return None

    try:
        comparison = load_frame(metrics_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return None

    return _extract_model_name(comparison)


def _comparison_metric_columns(task_name: str) -> tuple[str, str, str]:
    task_config = config.get_task_config(task_name)
    if task_config.problem_type == "binary_classification":
        return ("f1", "recall", "precision")
    return ("weighted_f1", "macro_f1", "weighted_recall")


def _validated_metric_values(
    task_name: str, metric_name: str, comparison: pd.DataFrame, metric_label: str
) -> pd.Series:
    metric_values = pd.to_numeric(comparison[metric_name], errors="coerce")
    if metric_values.isna().all() or not metric_values.map(math.isfinite).all():
        raise TaskArtifactError(
            task_name,
            "comparison",
            Path(f"{task_name}:comparison"),
            f"is malformed: {metric_label} contains non-numeric or non-finite values",
        )

    return metric_values.astype(float)


def _validated_model_names(task_name: str, comparison: pd.DataFrame) -> pd.Series:
    model_names = comparison["model_name"]
    if model_names.isna().any():
        raise TaskArtifactError(
            task_name,
            "comparison",
            Path(f"{task_name}:comparison"),
            "is malformed: model name contains empty values",
        )

    normalized_model_names = model_names.astype(str).str.strip()
    if normalized_model_names.eq("").any():
        raise TaskArtifactError(
            task_name,
            "comparison",
            Path(f"{task_name}:comparison"),
            "is malformed: model name contains empty values",
        )

    if normalized_model_names.duplicated().any():
        raise TaskArtifactError(
            task_name,
            "comparison",
            Path(f"{task_name}:comparison"),
            "is malformed: duplicate model names",
        )

    return normalized_model_names


def _comparison_rows(
    task_name: str, comparison: pd.DataFrame, primary_metric_name: str
) -> list[ModelInfoComparisonRow]:
    primary_col, metric_two_col, metric_three_col = _comparison_metric_columns(task_name)
    required_columns = ["model_name", primary_col, metric_two_col, metric_three_col]

    if comparison.empty:
        raise TaskArtifactError(
            task_name,
            "comparison",
            Path(f"{task_name}:comparison"),
            "is malformed: no comparison rows",
        )

    missing_columns = [column for column in required_columns if column not in comparison.columns]
    if missing_columns:
        raise TaskArtifactError(
            task_name,
            "comparison",
            Path(f"{task_name}:comparison"),
            f"is malformed: missing columns {', '.join(missing_columns)}",
        )

    model_names = _validated_model_names(task_name, comparison)
    primary_metric_values = _validated_metric_values(
        task_name, primary_col, comparison, "primary metric"
    )
    metric_two_values = _validated_metric_values(
        task_name, metric_two_col, comparison, "supporting metric"
    )
    metric_three_values = _validated_metric_values(
        task_name, metric_three_col, comparison, "supporting metric"
    )

    ranking_frame = pd.DataFrame(
        {
            "primary_metric": primary_metric_values,
            "metric_two": metric_two_values,
            "metric_three": metric_three_values,
        },
        index=comparison.index,
    )
    best_index = ranking_frame.sort_values(
        by=["primary_metric", "metric_two", "metric_three"], ascending=False
    ).index[0]
    best_model_name = str(model_names.loc[best_index])

    try:
        return [
            ModelInfoComparisonRow(
                model_name=str(model_names.loc[row.Index]),
                primary_metric=float(primary_metric_values.loc[row.Index]),
                metric_two=float(metric_two_values.loc[row.Index]),
                metric_three=float(metric_three_values.loc[row.Index]),
                is_best_model=str(model_names.loc[row.Index]) == best_model_name,
            )
            for row in comparison.itertuples()
        ]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise TaskArtifactError(
            task_name,
            "comparison",
            Path(f"{task_name}:comparison"),
            "is malformed",
        ) from exc


def _load_model_comparison(task_name: str, artifact_root: Path = ARTIFACT_ROOT) -> pd.DataFrame:
    artifact_paths = resolve_artifact_paths(task_name, _resolve_runtime_artifact_root(task_name, artifact_root))
    if not artifact_paths.metrics_path.exists():
        raise TaskArtifactError(task_name, "metrics", artifact_paths.metrics_path, "is missing")

    try:
        return load_frame(artifact_paths.metrics_path)
    except Exception as exc:
        raise TaskArtifactError(task_name, "metrics", artifact_paths.metrics_path, "could not be read") from exc


def _resolve_runtime_artifact_root(task_name: str, artifact_root: Path) -> Path:
    if artifact_root != ARTIFACT_ROOT:
        return artifact_root

    candidate = resolve_artifact_paths(task_name, artifact_root)
    if candidate.model_path.exists():
        return artifact_root

    for parent in artifact_root.parents:
        fallback_root = parent / "artifacts"
        if resolve_artifact_paths(task_name, fallback_root).model_path.exists():
            return fallback_root

    return artifact_root


def _artifact_signature(path: Path) -> tuple[int, int, int, int]:
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return (-1, -1, -1, -1)

    inode = getattr(stat_result, "st_ino", -1)
    change_time = getattr(stat_result, "st_ctime_ns", -1)
    return (inode, change_time, stat_result.st_mtime_ns, stat_result.st_size)


def _artifact_version(
    task_name: str, artifact_root: Path
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    artifact_paths = resolve_artifact_paths(task_name, _resolve_runtime_artifact_root(task_name, artifact_root))
    return (
        _artifact_signature(artifact_paths.model_path),
        _artifact_signature(artifact_paths.metrics_path),
        _artifact_signature(artifact_paths.importance_path),
    )


def build_model_info_payload(
    task_name: str | Path | None = None, artifact_root: Path = ARTIFACT_ROOT
) -> ModelInfoResponse:
    if task_name is None or isinstance(task_name, Path):
        if isinstance(task_name, Path):
            artifact_root = task_name
        tasks: list[ModelInfoTask] = []
        for candidate_task_name in config.TASK_CONFIGS:
            try:
                task_config = config.get_task_config(candidate_task_name)
                runtime_artifact_root = _resolve_runtime_artifact_root(candidate_task_name, artifact_root)
                bundle = load_task_bundle(candidate_task_name, runtime_artifact_root)
            except TaskArtifactError:
                continue

            tasks.append(
                ModelInfoTask(
                    task_name=candidate_task_name,
                    model_name=_extract_model_name(bundle["comparison"]),
                    primary_metric=task_config.primary_metric,
                )
            )
        return ModelInfoResponse(tasks=tasks)

    task_config = config.get_task_config(task_name)
    comparison = _load_model_comparison(task_name, artifact_root)
    rows = _comparison_rows(task_name, comparison, task_config.primary_metric)
    best_model_name = next(row.model_name for row in rows if row.is_best_model)

    return ModelInfoResponse(
        task_name=task_name,
        primary_metric_name=task_config.primary_metric,
        best_model_name=best_model_name,
        models_comparison=rows,
    )


def load_task_bundle(task_name: str, artifact_root: Path = ARTIFACT_ROOT) -> dict[str, object]:
    artifact_root = Path(artifact_root).resolve()
    return _load_task_bundle_cached(task_name, str(artifact_root), _artifact_version(task_name, artifact_root))


@lru_cache(maxsize=TASK_BUNDLE_CACHE_MAXSIZE)
def _load_task_bundle_cached(
    task_name: str,
    artifact_root: str,
    artifact_version: tuple[
        tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]
    ],
) -> dict[str, object]:
    resolved_artifact_root = Path(artifact_root)
    artifact_paths = resolve_artifact_paths(
        task_name, _resolve_runtime_artifact_root(task_name, resolved_artifact_root)
    )

    def require_artifact(artifact_name: str, artifact_path: Path, loader):
        if not artifact_path.exists():
            raise TaskArtifactError(task_name, artifact_name, artifact_path, "is missing")
        try:
            return loader(artifact_path)
        except Exception as exc:
            raise TaskArtifactError(task_name, artifact_name, artifact_path, "could not be read") from exc

    return {
        "task_config": config.get_task_config(task_name),
        "artifact_paths": artifact_paths,
        "model": require_artifact("model", artifact_paths.model_path, load_model),
        "comparison": require_artifact("metrics", artifact_paths.metrics_path, load_frame),
        "importance": require_artifact("importance", artifact_paths.importance_path, load_frame),
    }


def _validated_importance_features(task_name: str, importance: pd.DataFrame) -> list[str]:
    if "feature" not in importance.columns:
        raise TaskArtifactError(
            task_name,
            "importance",
            Path(f"{task_name}:importance"),
            "is malformed: missing feature column",
        )

    feature_values = importance["feature"]
    if feature_values.isna().any():
        raise TaskArtifactError(
            task_name,
            "importance",
            Path(f"{task_name}:importance"),
            "is malformed: feature contains empty values",
        )

    normalized_features = feature_values.astype(str).str.strip()
    if normalized_features.eq("").any():
        raise TaskArtifactError(
            task_name,
            "importance",
            Path(f"{task_name}:importance"),
            "is malformed: feature contains empty values",
        )

    return normalized_features.head(3).tolist()


def _validated_prediction_model_name(task_name: str, comparison: pd.DataFrame) -> str:
    model_name = _extract_model_name(comparison)
    if model_name is None:
        raise TaskArtifactError(
            task_name,
            "comparison",
            Path(f"{task_name}:comparison"),
            "is malformed: missing model name",
        )
    return model_name


def predict_failure_within_24h(raw_values: dict[str, float | str]) -> FailureWithin24hPredictionResponse:
    bundle = load_task_bundle("failure_within_24h")
    input_frame = build_input_frame(raw_values)
    probability = float(bundle["model"].predict_proba(input_frame)[:, 1][0])
    label, _ = summarize_probability(probability)
    importance_summary = _validated_importance_features("failure_within_24h", bundle["importance"])
    model_name = _validated_prediction_model_name("failure_within_24h", bundle["comparison"])
    return FailureWithin24hPredictionResponse(
        task_name="failure_within_24h",
        model_name=model_name,
        predicted_label=label,
        probability=probability,
        importance_summary=importance_summary,
    )


def predict_failure_type(raw_values: dict[str, float | str]) -> FailureTypePredictionResponse:
    bundle = load_task_bundle("failure_type")
    input_frame = build_input_frame(raw_values)
    probability_matrix = bundle["model"].predict_proba(input_frame)[0].tolist()
    predicted_class = str(bundle["model"].predict(input_frame)[0])
    probability_table = format_multiclass_probabilities(
        class_labels=bundle["model"].classes_.tolist(),
        probabilities=probability_matrix,
    )
    importance_summary = _validated_importance_features("failure_type", bundle["importance"])
    model_name = _validated_prediction_model_name("failure_type", bundle["comparison"])
    return FailureTypePredictionResponse(
        task_name="failure_type",
        model_name=model_name,
        predicted_class=predicted_class,
        class_probabilities=[
            ClassProbability(class_label=row.class_label, probability=float(row.probability))
            for row in probability_table.itertuples(index=False)
        ],
        importance_summary=importance_summary,
    )
