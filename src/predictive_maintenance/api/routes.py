from fastapi import APIRouter, HTTPException, Query

from predictive_maintenance import config
from predictive_maintenance.api.schemas import (
    FailureTypePredictionRequest,
    FailureTypePredictionResponse,
    FailureWithin24hPredictionRequest,
    FailureWithin24hPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from predictive_maintenance.api.service import (
    TaskArtifactError,
    build_model_info_payload,
    load_task_bundle,
    predict_failure_type,
    predict_failure_within_24h,
)


router = APIRouter()


def _validated_task_name(task_name: str) -> str:
    if task_name not in config.TASK_CONFIGS:
        raise HTTPException(status_code=422, detail=f"Unknown task: {task_name}")
    return task_name


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    available_tasks: list[str] = []
    for task_name in config.TASK_CONFIGS:
        try:
            load_task_bundle(task_name)
            build_model_info_payload(task_name)
        except TaskArtifactError:
            continue
        available_tasks.append(task_name)
    status = "ok" if len(available_tasks) == len(config.TASK_CONFIGS) else "degraded"
    return HealthResponse(status=status, available_tasks=available_tasks)


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    task_name: str = Query(config.DEFAULT_TASK_NAME),
) -> ModelInfoResponse:
    task_name = _validated_task_name(task_name)
    try:
        return build_model_info_payload(task_name)
    except TaskArtifactError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "task_name": exc.task_name,
                "artifact_name": exc.artifact_name,
                "reason": exc.reason,
            },
        ) from exc


@router.post("/predict/failure-within-24h", response_model=FailureWithin24hPredictionResponse)
def predict_binary(payload: FailureWithin24hPredictionRequest) -> FailureWithin24hPredictionResponse:
    try:
        return predict_failure_within_24h(payload.model_dump())
    except TaskArtifactError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "task_name": exc.task_name,
                "artifact_name": exc.artifact_name,
                "reason": exc.reason,
            },
        ) from exc


@router.post("/predict/failure-type", response_model=FailureTypePredictionResponse)
def predict_multiclass(payload: FailureTypePredictionRequest) -> FailureTypePredictionResponse:
    try:
        return predict_failure_type(payload.model_dump())
    except TaskArtifactError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "task_name": exc.task_name,
                "artifact_name": exc.artifact_name,
                "reason": exc.reason,
            },
        ) from exc
