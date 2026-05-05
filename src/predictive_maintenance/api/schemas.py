from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictStr, create_model

from predictive_maintenance.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


_REQUEST_FIELDS = {
    **{column: (StrictFloat, ...) for column in NUMERIC_COLUMNS},
    **{column: (StrictStr, ...) for column in CATEGORICAL_COLUMNS},
}


SharedPredictionRequest = create_model(  # type: ignore[call-overload]
    "SharedPredictionRequest",
    __base__=BaseModel,
    __config__={"extra": "forbid"},
    **_REQUEST_FIELDS,
)


class FailureWithin24hPredictionRequest(SharedPredictionRequest):
    pass


class FailureTypePredictionRequest(SharedPredictionRequest):
    pass


class HealthResponse(BaseModel):
    status: str
    available_tasks: list[str]


class ModelInfoComparisonRow(BaseModel):
    model_name: str
    primary_metric: float
    metric_two: float
    metric_three: float
    is_best_model: bool


class ModelInfoTask(BaseModel):
    task_name: str
    model_name: str | None
    primary_metric: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str = ""
    primary_metric_name: str = ""
    best_model_name: str = ""
    models_comparison: list[ModelInfoComparisonRow] = Field(default_factory=list)
    tasks: list[ModelInfoTask] = Field(default_factory=list)


class FailureWithin24hPredictionResponse(BaseModel):
    task_name: str
    model_name: str
    predicted_label: str
    probability: float
    importance_summary: list[str]


class ClassProbability(BaseModel):
    class_label: str
    probability: float


class FailureTypePredictionResponse(BaseModel):
    task_name: str
    model_name: str
    predicted_class: str
    class_probabilities: list[ClassProbability]
    importance_summary: list[str]
