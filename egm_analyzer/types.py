from enum import Enum

from pydantic import BaseModel


Hz = int
Gb = int
MicroSecond = float
Index = float  # 200 microseconds


class Stage(str, Enum):
    PEAK_SEARCH = 'peak_search'
    PEAK_FIX = 'peak_fix'
    HUMAN_EDIT = 'human_edit'


class Peak(BaseModel):
    position: Index
    creation_stage: Stage
    search_segment: tuple[Index, Index] | None


class InferenceMeta(BaseModel):
    threshold: float
    path_to_model: str
    path_to_signal: str


class InferenceResult(BaseModel):
    peaks: list[list[Peak]]
    meta: InferenceMeta


class ErrorType(str, Enum):
    FALSE_POSITIVE = 'false_positive'
    FALSE_NEGATIVE = 'false_negative'


class Error(BaseModel):
    position: Index
    channel: int
    error_type: ErrorType


class Metrics(BaseModel):
    precision: float
    recall: float
    f1_score: float


class MetricsMeta(BaseModel):
    inference_result_path: str
    ground_truth_path: str


class MetricsResult(BaseModel):
    errors: list[Error]
    metrics: Metrics
    meta: MetricsMeta
