from enum import Enum

from pydantic import BaseModel


Hz = int
Gb = int
MicroSecond = float
Index = float  # 200 microseconds


class Stage(str, Enum):
    PEAK_SEARCH = 'peak_search'
    PEAK_FIX = 'peak_fix'


class Peak(BaseModel):
    position: Index
    creation_stage: Stage
    search_segment: tuple[Index, Index] | None


class Meta(BaseModel):
    threshold: float
    path_to_model: str
    path_to_signal: str


class InferenceResult(BaseModel):
    peaks: list[list[Peak]]
    meta: Meta
