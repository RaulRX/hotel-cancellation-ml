"""Pydantic schemas for evaluation endpoints."""
from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    dataset_path: str | None = Field(
        default=None,
        description="Optional dataset CSV path. Defaults to the project's configured dataset.",
    )
    test_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    random_state: int | None = Field(default=None, ge=0)


class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int


class EvaluateResponse(BaseModel):
    model: str
    n_samples: int
    metrics: dict[str, float]
    confusion_matrix: ConfusionMatrix


class ModelInfo(BaseModel):
    loaded: bool
    artifact_path: str
    pipeline_steps: list[str] | None = None
