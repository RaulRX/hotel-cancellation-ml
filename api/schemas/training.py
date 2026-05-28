"""Pydantic schemas for training endpoints."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_path: str | None = Field(
        default=None,
        description="Optional override of the dataset CSV path. Defaults to the project's configured dataset.",
    )
    primary_metric: str | None = Field(
        default=None,
        description="Metric used to select the best model. One of: accuracy, precision, recall, f1, roc_auc.",
        examples=["f1"],
    )
    random_state: int | None = Field(default=None, ge=0, description="Seed for reproducibility.")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TrainJob(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    primary_metric: str | None = None
    best_model: str | None = None
    best_metric_value: float | None = None
    leaderboard: list[dict[str, Any]] | None = None
    artifact_path: str | None = None
    error: str | None = None


class TrainAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
