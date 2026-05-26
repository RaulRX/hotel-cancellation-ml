"""Pydantic schemas for prediction endpoints.

Booking records are intentionally typed as ``dict[str, Any]`` rather than a strict
schema: the input dataset has 30+ columns with mixed types and OneHotEncoder is
configured to ignore unknown categories. A strict schema would force the API to
mirror every dataset column and break on dataset changes.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of booking records (column → value). Must include the same feature columns used at training time.",
    )


class PredictResponse(BaseModel):
    model: str = Field(description="Name of the persisted model artifact used for inference.")
    n: int = Field(description="Number of records scored.")
    predictions: list[int]
    probabilities: list[float] | None = None
