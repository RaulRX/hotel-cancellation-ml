from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/predict", tags=["predict"])


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]


@router.post("")
def predict(request: PredictRequest):
    raise NotImplementedError("")
