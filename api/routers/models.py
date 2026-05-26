"""Model metadata endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas.evaluation import ModelInfo
from api.services.model_service import ModelService, get_model_service

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelInfo)
def model_info(svc: ModelService = Depends(get_model_service)) -> ModelInfo:
    return ModelInfo(**svc.describe())


@router.post("/reload")
def reload_model(svc: ModelService = Depends(get_model_service)) -> dict[str, str]:
    svc.invalidate()
    svc.get()
    return {"status": "reloaded", "artifact_path": str(svc.artifact_path)}
