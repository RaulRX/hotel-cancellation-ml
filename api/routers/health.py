"""Liveness/readiness probes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.services.model_service import ModelService, get_model_service

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(svc: ModelService = Depends(get_model_service)) -> dict[str, bool | str]:
    return {"ready": svc.artifact_path.exists(), "artifact_path": str(svc.artifact_path)}
