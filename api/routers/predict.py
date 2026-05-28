"""Synchronous prediction endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas.prediction import PredictRequest, PredictResponse
from api.services.model_service import ModelService, get_model_service
from src.predict import predict_records

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse, summary="Run inference on a batch of records")
def predict(
    request: PredictRequest,
    svc: ModelService = Depends(get_model_service),
) -> PredictResponse:
    model = svc.get()
    result = predict_records(model, request.records)
    return PredictResponse(
        model=svc.artifact_path.name,
        n=len(request.records),
        predictions=result["predictions"],
        probabilities=result.get("probabilities"),
    )
