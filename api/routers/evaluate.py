"""Evaluation endpoint: replays a holdout split against the persisted model."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas.evaluation import ConfusionMatrix, EvaluateRequest, EvaluateResponse
from api.services.model_service import ModelService, get_model_service
from src.data_loader import load_dataset, train_test_split_df
from src.evaluator import confusion_matrix_dict, evaluate_model
from src.preprocess_data import drop_leakage_columns

router = APIRouter(prefix="/evaluate", tags=["evaluate"])


@router.post(
    "",
    response_model=EvaluateResponse,
    summary="Evaluate the persisted model on a holdout split",
)
def evaluate(
    request: EvaluateRequest,
    svc: ModelService = Depends(get_model_service),
) -> EvaluateResponse:
    model = svc.get()
    df = drop_leakage_columns(load_dataset(request.dataset_path))
    _, X_test, _, y_test = train_test_split_df(
        df,
        test_size=request.test_size,
        random_state=request.random_state,
    )

    metrics = evaluate_model(model, X_test, y_test)
    cm = confusion_matrix_dict(model, X_test, y_test)
    return EvaluateResponse(
        model=svc.artifact_path.name,
        n_samples=int(len(y_test)),
        metrics=metrics,
        confusion_matrix=ConfusionMatrix(**cm),
    )
