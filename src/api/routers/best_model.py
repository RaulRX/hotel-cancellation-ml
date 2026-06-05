from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.trainer import train_all

router = APIRouter(prefix="/best-model", tags=["best_model"])


class BestModelRequest(BaseModel):
    hyperparams: dict[str, dict[str, Any]] | None = None
    """Optional per-model hyperparameter overrides.

    Keys must match available model names: logistic_regression, decision_tree, lightgbm.
    Example:
    {
        "lightgbm": {"num_leaves": 80, "n_estimators": 500},
        "decision_tree": {"max_depth": 6}
    }
    """


@router.get("")
def best_model():
    try:
        result = train_all()
        return {"status": "success", **result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
