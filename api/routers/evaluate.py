from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.evaluator import evaluate_all

router = APIRouter(prefix="/evaluate", tags=["evaluate"])


class EvaluateRequest(BaseModel):
    dataset_path: str | None = None


@router.post("")
def evaluate(request: EvaluateRequest):
    try:
        return evaluate_all(dataset_path=request.dataset_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
