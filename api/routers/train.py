from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.trainer import train_all

router = APIRouter(prefix="/train", tags=["train"])


class TrainRequest(BaseModel):
    dataset_path: str | None = None
    primary_metric: str | None = None


@router.post("")
def train(request: TrainRequest):
    try:
        result = train_all(
            dataset_path=request.dataset_path,
            primary_metric=request.primary_metric,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
