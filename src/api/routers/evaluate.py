from fastapi import APIRouter, HTTPException

from src.evaluator import evaluate_all

router = APIRouter(prefix="/evaluate", tags=["evaluate"])


@router.post("")
def evaluate():
    try:
        return evaluate_all()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
