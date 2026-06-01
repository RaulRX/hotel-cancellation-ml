import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from src.predict import predict_dataset

router = APIRouter(prefix="/predict", tags=["predict"])
_executor = ThreadPoolExecutor(max_workers=1)


@router.post("")
async def predict():
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, predict_dataset)
        return {"status": "ok", "message": "Predictions saved to outputs/predictions.json"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
