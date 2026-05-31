from fastapi import APIRouter, HTTPException

from src.trainer import train_all

router = APIRouter(prefix="/train", tags=["train"])


@router.post("")
def train():
    try:
        train_all()
        return {"status": "success", "message": "Modelo entrenado y guardado en archivo pkl."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
