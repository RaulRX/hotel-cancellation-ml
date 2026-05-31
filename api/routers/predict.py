import joblib

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/predict", tags=["predict"])

MODEL_PATH = "models/best_model.pkl"


class BookingRecord(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float | None = None
    babies: int
    meal: str
    country: str | None = None
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    deposit_type: str
    agent: str | None = None
    company: str | None = None
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int


class PredictRequest(BaseModel):
    records: list[BookingRecord]


@router.post("")
def predict(request: PredictRequest):
    try:
        pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="El modelo no ha sido entrenado aún. Llama primero a POST /train.",
        )

    try:
        data = [record.model_dump() for record in request.records]
        predictions = pipeline.predict(data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
