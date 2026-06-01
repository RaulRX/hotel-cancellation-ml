import logging

from fastapi import FastAPI

from api.routers import evaluate, predict, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(title="Hotel Cancellation ML API", version="0.1.0")

app.include_router(train.router)
app.include_router(predict.router)
app.include_router(evaluate.router)
