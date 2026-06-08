import logging

from fastapi import FastAPI

from src.api.routers import best_model, train, evaluate, predict, tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(title="Hotel Cancellation ML API", version="1.0.0")

app.include_router(tools.router)
app.include_router(best_model.router)
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(evaluate.router)
