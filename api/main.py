"""FastAPI entrypoint.

Run with:
    uvicorn api.main:app --reload

Interactive docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

import os

from fastapi import FastAPI

from api.core.errors import register_exception_handlers
from api.core.logging import configure_logging
from api.routers import evaluate, health, models, predict, train

configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hotel Cancellation ML API",
        description=(
            "REST API for the hotel booking cancellation classifier. "
            "Exposes training, inference and evaluation over a binary classification pipeline."
        ),
        version="0.1.0",
        contact={"name": "Hotel Cancellation ML project"},
    )
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(train.router)
    app.include_router(predict.router)
    app.include_router(evaluate.router)

    return app


app = create_app()
