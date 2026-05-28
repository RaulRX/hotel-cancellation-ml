"""FastAPI entrypoint.

Run with:
    fastapi dev ./api/main.py

Interactive docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

from api.core.errors import logger

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

    @app.get("/")
    def root():
        logger.info("Acceso a la raíz de la API.")
        return {"message": "This is the Hotel Cancellation ML API root. Use /docs for more information."}

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(train.router)
    app.include_router(predict.router)
    app.include_router(evaluate.router)

    return app


app = create_app()
