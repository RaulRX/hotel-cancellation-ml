"""API error model + exception handlers.

Maps domain exceptions (model not found, dataset not found, invalid request) to
consistent HTTP responses with a structured body. Avoids leaking internal stack
traces while keeping useful diagnostic info.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.data_loader import DatasetNotFoundError

logger = logging.getLogger(__name__)


class ErrorBody(BaseModel):
    error: str
    detail: str


def _json(status_code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ErrorBody(error=error, detail=detail).model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(FileNotFoundError)
    async def _file_not_found(_: Request, exc: FileNotFoundError) -> JSONResponse:
        if isinstance(exc, DatasetNotFoundError):
            return _json(status.HTTP_400_BAD_REQUEST, "dataset_not_found", str(exc))
        return _json(status.HTTP_409_CONFLICT, "model_not_trained", str(exc))

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return _json(status.HTTP_400_BAD_REQUEST, "invalid_input", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _json(status.HTTP_422_UNPROCESSABLE_ENTITY, "validation_error", str(exc.errors()))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return _json(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "An unexpected error occurred.")
