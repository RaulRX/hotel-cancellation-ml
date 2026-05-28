"""Training endpoint: kicks off training as a BackgroundTask and tracks state.

POST /train returns 202 Accepted with a job_id immediately. GET /train/jobs/{id}
returns the live status. This avoids holding HTTP connections open for minutes
while models train.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from api.schemas.training import (
    JobStatus,
    TrainAcceptedResponse,
    TrainJob,
    TrainRequest,
)
from api.services.job_store import JobStore, get_job_store
from api.services.model_service import ModelService, get_model_service
from src.trainer import train_all

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/train", tags=["train"])


def _run_training_job(
    job_id: str,
    request: TrainRequest,
    store: JobStore,
    model_svc: ModelService,
) -> None:
    store.update(job_id, status=JobStatus.RUNNING, started_at=datetime.now(timezone.utc))
    try:
        result = train_all(
            dataset_path=request.dataset_path,
            primary_metric=request.primary_metric,
            random_state=request.random_state,
        )
        model_svc.invalidate()
        store.update(
            job_id,
            status=JobStatus.SUCCEEDED,
            finished_at=datetime.now(timezone.utc),
            primary_metric=result.primary_metric,
            best_model=result.best_model_name,
            best_metric_value=result.best_metric_value,
            leaderboard=result.leaderboard,
            artifact_path=result.artifact_path,
        )
        logger.info("Training job %s succeeded: best=%s", job_id, result.best_model_name)
    except Exception as exc:
        logger.exception("Training job %s failed", job_id)
        store.update(
            job_id,
            status=JobStatus.FAILED,
            finished_at=datetime.now(timezone.utc),
            error=f"{type(exc).__name__}: {exc}",
        )


@router.post(
    "",
    response_model=TrainAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Launch a training run (async)",
)
def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
    model_svc: ModelService = Depends(get_model_service),
) -> TrainAcceptedResponse:
    job = store.create()
    background_tasks.add_task(_run_training_job, job.job_id, request, store, model_svc)
    return TrainAcceptedResponse(
        job_id=job.job_id,
        status=job.status,
        status_url=f"/train/jobs/{job.job_id}",
    )


@router.get("/jobs/{job_id}", response_model=TrainJob, summary="Get training job status")
def get_job(job_id: str, store: JobStore = Depends(get_job_store)) -> TrainJob:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    return job


@router.get("/jobs", response_model=list[TrainJob], summary="List all training jobs")
def list_jobs(store: JobStore = Depends(get_job_store)) -> list[TrainJob]:
    return store.list_all()
