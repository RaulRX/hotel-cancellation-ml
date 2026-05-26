"""In-memory job store for training tasks.

Trade-off: jobs are lost on process restart. For a single-instance educational
deployment this is acceptable; for multi-instance production replace with Redis
or a database without touching the router code.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

from api.schemas.training import JobStatus, TrainJob


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, TrainJob] = {}
        self._lock = threading.RLock()

    def create(self) -> TrainJob:
        job = TrainJob(
            job_id=str(uuid.uuid4()),
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> TrainJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> TrainJob:
        with self._lock:
            current = self._jobs[job_id]
            updated = current.model_copy(update=fields)
            self._jobs[job_id] = updated
            return updated

    def list_all(self) -> list[TrainJob]:
        with self._lock:
            return list(self._jobs.values())


_singleton: JobStore | None = None
_singleton_lock = threading.Lock()


def get_job_store() -> JobStore:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = JobStore()
    return _singleton
