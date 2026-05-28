"""Model service: lazy-loads the persisted pipeline and caches it in memory.

Thread-safe. The cached instance is invalidated whenever a new training run
finishes so subsequent /predict and /evaluate calls pick up the latest model
without restarting the process.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.config import PATHS

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, artifact_path: Path | None = None) -> None:
        self._artifact_path: Path = artifact_path or PATHS.best_model
        self._model: Pipeline | None = None
        self._lock = threading.RLock()

    @property
    def artifact_path(self) -> Path:
        return self._artifact_path

    def is_loaded(self) -> bool:
        return self._model is not None

    def invalidate(self) -> None:
        with self._lock:
            self._model = None
            logger.info("Model cache invalidated")

    def get(self) -> Pipeline:
        with self._lock:
            if self._model is None:
                if not self._artifact_path.exists():
                    raise FileNotFoundError(
                        f"No trained model found at {self._artifact_path}. Run /train first."
                    )
                logger.info("Loading model from %s", self._artifact_path)
                self._model = joblib.load(self._artifact_path)
            return self._model

    def describe(self) -> dict:
        loaded = self.is_loaded()
        info: dict = {"loaded": loaded, "artifact_path": str(self._artifact_path)}
        if loaded and isinstance(self._model, Pipeline):
            info["pipeline_steps"] = [name for name, _ in self._model.steps]
        return info


_singleton: ModelService | None = None
_singleton_lock = threading.Lock()


def get_model_service() -> ModelService:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ModelService()
    return _singleton
