"""Project configuration: paths, target, primary metric, model registry layout.

Centralizes constants so the rest of the codebase (and the API) does not hardcode
paths or magic strings. Values can be overridden via environment variables for
deployment without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


def _env_path(var: str, default: Path) -> Path:
    raw = os.getenv(var)
    return Path(raw).resolve() if raw else default


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data_raw: Path = field(default_factory=lambda: _env_path("DATA_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
    data_processed: Path = field(default_factory=lambda: _env_path("DATA_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
    models: Path = field(default_factory=lambda: _env_path("MODELS_DIR", PROJECT_ROOT / "models"))
    outputs: Path = field(default_factory=lambda: _env_path("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

    @property
    def dataset_csv(self) -> Path:
        return Path(os.getenv("DATASET_CSV", str(self.data_raw / "dataset.csv")))

    @property
    def best_model(self) -> Path:
        return self.models / "best_model.pkl"

    @property
    def metrics_json(self) -> Path:
        return self.outputs / "metrics.json"


@dataclass(frozen=True)
class TrainingConfig:
    target: str = "is_canceled"
    test_size: float = 0.2
    random_state: int = 42
    primary_metric: str = "f1"  # f1 chosen: class imbalance + asymmetric error cost
    cv_folds: int = 0  # 0 = simple holdout; >1 enables cross-validation


PATHS = Paths()
TRAINING = TrainingConfig()
