"""Model training: fit several binary classifiers and pick the best one.

The trainer wraps each estimator behind the same preprocessing ``ColumnTransformer``
inside a single ``Pipeline``, so the persisted artifact is self-contained and can
be used for inference without re-running preprocessing logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import PATHS, TRAINING
from src.data_loader import load_dataset, train_test_split_df
from src.evaluator import evaluate_model
from src.preprocess_data import build_preprocessor, drop_leakage_columns

logger = logging.getLogger(__name__)


def _candidate_estimators(random_state: int) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1000, n_jobs=None, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }
    try:
        from xgboost import XGBClassifier  # noqa: WPS433

        candidates["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    except ImportError:
        logger.info("xgboost not available — skipping XGBClassifier")
    return candidates


@dataclass
class TrainingResult:
    best_model_name: str
    best_metric_value: float
    primary_metric: str
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    artifact_path: str = ""


def train_all(
    dataset_path: str | None = None,
    primary_metric: str | None = None,
    random_state: int | None = None,
) -> TrainingResult:
    metric = primary_metric or TRAINING.primary_metric
    seed = random_state if random_state is not None else TRAINING.random_state

    df = load_dataset(dataset_path) if dataset_path else load_dataset()
    df = drop_leakage_columns(df)

    X_train, X_test, y_train, y_test = train_test_split_df(df, random_state=seed)
    preprocessor = build_preprocessor(X_train)

    leaderboard: list[dict[str, Any]] = []
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_score = float("-inf")

    for name, estimator in _candidate_estimators(seed).items():
        logger.info("Training %s", name)
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        leaderboard.append({"model": name, **metrics})

        score = metrics.get(metric)
        if score is None:
            raise ValueError(f"Primary metric '{metric}' not produced by evaluator")
        if score > best_score:
            best_score, best_name, best_pipeline = score, name, pipe

    if best_pipeline is None or best_name is None:
        raise RuntimeError("No models were trained successfully")

    PATHS.models.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, PATHS.best_model)
    logger.info("Best model: %s (%s=%.4f) → %s", best_name, metric, best_score, PATHS.best_model)

    leaderboard.sort(key=lambda row: row[metric], reverse=True)
    return TrainingResult(
        best_model_name=best_name,
        best_metric_value=float(best_score),
        primary_metric=metric,
        leaderboard=leaderboard,
        artifact_path=str(PATHS.best_model),
    )


def load_best_model() -> Pipeline:
    if not PATHS.best_model.exists():
        raise FileNotFoundError(
            f"No trained model found at {PATHS.best_model}. Run training first via /train."
        )
    return joblib.load(PATHS.best_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    result = train_all()
    print(result)
