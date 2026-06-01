import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import BEST_MODEL_PATH

logger = logging.getLogger(__name__)


def make_predictions(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run predict and predict_proba through a fitted pipeline.

    Used internally by trainer and evaluator — the pipeline is already loaded
    in memory, so no disk access happens here.

    Parameters
    ----------
    model : fitted sklearn Pipeline (preprocessor + estimator)
    X : raw DataFrame (no preprocessing applied yet — the pipeline handles it)

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
    y_proba : ndarray of shape (n_samples,)  — P(is_canceled=1)
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return np.asarray(y_pred), np.asarray(y_proba)


def predict_records(records: list[dict], model_path: Path = BEST_MODEL_PATH) -> dict:
    """Load a persisted pipeline and predict over raw booking records.

    Intended for the API endpoint — loads the pipeline from disk, builds a
    DataFrame from the raw input records and delegates to make_predictions.
    The pipeline internally handles all preprocessing before inference.

    Parameters
    ----------
    records : list[dict]
        Raw booking records (BookingRecord.model_dump() from the API request).
    model_path : Path
        Path to a persisted pipeline (.pkl). Defaults to best_model.pkl.

    Returns
    -------
    dict with keys:
        - predictions   : list[int]    — binary class per record (0 or 1)
        - probabilities : list[float]  — P(is_canceled=1) per record
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run POST /train first."
        )

    logger.info("Loading pipeline from %s", model_path)
    pipeline: Pipeline = joblib.load(model_path)

    X = pd.DataFrame(records)
    y_pred, y_proba = make_predictions(pipeline, X)

    return {
        "predictions": y_pred.tolist(),
        "probabilities": y_proba.tolist(),
    }
