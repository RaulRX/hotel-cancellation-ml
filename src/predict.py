import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import MODELS_TESTS_DIR, PREDICTIONS_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN
from src.data_loader import load_processed_data

logger = logging.getLogger(__name__)


def make_predictions(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return np.asarray(y_pred), np.asarray(y_proba)


def predict_dataset() -> dict:
    candidates = sorted(MODELS_TESTS_DIR.glob("*.pkl"))
    if not candidates:
        raise FileNotFoundError("No trained models found. Run POST /train first.")

    df = load_processed_data(PROCESSED_DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y_true = df[TARGET_COLUMN].tolist()

    results = {}
    for model_file in candidates:
        logger.info("Running predictions with %s", model_file.name)
        pipeline: Pipeline = joblib.load(model_file)
        y_pred, y_proba = make_predictions(pipeline, X)
        results[model_file.stem] = {
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist(),
        }

    payload = {"y_true": y_true, "models": results}

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(json.dumps(payload))
    logger.info("Predictions persisted to %s", PREDICTIONS_PATH)

    return {model: {"predictions": v["predictions"]} for model, v in results.items()}
