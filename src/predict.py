"""Inference helpers on top of a persisted sklearn Pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocess_data import drop_leakage_columns


def predict_dataframe(model: Pipeline, X: pd.DataFrame) -> dict[str, Any]:
    X = drop_leakage_columns(X)
    y_pred = model.predict(X)

    out: dict[str, Any] = {"predictions": y_pred.astype(int).tolist()}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        out["probabilities"] = np.asarray(proba, dtype=float).tolist()
    return out


def predict_records(model: Pipeline, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"predictions": [], "probabilities": []}
    return predict_dataframe(model, pd.DataFrame(records))
