"""Preprocessing pipeline builder for the hotel cancellation dataset.

The builder returns a ``ColumnTransformer`` that handles numeric and categorical
features. It is intentionally generic: column lists are inferred from the input
``DataFrame`` so the same pipeline works for the raw dataset and for new
inference payloads.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LEAKAGE_COLUMNS = ("reservation_status", "reservation_status_date")


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric, categorical


def drop_leakage_columns(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in LEAKAGE_COLUMNS if c in X.columns]
    return X.drop(columns=cols) if cols else X


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = infer_feature_types(X)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
