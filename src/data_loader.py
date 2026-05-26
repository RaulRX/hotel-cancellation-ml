"""Dataset loading and basic train/test splitting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PATHS, TRAINING


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the configured dataset CSV cannot be located."""


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else PATHS.dataset_csv
    if not csv_path.exists():
        raise DatasetNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    if TRAINING.target not in df.columns:
        raise ValueError(
            f"Target column '{TRAINING.target}' not present in dataset columns: {list(df.columns)[:10]}..."
        )
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TRAINING.target].astype(int)
    X = df.drop(columns=[TRAINING.target])
    return X, y


def train_test_split_df(
    df: pd.DataFrame,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = split_features_target(df)
    return train_test_split(
        X,
        y,
        test_size=test_size or TRAINING.test_size,
        random_state=random_state or TRAINING.random_state,
        stratify=y,
    )
