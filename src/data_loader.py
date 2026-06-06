import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_PATH, RANDOM_STATE, RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE
from src.preprocess_data import clean_dataset

logger = logging.getLogger(__name__)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset from CSV, treating 'NULL' strings as NaN."""
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path, na_values=["NULL"])
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def save_processed_data(df: pd.DataFrame, path=PROCESSED_DATA_PATH) -> None:
    """Persist the cleaned dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Processed data saved to %s (%d rows)", path, len(df))


def load_processed_data(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the previously saved processed dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}. Run training first.")
    logger.info("Loading processed data from %s", path)
    return pd.read_csv(path)

def prepare_dataset(force_reprocess: bool = False):
    """Load and optionally reprocess the dataset. Returns (X_train, X_test, y_train, y_test)."""
    if force_reprocess or not PROCESSED_DATA_PATH.exists():
        logger.info("Loading and cleaning raw data...")
        df = clean_dataset(load_raw_data())
        save_processed_data(df)
    else:
        logger.info("Reusing existing processed dataset.")
        df = load_processed_data()

    return __split_dataset(df)

def __split_dataset(df: pd.DataFrame):
    """Reproducible train/test split shared by training and prediction.

    Deterministic given fixed RANDOM_STATE/TEST_SIZE and stratify=y, so the
    same train subset can be reconstructed outside training.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
