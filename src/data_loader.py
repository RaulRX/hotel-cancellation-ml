import logging

import pandas as pd

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH

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
