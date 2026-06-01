from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_TESTS_DIR = MODELS_DIR / "tests"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.json"

TARGET_COLUMN = "is_canceled"
RANDOM_STATE = 11
TEST_SIZE = 0.2
PRIMARY_METRIC = "f1"