import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    BEST_MODEL_PATH,
    MODELS_TESTS_DIR,
    OUTPUTS_DIR,
    PRIMARY_METRIC,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_processed_data
from src.predict import make_predictions

logger = logging.getLogger(__name__)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute classification metrics for a single fitted pipeline.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc,
                    confusion_matrix (list[list[int]]),
                    fpr (list[float]), tpr (list[float])
    """
    y_pred, y_proba = make_predictions(model, X_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "fpr": roc_curve(y_test, y_proba)[0].tolist(),
        "tpr": roc_curve(y_test, y_proba)[1].tolist(),
    }


def _plot_roc_curves(results: dict[str, dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, metrics in results.items():
        ax.plot(metrics["fpr"], metrics["tpr"], label=f"{name} (AUC={metrics['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — all models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved roc_curves.png")


def _plot_confusion_matrix(name: str, cm: list[list[int]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not canceled", "Canceled"],
        yticklabels=["Not canceled", "Canceled"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved confusion_matrix_%s.png", name)


def _plot_feature_importance(name: str, model: Pipeline, output_dir: Path) -> None:
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        importance_type = "Feature Importance"
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
        importance_type = "|Coefficient|"
    else:
        logger.info("Skipping feature importance for %s — not available", name)
        return

    # Retrieve feature names from the last transformer before the model
    try:
        feature_names = model[:-1].get_feature_names_out()
    except AttributeError:
        feature_names = [f"f{i}" for i in range(len(importances))]

    top_n = min(20, len(importances))
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[indices], importances[indices])
    ax.set_xlabel(importance_type)
    ax.set_title(f"Top {top_n} Features — {name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"feature_importance_{name}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved feature_importance_%s.png", name)


def evaluate_all(dataset_path: str | None = None) -> list[dict]:
    """Evaluate all models in models/tests/ and return the comparison table.

    Loads the processed dataset, recreates the same train/test split used
    during training, evaluates each saved pipeline on X_test, saves output
    plots to outputs/, and returns the comparison table ordered by F1 desc.

    Parameters
    ----------
    dataset_path : str | None
        Optional override path to the processed dataset CSV.

    Returns
    -------
    list[dict] — one entry per model, keys: model, accuracy, precision,
                 recall, f1, roc_auc, is_best
    """
    path = Path(dataset_path) if dataset_path else PROCESSED_DATA_PATH
    df = load_processed_data(path)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model_files = sorted(MODELS_TESTS_DIR.glob("*.pkl"))
    if not model_files:
        raise FileNotFoundError(
            f"No trained models found in {MODELS_TESTS_DIR}. Run POST /train first."
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for model_file in model_files:
        name = model_file.stem
        logger.info("Evaluating %s", name)
        pipeline: Pipeline = joblib.load(model_file)
        results[name] = evaluate_model(pipeline, X_test, y_test)

    # Save plots
    _plot_roc_curves(results, OUTPUTS_DIR)
    for name, metrics in results.items():
        _plot_confusion_matrix(name, metrics["confusion_matrix"], OUTPUTS_DIR)
        pipeline = joblib.load(MODELS_TESTS_DIR / f"{name}.pkl")
        _plot_feature_importance(name, pipeline, OUTPUTS_DIR)

    def _pct(value: float) -> str:
        return f"{round(value * 100, 2)}%"

    best_name = max(results, key=lambda n: results[n][PRIMARY_METRIC])
    best_pipeline = joblib.load(MODELS_TESTS_DIR / f"{best_name}.pkl")
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    logger.info("Best model: %s — saved to %s", best_name, BEST_MODEL_PATH)
    table = [
        {
            "model": name,
            "accuracy": _pct(metrics["accuracy"]),
            "precision": _pct(metrics["precision"]),
            "recall": _pct(metrics["recall"]),
            "f1": _pct(metrics["f1"]),
            "roc_auc": _pct(metrics["roc_auc"]),
            "is_best": name == best_name,
        }
        for name, metrics in sorted(
            results.items(), key=lambda kv: kv[1][PRIMARY_METRIC], reverse=True
        )
    ]

    logger.info("Best model by %s: %s", PRIMARY_METRIC, best_name)
    return table
