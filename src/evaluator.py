import json
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
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    BEST_MODEL_PATH,
    MODELS_TESTS_DIR,
    OUTPUTS_DIR,
    PREDICTIONS_PATH,
    PRIMARY_METRIC,
)

logger = logging.getLogger(__name__)


def _compute_metrics(y_true: list, y_pred: list, y_proba: list) -> dict:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_proba_arr = np.array(y_proba)

    fpr, tpr, _ = roc_curve(y_true_arr, y_proba_arr)
    return {
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred_arr)), 4),
        "precision": round(float(precision_score(y_true_arr, y_pred_arr, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true_arr, y_pred_arr, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true_arr, y_pred_arr, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true_arr, y_proba_arr)), 4),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

def _compute_metrics_ANN(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    estimator = model.named_steps["model"]
    X_transformed = model[:-1].transform(X_test)

    loss, accuracy = estimator.model_.evaluate(X_transformed, y_test, verbose=1)
    logger.info("Test Loss: %.4f, Test Accuracy: %.4f", loss, accuracy)

    y_prob_raw = np.asarray(model.predict(X_test)).ravel()
    y_pred = (y_prob_raw > 0.5).astype(int)

    metrics = _compute_metrics(y_test.tolist(), y_pred.tolist(), y_prob_raw.tolist())
    metrics["loss"] = round(float(loss), 4)
    return metrics


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    if isinstance(model.named_steps["model"], KerasClassifier):
        return _compute_metrics_ANN(model, X_test, y_test)
    y_pred = np.asarray(model.predict(X_test))
    y_proba = np.asarray(model.predict_proba(X_test)[:, 1])
    return _compute_metrics(y_test.tolist(), y_pred.tolist(), y_proba.tolist())


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


def _plot_loss_curve(name: str, model: Pipeline, output_dir: Path) -> None:
    history = model.named_steps["model"].history_

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history["loss"], label="Loss train")
    ax.plot(history["val_loss"], label="Loss val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Curva de aprendizaje — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"loss_curve_{name}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved loss_curve_%s.png", name)


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


def evaluate_all() -> list[dict]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError("Predictions not found. Run POST /predict first.")

    payload = json.loads(PREDICTIONS_PATH.read_text())
    y_true = payload["y_true"]
    models_data = payload["models"]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name, data in models_data.items():
        logger.info("Evaluating %s", name)
        results[name] = _compute_metrics(y_true, data["predictions"], data["probabilities"])

    _plot_roc_curves(results, OUTPUTS_DIR)
    for name, metrics in results.items():
        _plot_confusion_matrix(name, metrics["confusion_matrix"], OUTPUTS_DIR)
        pipeline: Pipeline = joblib.load(MODELS_TESTS_DIR / f"{name}.pkl")
        _plot_feature_importance(name, pipeline, OUTPUTS_DIR)
        if isinstance(pipeline.named_steps["model"], KerasClassifier):
            _plot_loss_curve(name, pipeline, OUTPUTS_DIR)

    best_name = max(results, key=lambda n: results[n][PRIMARY_METRIC])
    best_pipeline = joblib.load(MODELS_TESTS_DIR / f"{best_name}.pkl")
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    logger.info("Best model: %s — saved to %s", best_name, BEST_MODEL_PATH)

    def _pct(v: float) -> str:
        return f"{round(v * 100, 2)}%"

    return [
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
