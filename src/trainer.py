import logging
import shutil

import joblib

from src.config import (
    BEST_MODEL_PATH,
    MODELS_TESTS_DIR,
    PRIMARY_METRIC
)

from src.data_loader import prepare_dataset

from src.preprocess_data import MODEL_BUILDERS

logger = logging.getLogger(__name__)


def _clear_existing_models() -> None:
    """Remove best_model.pkl and all models in models/tests/ before retraining."""
    if BEST_MODEL_PATH.exists():
        BEST_MODEL_PATH.unlink()
    if MODELS_TESTS_DIR.exists():
        shutil.rmtree(MODELS_TESTS_DIR)
    MODELS_TESTS_DIR.mkdir(parents=True, exist_ok=True)


def train_models(hyperparams: dict[str, dict] | None = None) -> dict:
    """API-oriented training: clean (if needed) + train. No evaluation, no best model selection.

    Designed to be called by POST /train. Evaluation and best model selection
    happen separately via POST /evaluate.

    Parameters
    ----------
    hyperparams : dict[str, dict] | None
        Optional per-model hyperparameter overrides.

    Returns
    -------
    dict with keys:
        - trained_models : list[str]  — names of models saved to models/tests/
        - message        : str
    """
    hyperparams = hyperparams or {}

    if MODELS_TESTS_DIR.exists() and any(MODELS_TESTS_DIR.glob("*.pkl")):
        logger.info("Existing models found — clearing before retraining.")
        _clear_existing_models()

    X_train, _, y_train, _ = prepare_dataset(force_reprocess=False)
    logger.info("Split: %d train samples", len(X_train))

    trained = []
    for name, builder in MODEL_BUILDERS.items():
        logger.info("Training %s...", name)
        pipeline = builder(**hyperparams.get(name, {}))
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODELS_TESTS_DIR / f"{name}.pkl")
        logger.info("Saved %s", name)
        trained.append(name)

    return {
        "trained_models": trained,
        "message": f"{len(trained)} model(s) trained and saved to models/tests/.",
    }


def train_all(hyperparams: dict[str, dict] | None = None) -> dict:
    """CLI-oriented full pipeline: clean → preprocess → train → evaluate → select best.

    Designed to be run as a script: python -m src.trainer

    Parameters
    ----------
    hyperparams : dict[str, dict] | None
        Optional per-model hyperparameter overrides.

    Returns
    -------
    dict with keys:
        - best_model : str
        - metric     : str
        - results    : list[dict]  — comparison table with percentage metrics
    """
    from src.evaluator import evaluate_model

    hyperparams = hyperparams or {}

    tests_exist = MODELS_TESTS_DIR.exists() and any(MODELS_TESTS_DIR.glob("*.pkl"))
    if BEST_MODEL_PATH.exists() or tests_exist:
        logger.info("Existing models found — clearing before retraining.")
        _clear_existing_models()

    X_train, X_test, y_train, y_test = prepare_dataset(force_reprocess=True)
    logger.info("Split: %d train / %d test samples", len(X_train), len(X_test))

    metrics_per_model: dict[str, dict] = {}
    for name, builder in MODEL_BUILDERS.items():
        logger.info("Training %s...", name)
        pipeline = builder(**hyperparams.get(name, {}))
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODELS_TESTS_DIR / f"{name}.pkl")

        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics_per_model[name] = metrics
        logger.info(
            "%s — accuracy=%.4f  precision=%.4f  recall=%.4f  f1=%.4f  auc=%.4f",
            name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )

    best_name = max(metrics_per_model, key=lambda n: metrics_per_model[n][PRIMARY_METRIC])
    best_pipeline = joblib.load(MODELS_TESTS_DIR / f"{best_name}.pkl")
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    logger.info("Best model: %s (Accuracy=%.4f) saved to %s", best_name, metrics_per_model[best_name][PRIMARY_METRIC], BEST_MODEL_PATH)

    def _pct(value: float) -> str:
        return f"{round(value * 100, 2)}%"

    results = [
        {
            "model": name,
            "accuracy": _pct(m["accuracy"]),
            "precision": _pct(m["precision"]),
            "recall": _pct(m["recall"]),
            "f1": _pct(m["f1"]),
            "roc_auc": _pct(m["roc_auc"]),
            "is_best": name == best_name,
        }
        for name, m in sorted(
            metrics_per_model.items(),
            key=lambda kv: kv[1][PRIMARY_METRIC],
            reverse=True,
        )
    ]

    return {"best_model": best_name, "metric": PRIMARY_METRIC, "results": results}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    summary = train_all()
    print(f"\nBest model: {summary['best_model']} (by {summary['metric']})\n")
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}"
    print(header)
    print("-" * len(header))
    for row in summary["results"]:
        marker = " (*)" if row["is_best"] else ""
        print(
            f"{row['model']:<25} {row['accuracy']:>10} {row['precision']:>10}"
            f" {row['recall']:>10} {row['f1']:>10} {row['roc_auc']:>10}{marker}"
        )
