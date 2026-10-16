"""Classical machine learning baselines: Logistic Regression, RBF-SVM, and Random Forest.

Author: Jashan Goyal (IIT2024223)
"""

from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    DEFAULT_CHECKPOINTS_DIR,
)
from acforensics.evaluation.metrics import full_evaluation
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def build_predefined_split(n_train: int, n_dev: int) -> PredefinedSplit:
    """Constructs a PredefinedSplit where train index = -1 and dev index = 0.

    Ensures hyperparameter tuning is strictly performed on the validation (dev) split,
    never on the test (eval) split.
    """
    test_fold = np.concatenate([np.full(n_train, -1), np.full(n_dev, 0)])
    return PredefinedSplit(test_fold)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    param_grid: Optional[dict[str, list[Any]]] = None,
    seed: int = 42,
) -> Pipeline:
    """Tunes and trains Logistic Regression with L2 regularization and balanced weights.

    Args:
        X_train, y_train: Training features and labels.
        X_dev, y_dev: Dev features and labels.
        param_grid: Tuning values for regularization strength C.
        seed: RNG seed.

    Returns:
        Fitted scikit-learn Pipeline with StandardScaler and LogisticRegression.
    """
    if param_grid is None:
        param_grid = {"classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=seed)),
    ])

    X_all = np.vstack([X_train, X_dev])
    y_all = np.concatenate([y_train, y_dev])
    cv = build_predefined_split(len(X_train), len(X_dev))

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    grid.fit(X_all, y_all)

    logger.info("Best Logistic Regression params: %s (Dev AUC: %.4f)", grid.best_params_, grid.best_score_)
    return grid.best_estimator_


def train_rbf_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    max_train_samples: int = 10000,
    param_grid: Optional[dict[str, list[Any]]] = None,
    seed: int = 42,
) -> Pipeline:
    """Trains RBF-SVM with quadratic complexity protection and stratified subsampling.

    If training sample count exceeds max_train_samples, uses a stratified subset
    to avoid O(N^2) training hang, while preserving class balance.

    Args:
        X_train, y_train: Training split.
        X_dev, y_dev: Dev split.
        max_train_samples: Maximum instances before subsampling.
        param_grid: Tuning grid for C and gamma.
        seed: RNG seed.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    if param_grid is None:
        param_grid = {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", 0.01, 0.1],
        }

    # Quadratic protection subsampling
    if len(X_train) > max_train_samples:
        logger.warning(
            "Subsampling SVM training set from %d to %d for quadratic complexity safety.",
            len(X_train),
            max_train_samples,
        )
        rng = np.random.RandomState(seed)
        bf_idx = np.where(y_train == BONAFIDE)[0]
        sp_idx = np.where(y_train == SPOOF)[0]

        ratio = len(bf_idx) / len(y_train)
        n_bf = int(max_train_samples * ratio)
        n_sp = max_train_samples - n_bf

        sub_bf = rng.choice(bf_idx, size=n_bf, replace=False)
        sub_sp = rng.choice(sp_idx, size=n_sp, replace=False)
        sub_idx = np.concatenate([sub_bf, sub_sp])
        rng.shuffle(sub_idx)

        X_tr = X_train[sub_idx]
        y_tr = y_train[sub_idx]
    else:
        X_tr = X_train
        y_tr = y_train

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
    ])

    X_all = np.vstack([X_tr, X_dev])
    y_all = np.concatenate([y_tr, y_dev])
    cv = build_predefined_split(len(X_tr), len(X_dev))

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    grid.fit(X_all, y_all)

    logger.info("Best RBF-SVM params: %s (Dev AUC: %.4f)", grid.best_params_, grid.best_score_)
    return grid.best_estimator_


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    param_grid: Optional[dict[str, list[Any]]] = None,
    seed: int = 42,
) -> Pipeline:
    """Tunes and trains Random Forest with balanced subsampling.

    Args:
        X_train, y_train: Training set.
        X_dev, y_dev: Dev set.
        param_grid: Tuning grid for n_estimators and max_depth.
        seed: RNG seed.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    if param_grid is None:
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20, None],
        }

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
    ])

    X_all = np.vstack([X_train, X_dev])
    y_all = np.concatenate([y_train, y_dev])
    cv = build_predefined_split(len(X_train), len(X_dev))

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    grid.fit(X_all, y_all)

    logger.info("Best Random Forest params: %s (Dev AUC: %.4f)", grid.best_params_, grid.best_score_)
    return grid.best_estimator_


def evaluate_classical_model(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    dev_threshold: Optional[float] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generates continuous scores for class 1 (bona fide) and computes full evaluation metrics."""
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X).astype(float)

    metrics_report = full_evaluation(y, scores, dev_threshold=dev_threshold, compute_ci=True)
    return scores, metrics_report
