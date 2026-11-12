"""Classical machine learning baselines: Logistic Regression, RBF-SVM, and Random Forest.

Upgraded with:
- Utterance-level statistical pooling (mean and std of frame features + deltas).
- StandardScaler fitted strictly on TRAIN split only.
- Principal Component Analysis (PCA) keeping 95% variance (configurable), fitted strictly on TRAIN split only.
- Log of the number of components retained by PCA.
- Hyperparameter tuning strictly performed on DEV split.
- Unified continuous scoring function predict_classical_score() where higher value = more bona fide.

"""

from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    DEFAULT_CHECKPOINTS_DIR,
)
from acforensics.evaluation.metrics import full_evaluation
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def extract_mean_std_features(frames: np.ndarray) -> np.ndarray:
    """Computes fixed-size utterance representation: frame mean and std.

    Args:
        frames: 2D array of shape (F, T) or (T, F).

    Returns:
        1D array of shape (2 * F,) containing concatenated mean and std.
    """
    if frames.ndim != 2:
        raise ValueError(f"Expected 2D frames matrix, got shape {frames.shape}")

    if frames.shape[0] > frames.shape[1] and frames.shape[1] in (20, 60, 80):
        frames = frames.T

    mean_vec = np.mean(frames, axis=1)
    std_vec = np.std(frames, axis=1)
    return np.concatenate([mean_vec, std_vec]).astype(np.float32)


def predict_classical_score(model: Any, X: np.ndarray) -> np.ndarray:
    """Computes continuous countermeasure score where higher value = more bona fide (genuine).

    Score definitions:
        - Logistic Regression / SVC: decision_function(X) (positive for bona fide = class 1)
        - Random Forest: log-odds log(p / (1 - p)) where p = P(bona fide)

    Convention:
        Higher score = more likely genuine (bona fide = class 1).
        Lower score = more likely synthetic / spoof (class 0).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_arr)
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_arr)
        p_bonafide = probs[:, BONAFIDE] if probs.shape[1] > 1 else probs[:, 0]
        p_clamped = np.clip(p_bonafide, 1e-6, 1.0 - 1e-6)
        scores = np.log(p_clamped / (1.0 - p_clamped))
    else:
        scores = model.predict(X_arr).astype(float)
    return np.asarray(scores, dtype=np.float64)


def fit_train_preprocessors(
    X_train: np.ndarray,
    pca_variance: float = 0.95,
    seed: int = 42,
) -> tuple[StandardScaler, PCA, np.ndarray]:
    """Fits StandardScaler and PCA strictly on TRAIN data only.

    Prevents any data leakage into validation or test splits.

    Returns:
        Tuple of (fitted_scaler, fitted_pca, X_train_transformed).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    max_components = min(X_train.shape[0], X_train.shape[1])
    n_comp: Any = pca_variance
    if isinstance(pca_variance, float) and pca_variance < 1.0:
        if max_components < 2:
            n_comp = 1
    elif isinstance(pca_variance, int) and pca_variance > max_components:
        n_comp = max_components

    pca = PCA(n_components=n_comp, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)

    logger.info(
        "Fitted PCA on TRAIN only: retained %d components out of %d (explained variance ratio sum: %.4f)",
        pca.n_components_,
        X_train.shape[1],
        float(np.sum(pca.explained_variance_ratio_)),
    )
    return scaler, pca, X_train_pca


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    pca_variance: float = 0.95,
    param_grid: Optional[dict[str, list[Any]]] = None,
    seed: int = 42,
) -> Pipeline:
    """Tunes and trains Logistic Regression with PCA and balanced weights.

    Scaler and PCA are fitted strictly on TRAIN only. Hyperparameters are selected
    via DEV ROC-AUC.
    """
    if param_grid is None:
        param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

    scaler, pca, X_tr_pca = fit_train_preprocessors(X_train, pca_variance=pca_variance, seed=seed)
    X_dv_pca = pca.transform(scaler.transform(X_dev))

    best_c = 1.0
    best_auc = -1.0
    c_candidates = param_grid.get("C", [0.01, 0.1, 1.0, 10.0])

    for c in c_candidates:
        clf = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(X_tr_pca, y_train)
        dev_scores = predict_classical_score(clf, X_dv_pca)
        auc = roc_auc_score(y_dev, dev_scores) if len(np.unique(y_dev)) > 1 else 0.5
        if auc > best_auc:
            best_auc = auc
            best_c = c

    logger.info("Best Logistic Regression C: %s (Dev AUC: %.4f)", best_c, best_auc)

    final_clf = LogisticRegression(C=best_c, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=seed)
    final_clf.fit(X_tr_pca, y_train)

    return Pipeline([
        ("scaler", scaler),
        ("pca", pca),
        ("classifier", final_clf),
    ])


def train_rbf_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    pca_variance: float = 0.95,
    max_train_samples: int = 10000,
    param_grid: Optional[dict[str, list[Any]]] = None,
    seed: int = 42,
) -> Pipeline:
    """Trains RBF-SVM with PCA and stratified subsampling for quadratic safety.

    Scaler and PCA are fitted strictly on TRAIN only.
    """
    if param_grid is None:
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", 0.01, 0.1],
        }

    if len(X_train) > max_train_samples:
        logger.warning("Subsampling SVM training set from %d to %d for quadratic complexity safety.", len(X_train), max_train_samples)
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

    scaler, pca, X_tr_pca = fit_train_preprocessors(X_tr, pca_variance=pca_variance, seed=seed)
    X_dv_pca = pca.transform(scaler.transform(X_dev))

    best_params = {"C": 1.0, "gamma": "scale"}
    best_auc = -1.0
    c_candidates = param_grid.get("C", [0.1, 1.0, 10.0])
    gamma_candidates = param_grid.get("gamma", ["scale", 0.01])

    for c in c_candidates:
        for g in gamma_candidates:
            clf = SVC(C=c, gamma=g, kernel="rbf", class_weight="balanced", random_state=seed)
            clf.fit(X_tr_pca, y_tr)
            dev_scores = predict_classical_score(clf, X_dv_pca)
            auc = roc_auc_score(y_dev, dev_scores) if len(np.unique(y_dev)) > 1 else 0.5
            if auc > best_auc:
                best_auc = auc
                best_params = {"C": c, "gamma": g}

    logger.info("Best RBF-SVM params: %s (Dev AUC: %.4f)", best_params, best_auc)

    final_clf = SVC(
        C=best_params["C"],
        gamma=best_params["gamma"],
        kernel="rbf",
        class_weight="balanced",
        random_state=seed,
    )
    final_clf.fit(X_tr_pca, y_tr)

    return Pipeline([
        ("scaler", scaler),
        ("pca", pca),
        ("classifier", final_clf),
    ])


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    pca_variance: float = 0.95,
    param_grid: Optional[dict[str, list[Any]]] = None,
    seed: int = 42,
) -> Pipeline:
    """Tunes and trains Random Forest with PCA and balanced subsampling.

    Scaler and PCA are fitted strictly on TRAIN only.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
        }

    scaler, pca, X_tr_pca = fit_train_preprocessors(X_train, pca_variance=pca_variance, seed=seed)
    X_dv_pca = pca.transform(scaler.transform(X_dev))

    best_params = {"n_estimators": 100, "max_depth": 10}
    best_auc = -1.0
    n_est_candidates = param_grid.get("n_estimators", [100, 200])
    depth_candidates = param_grid.get("max_depth", [10, None])

    for n_est in n_est_candidates:
        for depth in depth_candidates:
            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
            clf.fit(X_tr_pca, y_train)
            dev_scores = predict_classical_score(clf, X_dv_pca)
            auc = roc_auc_score(y_dev, dev_scores) if len(np.unique(y_dev)) > 1 else 0.5
            if auc > best_auc:
                best_auc = auc
                best_params = {"n_estimators": n_est, "max_depth": depth}

    logger.info("Best Random Forest params: %s (Dev AUC: %.4f)", best_params, best_auc)

    final_clf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    final_clf.fit(X_tr_pca, y_train)

    return Pipeline([
        ("scaler", scaler),
        ("pca", pca),
        ("classifier", final_clf),
    ])


def evaluate_classical_model(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    dev_threshold: Optional[float] = None,
    asv_score_file: Optional[str] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generates continuous scores for bona fide and computes full evaluation metrics."""
    scores = predict_classical_score(model, X)
    metrics_report = full_evaluation(
        y,
        scores,
        dev_threshold=dev_threshold,
        asv_score_file=asv_score_file,
        compute_ci=True,
    )
    return scores, metrics_report
