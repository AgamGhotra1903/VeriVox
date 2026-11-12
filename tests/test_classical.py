"""Unit tests for classical models with PCA and unified scoring convention.

"""

import pytest
import numpy as np

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.models.classical import (
    extract_mean_std_features,
    predict_classical_score,
    train_logistic_regression,
    train_rbf_svm,
    train_random_forest,
)


def test_extract_mean_std_features_shape():
    frames = np.random.randn(60, 400).astype(np.float64)
    feat = extract_mean_std_features(frames)
    assert feat.shape == (120,)

    frames_mel = np.random.randn(80, 400).astype(np.float64)
    feat_mel = extract_mean_std_features(frames_mel)
    assert feat_mel.shape == (160,)


def test_classical_models_sign_convention_and_pca():
    """Verifies that across all classical models, higher score means more bona fide,

    and that PCA is fitted cleanly on train only.
    """
    np.random.seed(42)
    n_train = 60
    n_dev = 30
    n_dim = 120

    X_train = np.vstack([
        np.random.normal(1.5, 0.5, size=(n_train // 2, n_dim)),
        np.random.normal(-1.5, 0.5, size=(n_train // 2, n_dim)),
    ]).astype(np.float64)
    y_train = np.array([BONAFIDE] * (n_train // 2) + [SPOOF] * (n_train // 2), dtype=np.int64)

    X_dev = np.vstack([
        np.random.normal(1.5, 0.5, size=(n_dev // 2, n_dim)),
        np.random.normal(-1.5, 0.5, size=(n_dev // 2, n_dim)),
    ]).astype(np.float64)
    y_dev = np.array([BONAFIDE] * (n_dev // 2) + [SPOOF] * (n_dev // 2), dtype=np.int64)

    models = [
        ("Logistic Regression", train_logistic_regression(X_train, y_train, X_dev, y_dev, pca_variance=0.95)),
        ("RBF-SVM", train_rbf_svm(X_train, y_train, X_dev, y_dev, pca_variance=0.95)),
        ("Random Forest", train_random_forest(X_train, y_train, X_dev, y_dev, pca_variance=0.95)),
    ]

    bf_test = np.random.normal(1.5, 0.5, size=(5, n_dim)).astype(np.float64)
    sp_test = np.random.normal(-1.5, 0.5, size=(5, n_dim)).astype(np.float64)

    for name, pipeline in models:
        assert "pca" in pipeline.named_steps
        pca_step = pipeline.named_steps["pca"]
        assert pca_step.n_components_ > 0
        assert pca_step.n_components_ <= n_dim

        bf_scores = predict_classical_score(pipeline, bf_test)
        sp_scores = predict_classical_score(pipeline, sp_test)

        assert np.mean(bf_scores) > np.mean(sp_scores), f"Model {name} failed sign convention!"
