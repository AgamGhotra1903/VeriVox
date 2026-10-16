"""Model definitions and classical baseline implementations."""

from acforensics.models.classical import (
    train_logistic_regression,
    train_rbf_svm,
    train_random_forest,
    evaluate_classical_model,
)

__all__ = [
    "train_logistic_regression",
    "train_rbf_svm",
    "train_random_forest",
    "evaluate_classical_model",
]
