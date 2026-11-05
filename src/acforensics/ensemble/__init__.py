"""Score-level fusion and stacked ensemble modules for Acoustic Forensics."""

from acforensics.ensemble.scores import build_score_matrix, load_score_matrix
from acforensics.ensemble.stacking import (
    StackedEnsemble,
    NonNegativeStackedEnsemble,
    assert_zero_data_leakage,
)
from acforensics.ensemble.predictor import EnsemblePredictor

__all__ = [
    "build_score_matrix",
    "load_score_matrix",
    "StackedEnsemble",
    "NonNegativeStackedEnsemble",
    "assert_zero_data_leakage",
    "EnsemblePredictor",
]
