"""Unit tests for Stacked Ensemble and score-level fusion.

"""

import pytest
import numpy as np

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.ensemble.stacking import StackedEnsemble, assert_zero_data_leakage


def test_leakage_assertion_detects_overlap():
    train_ids = ["f1", "f2", "f3"]
    dev_ids = ["f4", "f5"]
    eval_ids = ["f6", "f7"]

    assert_zero_data_leakage(train_ids, dev_ids, eval_ids)

    with pytest.raises(AssertionError):
        assert_zero_data_leakage(["f1", "f_LEAK"], ["f_LEAK", "f2"], ["f3"])

    with pytest.raises(AssertionError):
        assert_zero_data_leakage(["f1"], ["f2", "f_LEAK"], ["f_LEAK", "f3"])


def test_stacked_ensemble_fit_and_predict():
    np.random.seed(42)
    n_dev = 50
    n_models = 4
    model_names = ["CRNN-LFCC", "CRNN-LogMel", "GMM", "SVM"]

    dev_scores_bf = np.random.normal(loc=2.0, scale=0.8, size=(25, n_models))
    dev_scores_sp = np.random.normal(loc=-2.0, scale=0.8, size=(25, n_models))
    dev_scores = np.vstack([dev_scores_bf, dev_scores_sp])
    dev_labels = np.array([BONAFIDE] * 25 + [SPOOF] * 25)

    stacker = StackedEnsemble(model_names=model_names, seed=42)
    stacker.fit(dev_scores, dev_labels)

    assert stacker.is_fitted is True

    test_bf = np.random.normal(loc=2.0, scale=0.8, size=(5, n_models))
    test_sp = np.random.normal(loc=-2.0, scale=0.8, size=(5, n_models))

    scores_bf = stacker.predict_score(test_bf)
    scores_sp = stacker.predict_score(test_sp)
    assert np.mean(scores_bf) > np.mean(scores_sp)

    probs_bf = stacker.predict_proba(test_bf)
    probs_sp = stacker.predict_proba(test_sp)
    assert np.all((probs_bf >= 0.0) & (probs_bf <= 1.0))
    assert np.mean(probs_bf) > np.mean(probs_sp)

    mean_bf = stacker.predict_baseline_mean(test_bf)
    mean_sp = stacker.predict_baseline_mean(test_sp)
    assert np.mean(mean_bf) > np.mean(mean_sp)

    weights_df = stacker.get_coefficients_table()
    assert len(weights_df) == n_models
    assert "Base Model" in weights_df.columns
    assert "Meta-Learner Weight" in weights_df.columns

