"""Unit tests for GMM baseline model and log-likelihood ratio scoring.

Author: Agam Ghotra (IIT2024226)
"""

import pytest
import numpy as np

from acforensics.models.gmm import GMMForensics


def test_gmm_fit_and_score():
    np.random.seed(42)
    # 200 frames of 60-dim LFCC
    bf_frames = np.random.normal(loc=1.0, scale=0.5, size=(200, 60))
    sp_frames = np.random.normal(loc=-1.0, scale=0.5, size=(200, 60))

    gmm = GMMForensics(n_components=4, covariance_type="diag", max_iter=20, random_state=42)
    gmm.fit(bf_frames, sp_frames)

    assert gmm.is_fitted is True

    # Test scoring: genuine frames should have positive LLR
    test_bf = np.random.normal(loc=1.0, scale=0.5, size=(50, 60))
    test_sp = np.random.normal(loc=-1.0, scale=0.5, size=(50, 60))

    score_bf = gmm.score_utterance(test_bf)
    score_sp = gmm.score_utterance(test_sp)

    # Score convention: Higher score = more bona fide (genuine)
    assert score_bf > score_sp
    assert score_bf > 0.0
    assert score_sp < 0.0


def test_gmm_batch_predict():
    np.random.seed(42)
    bf_frames = np.random.normal(loc=1.0, scale=0.5, size=(100, 60))
    sp_frames = np.random.normal(loc=-1.0, scale=0.5, size=(100, 60))

    gmm = GMMForensics(n_components=2, max_iter=10)
    gmm.fit(bf_frames, sp_frames)

    test_list = [np.random.randn(30, 60), np.random.randn(40, 60)]
    scores = gmm.predict_score(test_list)

    assert scores.shape == (2,)
    assert isinstance(scores[0], float)

