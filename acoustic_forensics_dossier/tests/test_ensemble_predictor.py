"""Tests for the unified EnsemblePredictor."""

import pytest
import numpy as np
import torch
from pathlib import Path

from acforensics.ensemble.predictor import EnsemblePredictor


def test_ensemble_predictor_waveform_inference():
    predictor = EnsemblePredictor()
    dummy_wave = torch.randn(32000)

    result = predictor.predict_waveform(dummy_wave)

    assert "base_scores" in result
    assert "fused_score" in result
    assert "baseline_mean_score" in result
    assert "probability_bonafide" in result
    assert "verdict" in result

    base_scores = result["base_scores"]
    assert len(base_scores) == 6
    for name in ["CRNN-LFCC", "CRNN-LogMel", "GMM", "SVM", "Random Forest", "Logistic Regression"]:
        assert name in base_scores
        assert isinstance(base_scores[name], float)

    assert 0.0 <= result["probability_bonafide"] <= 1.0
    assert result["verdict"] in ["BONA FIDE (GENUINE)", "SPOOF (SYNTHETIC)"]

