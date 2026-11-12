"""Unit tests for CRNN model architecture, ablation toggles, tensor dimensions, and initialization.

"""

import pytest
import torch

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.models.crnn import CRNNForensics


def test_crnn_default_forward():
    x = torch.randn(4, 1, 60, 400)
    model = CRNNForensics(in_channels=1, in_freq_dim=60, features="lfcc")
    logits, attn_weights = model(x)

    assert logits.shape == (4, 2)
    assert attn_weights.shape == (4, 100)
    assert torch.allclose(attn_weights.sum(dim=1), torch.ones(4), atol=1e-4)


def test_crnn_logmel_variant_forward():
    x = torch.randn(4, 1, 80, 400)
    model = CRNNForensics(in_channels=1, in_freq_dim=80, features="logmel")
    logits, attn_weights = model(x)

    assert logits.shape == (4, 2)
    assert attn_weights.shape == (4, 100)
    assert torch.allclose(attn_weights.sum(dim=1), torch.ones(4), atol=1e-4)


def test_crnn_predict_score():
    x = torch.randn(3, 1, 60, 400)
    model = CRNNForensics(in_channels=1, in_freq_dim=60, features="lfcc")
    model.eval()
    with torch.no_grad():
        scores = model.predict_score(x)
        assert scores.shape == (3,)
        logits, _ = model(x)
        expected_scores = logits[:, BONAFIDE] - logits[:, SPOOF]
        assert torch.allclose(scores, expected_scores)


def test_crnn_shape_assertions():
    model = CRNNForensics(in_channels=1, in_freq_dim=60, features="lfcc")
    with pytest.raises(AssertionError):
        invalid_5d = torch.randn(2, 1, 1, 60, 400)
        _ = model(invalid_5d)


def test_crnn_cnn_only_ablation():
    x = torch.randn(2, 1, 60, 400)
    model = CRNNForensics(in_channels=1, in_freq_dim=60, use_gru=False, use_attention=False)
    logits, attn = model(x)
    assert logits.shape == (2, 2)


def test_crnn_mean_pooling_ablation():
    x = torch.randn(2, 1, 60, 400)
    model = CRNNForensics(in_channels=1, in_freq_dim=60, pooling="mean")
    logits, attn = model(x)
    assert logits.shape == (2, 2)


def test_crnn_combined_two_branch():
    x_lfcc = torch.randn(2, 1, 60, 400)
    x_logmel = torch.randn(2, 1, 80, 400)
    model = CRNNForensics(features="combined")
    logits, attn = model(x_lfcc, x_aux=x_logmel)
    assert logits.shape == (2, 2)


def test_crnn_custom_dropout_and_init():
    model = CRNNForensics(
        in_channels=1,
        in_freq_dim=60,
        conv_dropout=0.2,
        gru_dropout=0.4,
        classifier_dropout=0.5,
        init_weights=True,
    )
    assert model.conv_dropout == 0.2
    assert model.gru_dropout == 0.4
    assert model.classifier_dropout == 0.5
