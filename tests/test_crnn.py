"""Unit tests for CRNN model architecture, ablation toggles, and tensor dimensions.

Author: Agam Ghotra (IIT2024226)
"""

import pytest
import torch

from acforensics.models.crnn import CRNNForensics


def test_crnn_default_forward():
    # Batch size 4, 1 channel, 60 LFCC coeffs, 400 time frames
    x = torch.randn(4, 1, 60, 400)
    model = CRNNForensics(in_channels=1, in_freq_dim=60, features="lfcc")
    logits, attn_weights = model(x)

    assert logits.shape == (4, 2)
    assert attn_weights.shape[0] == 4
    # Attention weights should sum to 1.0 along the time dimension
    assert torch.allclose(attn_weights.sum(dim=1), torch.ones(4), atol=1e-4)


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
