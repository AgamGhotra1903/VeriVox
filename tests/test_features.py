"""Unit tests for LFCC, Log-Mel, Normalization, and Utterance Statistics.

Author: Paras Bansal (IIT2024221), Jashan Goyal (IIT2024223)
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from acforensics.features.lfcc import LFCCExtractor, create_linear_filterbank, create_dct_matrix
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.normalization import FeatureNormalizer, per_utterance_cmvn
from acforensics.features.statistics import extract_utterance_statistics


def test_linear_filterbank_shape():
    fb = create_linear_filterbank(n_filters=20, n_fft=512, sample_rate=16000)
    assert fb.shape == (20, 257)
    # Check that coefficients are non-negative and peak at 1.0
    assert torch.all(fb >= 0.0)
    assert torch.all(fb <= 1.0)


def test_dct_matrix_orthonormality():
    dct_mat = create_dct_matrix(n_coeffs=20, n_filters=20)
    assert dct_mat.shape == (20, 20)
    # Check orthonormality: M * M^T approx Identity
    identity = torch.eye(20)
    prod = torch.mm(dct_mat, dct_mat.t())
    assert torch.allclose(prod, identity, atol=1e-5)


def test_lfcc_extractor_shape():
    extractor = LFCCExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    # 4.0s signal = 64,000 samples
    waveform = torch.randn(64000)
    feat = extractor(waveform)

    # Expected: 60 coefficients (20 static + 20 delta + 20 delta-delta)
    # Expected frames for 64,000 samples with hop 160 is (64000 // 160) + 1 = 401 frames
    assert feat.dim() == 2
    assert feat.shape[0] == 60
    assert 395 <= feat.shape[1] <= 405


def test_lfcc_extractor_batched():
    extractor = LFCCExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    batch_waveform = torch.randn(4, 64000)
    feat = extractor(batch_waveform)
    assert feat.dim() == 3
    assert feat.shape[0] == 4
    assert feat.shape[1] == 60


def test_logmel_extractor_shape():
    extractor = LogMelExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80)
    waveform = torch.randn(64000)
    feat = extractor(waveform)
    assert feat.dim() == 2
    assert feat.shape[0] == 80
    assert 395 <= feat.shape[1] <= 405


def test_per_utterance_cmvn():
    feat = torch.randn(60, 400) * 5.0 + 3.0
    norm_feat = per_utterance_cmvn(feat)
    # Mean along time axis should be approx 0, std approx 1
    mean_t = norm_feat.mean(dim=-1)
    std_t = norm_feat.std(dim=-1)
    assert torch.allclose(mean_t, torch.zeros_like(mean_t), atol=1e-4)
    assert torch.allclose(std_t, torch.ones_like(std_t), atol=1e-3)


def test_global_feature_normalizer(tmp_path: Path):
    normalizer = FeatureNormalizer()
    feats = [torch.randn(60, 400) * 2.0 + 1.0 for _ in range(5)]
    normalizer.fit_from_tensors(feats)
    assert normalizer.is_fitted
    assert normalizer.mean.shape == (60, 1)

    # Save and reload
    save_file = tmp_path / "norm_stats.json"
    normalizer.save(save_file)

    reloaded = FeatureNormalizer().load(save_file)
    assert reloaded.is_fitted
    assert torch.allclose(reloaded.mean, normalizer.mean)

    # Test normalization
    test_feat = torch.randn(60, 400)
    norm = reloaded(test_feat)
    assert norm.shape == (60, 400)


def test_extract_utterance_statistics_lfcc():
    # LFCC: 60 channels x 400 frames
    feat = np.random.randn(60, 400).astype(np.float32)
    stats_vec = extract_utterance_statistics(feat)
    # 60 channels * 6 statistics = 360 dimensions
    assert stats_vec.shape == (360,)
    assert not np.isnan(stats_vec).any()


def test_extract_utterance_statistics_logmel():
    # LogMel: 80 channels x 400 frames
    feat = np.random.randn(80, 400).astype(np.float32)
    stats_vec = extract_utterance_statistics(feat)
    # 80 channels * 6 statistics = 480 dimensions
    assert stats_vec.shape == (480,)
    assert not np.isnan(stats_vec).any()
