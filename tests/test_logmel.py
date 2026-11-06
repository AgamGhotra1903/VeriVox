from pathlib import Path
import numpy as np
import torch
import pytest

from acforensics.constants import BONAFIDE, SPOOF, TARGET_SAMPLES
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics

def test_logmel_cmvn_normalization_properties():
    extractor = LogMelExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80)
    waveform = torch.randn(TARGET_SAMPLES)
    raw_feats = extractor(waveform)
    assert raw_feats.shape[0] == 80
    assert raw_feats.shape[1] > 0
    norm_feats = per_utterance_cmvn(raw_feats)
    assert norm_feats.shape == raw_feats.shape
    assert not torch.isnan(norm_feats).any()
    assert not torch.isinf(norm_feats).any()
    channel_means = norm_feats.mean(dim=1)
    assert torch.all(torch.abs(channel_means) < 1e-4)

def test_crnn_logmel_architecture_forward():
    model = CRNNForensics(
        in_channels=1,
        in_freq_dim=80,
        features="logmel",
        use_gru=True,
        use_attention=True,
        pooling="attention",
        proj_dim=128,
        gru_hidden=128,
        num_gru_layers=2,
        num_classes=2,
    )
    model.eval()
    batch = torch.randn(4, 1, 80, 400)
    with torch.no_grad():
        logits, attn = model(batch)
    assert logits.shape == (4, 2)
    assert not torch.isnan(logits).any()
    assert attn is not None

