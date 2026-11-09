from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from acforensics.constants import SAMPLE_RATE, TARGET_SAMPLES, BONAFIDE, SPOOF
from acforensics.evaluation.robustness import (
    apply_additive_noise,
    apply_synthetic_reverb,
    apply_packet_loss,
    align_codec_delay,
)
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.evaluation.metrics import compute_eer


def test_additive_noise_preserves_shape_and_adds_energy():
    sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
    noisy = apply_additive_noise(sig, snr_db=10.0, seed=42)
    assert noisy.shape == sig.shape
    assert noisy.dtype == sig.dtype
    assert not np.allclose(sig, noisy)
    p_diff = np.mean((noisy - sig) ** 2)
    p_sig = np.mean(sig ** 2)
    snr_est = 10 * np.log10(p_sig / p_diff)
    assert abs(snr_est - 10.0) < 1.0


def test_synthetic_reverb_shape_and_decay():
    sig = np.zeros(16000, dtype=np.float32)
    sig[100] = 1.0
    rev = apply_synthetic_reverb(sig, rt60=0.3, sr=16000, seed=42)
    assert rev.shape == sig.shape
    assert rev.dtype == sig.dtype
    assert np.any(rev[200:2000] != 0.0)


def test_packet_loss_zeros_frames():
    sig = np.ones(16000, dtype=np.float32)
    dropped = apply_packet_loss(sig, loss_rate=0.20, packet_ms=20, sr=16000, seed=42)
    assert dropped.shape == sig.shape
    zero_fraction = np.mean(dropped == 0.0)
    assert 0.10 <= zero_fraction <= 0.35


def test_align_codec_delay():
    sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
    delay = 120
    delayed = np.concatenate([np.zeros(delay, dtype=np.float32), sig[:-delay]])
    aligned = align_codec_delay(sig, delayed, max_search_samples=500)
    assert len(aligned) > 0


def test_clean_subset_eer_guard():
    eval_parquet = Path("data/manifests/eval.parquet")
    if not eval_parquet.exists():
        pytest.skip("ASVspoof Eval manifest not found")

    ckpt_path = Path("checkpoints/crnn_lfcc_best.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/crnn_best.pt")
    if not ckpt_path.exists():
        pytest.skip("CRNN checkpoint not found")

    eval_df = pd.read_parquet(eval_parquet)
    sample_size = 500
    sample_dfs = []
    for (att, lbl), group in eval_df.groupby(["attack_id", "label"]):
        n_sample = max(1, int(round(len(group) / len(eval_df) * sample_size)))
        sample_dfs.append(group.sample(n=min(n_sample, len(group)), random_state=42))
    strat_df = pd.concat(sample_dfs).head(sample_size).reset_index(drop=True)

    device = torch.device("cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    crnn = CRNNForensics(in_channels=1, in_freq_dim=60)
    crnn.load_state_dict(ckpt["model_state_dict"])
    crnn.eval()

    extractor = LFCCExtractor(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_lfcc=20,
        with_deltas=True,
    )

    waves, labels = [], []
    for _, r in strat_df.iterrows():
        res = load_and_validate_audio(r["path"], target_sr=SAMPLE_RATE)
        if res.is_valid and res.waveform is not None:
            waves.append(res.waveform)
            labels.append(r["label"])

    feats = [per_utterance_cmvn(extractor(fixed_length_crop_or_pad(w, TARGET_SAMPLES, mode="center"))) for w in waves]
    scores = []
    with torch.no_grad():
        for i in range(0, len(feats), 64):
            batch = torch.stack(feats[i:i+64]).unsqueeze(1)
            logits, _ = crnn(batch)
            scores.extend((logits[:, BONAFIDE] - logits[:, SPOOF]).numpy())

    eer, _ = compute_eer(np.array(labels), np.array(scores))
    full_eval_eer = 8.39
    assert abs(eer - full_eval_eer) <= 4.0, f"Clean subset EER {eer:.2f}% deviates from full eval {full_eval_eer:.2f}%"

