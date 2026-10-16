"""Unit tests for audio validation, resampling, and normalization.

Author: Paras Bansal (IIT2024221)
"""

import pytest
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

from acforensics.constants import SAMPLE_RATE
from acforensics.data.audio_io import (
    load_and_validate_audio,
    normalize_audio,
    resample_audio,
)


def create_synthetic_wav(path: Path, duration_sec: float, sr: int, channels: int = 1, inject_nan: bool = False):
    samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, samples, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    if channels == 2:
        sig = np.column_stack([sig, sig * 0.8])

    if inject_nan:
        sig[10] = np.nan
        sf.write(str(path), sig, sr, subtype="FLOAT")
    else:
        sf.write(str(path), sig, sr)


def test_valid_audio_loading(tmp_path: Path):
    audio_file = tmp_path / "valid.wav"
    create_synthetic_wav(audio_file, duration_sec=1.0, sr=16000, channels=1)

    res = load_and_validate_audio(audio_file, target_sr=16000)
    assert res.is_valid
    assert res.waveform is not None
    assert res.sample_rate == 16000
    assert pytest.approx(res.duration_sec, rel=1e-2) == 1.0
    assert res.waveform.ndim == 1


def test_stereo_to_mono_conversion(tmp_path: Path):
    audio_file = tmp_path / "stereo.wav"
    create_synthetic_wav(audio_file, duration_sec=1.0, sr=16000, channels=2)

    res = load_and_validate_audio(audio_file, target_sr=16000)
    assert res.is_valid
    assert res.waveform.ndim == 1


def test_resampling_from_44k(tmp_path: Path):
    audio_file = tmp_path / "audio_44k.wav"
    create_synthetic_wav(audio_file, duration_sec=1.0, sr=44100, channels=1)

    res = load_and_validate_audio(audio_file, target_sr=16000)
    assert res.is_valid
    assert res.sample_rate == 16000
    assert abs(len(res.waveform) - 16000) < 50


def test_reject_too_short_audio(tmp_path: Path):
    audio_file = tmp_path / "short.wav"
    create_synthetic_wav(audio_file, duration_sec=0.2, sr=16000)  # Under 0.5s

    res = load_and_validate_audio(audio_file, min_duration_sec=0.5)
    assert not res.is_valid
    assert "shorter than minimum" in res.error_reason.lower()


def test_reject_nan_audio(tmp_path: Path):
    audio_file = tmp_path / "nan.wav"
    create_synthetic_wav(audio_file, duration_sec=1.0, sr=16000, inject_nan=True)

    res = load_and_validate_audio(audio_file)
    assert not res.is_valid
    assert "nan" in res.error_reason.lower()


def test_peak_normalization():
    sig = torch.tensor([0.2, -0.8, 0.4, -0.1], dtype=torch.float32)
    norm = normalize_audio(sig, method="peak")
    assert torch.max(torch.abs(norm)).item() == pytest.approx(1.0)
