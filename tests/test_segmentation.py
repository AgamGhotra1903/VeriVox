"""Unit tests for temporal segmentation, repeat-padding, and multi-crop windowing.

"""

import pytest
import torch

from acforensics.constants import TARGET_SAMPLES
from acforensics.data.segmentation import (
    repeat_pad,
    fixed_length_crop_or_pad,
    generate_multi_crop_windows,
)


def test_repeat_pad_short_waveform():
    short_wave = torch.arange(1000, dtype=torch.float32)
    padded = repeat_pad(short_wave, target_samples=64000)
    assert padded.size(0) == 64000
    assert torch.equal(padded[:1000], short_wave)
    assert torch.equal(padded[1000:2000], short_wave)


def test_repeat_pad_already_sufficient():
    long_wave = torch.arange(80000, dtype=torch.float32)
    trimmed = repeat_pad(long_wave, target_samples=64000)
    assert trimmed.size(0) == 64000
    assert torch.equal(trimmed, long_wave[:64000])


def test_fixed_length_center_crop():
    wave = torch.arange(100000, dtype=torch.float32)
    cropped = fixed_length_crop_or_pad(wave, target_samples=64000, mode="center")
    assert cropped.size(0) == 64000
    expected_start = (100000 - 64000) // 2
    assert cropped[0].item() == expected_start


def test_fixed_length_random_crop():
    wave = torch.arange(100000, dtype=torch.float32)
    cropped = fixed_length_crop_or_pad(wave, target_samples=64000, mode="random")
    assert cropped.size(0) == 64000


def test_multi_crop_window_generation():
    wave = torch.ones(100000, dtype=torch.float32)
    windows = generate_multi_crop_windows(wave, window_samples=64000, hop_ratio=0.5)

    assert len(windows) >= 2
    for w in windows:
        assert w["waveform"].size(0) == 64000
        assert w["end_sample"] <= 100000
