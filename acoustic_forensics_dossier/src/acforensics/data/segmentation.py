"""Audio temporal segmentation, fixed-length cropping, repeat-padding, and multi-crop windowing.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

from typing import Any, Literal, Optional
import math
import numpy as np
import torch

from acforensics.constants import TARGET_SAMPLES, SAMPLE_RATE


def repeat_pad(waveform: torch.Tensor, target_samples: int = TARGET_SAMPLES) -> torch.Tensor:
    """Repeats a 1D audio waveform until it reaches or exceeds target length, then trims.

    Guarantees no zero-padding silence artifacts that could introduce artificial boundaries.

    Args:
        waveform: 1D torch.Tensor of audio samples.
        target_samples: Target length in samples (e.g. 64,000 for 4.0s).

    Returns:
        1D torch.Tensor of length exactly target_samples.
    """
    length = waveform.size(0)
    if length == 0:
        return torch.zeros(target_samples, dtype=waveform.dtype)

    if length >= target_samples:
        return waveform[:target_samples]

    repeat_count = math.ceil(target_samples / length)
    tiled = waveform.repeat(repeat_count)
    return tiled[:target_samples]


def fixed_length_crop_or_pad(
    waveform: torch.Tensor,
    target_samples: int = TARGET_SAMPLES,
    mode: Literal["random", "center", "start"] = "center",
) -> torch.Tensor:
    """Produces a fixed-length window via cropping or repeat-padding.

    Strategies:
    - 'random': Used during training to augment temporal diversity.
    - 'center': Deterministic center crop used for headline benchmark evaluation and dev scoring.
    - 'start': Takes the initial slice of the utterance.

    Args:
        waveform: 1D torch.Tensor of shape (samples,).
        target_samples: Required length (default: 64,000).
        mode: Cropping strategy ('random', 'center', 'start').

    Returns:
        1D torch.Tensor of shape (target_samples,).
    """
    length = waveform.size(0)

    if length == target_samples:
        return waveform
    elif length < target_samples:
        return repeat_pad(waveform, target_samples)
    else:
        # waveform is longer than target_samples -> crop
        excess = length - target_samples
        if mode == "center":
            start_idx = excess // 2
        elif mode == "start":
            start_idx = 0
        elif mode == "random":
            start_idx = int(torch.randint(0, excess + 1, (1,)).item())
        else:
            start_idx = excess // 2

        return waveform[start_idx : start_idx + target_samples]


def generate_multi_crop_windows(
    waveform: torch.Tensor,
    window_samples: int = TARGET_SAMPLES,
    hop_ratio: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
) -> list[dict[str, Any]]:
    """Segments arbitrary-length recording into overlapping fixed windows for multi-crop inference.

    Args:
        waveform: 1D torch.Tensor of shape (samples,).
        window_samples: Window length in samples (e.g. 64,000 for 4s).
        hop_ratio: Step ratio between consecutive windows (0.5 = 50% overlap).
        sample_rate: Sampling frequency in Hz.

    Returns:
        List of dictionaries with keys:
        - 'waveform': 1D torch.Tensor of length window_samples
        - 'window_idx': int
        - 'start_sample': int
        - 'end_sample': int
        - 'start_sec': float
        - 'end_sec': float
    """
    total_samples = waveform.size(0)

    # If shorter than window, repeat-pad to one window
    if total_samples <= window_samples:
        padded = repeat_pad(waveform, window_samples)
        return [{
            "waveform": padded,
            "window_idx": 0,
            "start_sample": 0,
            "end_sample": total_samples,
            "start_sec": 0.0,
            "end_sec": total_samples / sample_rate,
        }]

    hop_samples = max(1, int(window_samples * hop_ratio))
    windows = []
    window_idx = 0
    start = 0

    while start < total_samples:
        end = start + window_samples
        if end <= total_samples:
            chunk = waveform[start:end]
            actual_end = end
        else:
            # Final window: pad or take last window_samples
            chunk = waveform[total_samples - window_samples : total_samples]
            start = total_samples - window_samples
            actual_end = total_samples

        windows.append({
            "waveform": chunk,
            "window_idx": window_idx,
            "start_sample": start,
            "end_sample": actual_end,
            "start_sec": start / sample_rate,
            "end_sec": actual_end / sample_rate,
        })
        window_idx += 1

        if actual_end >= total_samples:
            break
        start += hop_samples

    return windows
