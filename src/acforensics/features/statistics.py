"""Utterance-level statistical pooling for classical machine learning models.

Author: Jashan Goyal (IIT2024223)
"""

import numpy as np
from scipy import stats
import torch


def extract_utterance_statistics(
    feature_matrix: torch.Tensor | np.ndarray,
    stat_types: tuple[str, ...] = ("mean", "std", "min", "max", "skewness", "kurtosis"),
) -> np.ndarray:
    """Computes utterance-level summary statistics across temporal frames for classical models.

    For an input feature matrix of shape (F, T):
    - F is the number of frequency/cepstral channels (e.g. 60 for LFCC, 80 for Log-Mel).
    - T is the number of time frames (~400).
    Computes 6 summary statistics per channel:
    1. Mean: Central tendency of spectral energy.
    2. Standard Deviation: Temporal variability/modulation depth.
    3. Minimum: Floor spectral envelope.
    4. Maximum: Peak resonant envelope.
    5. Skewness: Asymmetry in energy distribution over time.
    6. Kurtosis: Heaviness of tails/peakedness (impulsiveness of vocoder glitches).

    Final Dimensions:
    - LFCC:    60 channels * 6 statistics = 360-dimensional vector
    - Log-Mel: 80 channels * 6 statistics = 480-dimensional vector
    - Combined: 360 + 480 = 840-dimensional vector

    Args:
        feature_matrix: 2D array or tensor of shape (F, T).
        stat_types: Tuple of statistical descriptors to compute.

    Returns:
        1D NumPy array of length F * len(stat_types).
    """
    if isinstance(feature_matrix, torch.Tensor):
        feat = feature_matrix.detach().cpu().numpy()
    else:
        feat = np.asarray(feature_matrix)

    if feat.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix (F, T), got shape {feat.shape}")

    # feat shape: (F, T)
    stats_list = []

    # Mean: shape (F,)
    mean_val = np.mean(feat, axis=1)
    # Std: shape (F,)
    std_val = np.std(feat, axis=1)
    # Min: shape (F,)
    min_val = np.min(feat, axis=1)
    # Max: shape (F,)
    max_val = np.max(feat, axis=1)
    # Skewness: shape (F,)
    skew_val = stats.skew(feat, axis=1, bias=False)
    # Handle NaN in case of constant values
    skew_val = np.nan_to_num(skew_val, nan=0.0)
    # Kurtosis: shape (F,)
    kurt_val = stats.kurtosis(feat, axis=1, bias=False)
    kurt_val = np.nan_to_num(kurt_val, nan=0.0)

    stat_map = {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
    }

    if isinstance(stat_types, str):
        stat_types = (stat_types,)

    selected = [stat_map[s] for s in stat_types if s in stat_map]
    if not selected:
        raise ValueError(
            f"No valid statistics found in stat_types={stat_types}. "
            f"Allowed types: {list(stat_map.keys())}"
        )
    pooled_vector = np.concatenate(selected, axis=0)

    return pooled_vector.astype(np.float32)
