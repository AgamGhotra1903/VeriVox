"""Feature Normalization: Train-Only Global Mean/Std and Per-Utterance CMVN.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

from pathlib import Path
from typing import Optional
import json
import torch
import torch.nn as nn

from acforensics.utils.io import save_json, load_json
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def per_utterance_cmvn(feature: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Applies Cepstral Mean and Variance Normalization (CMVN) along the temporal axis.

    Normalizes each frequency/cepstral channel independently to zero mean and unit variance.

    Args:
        feature: Tensor of shape (F, T) or (B, F, T).
        eps: Small constant to avoid zero-division in static regions.

    Returns:
        Normalized tensor with same shape.
    """
    if feature.dim() == 2:
        # Shape: (F, T) -> compute stats across T (dim=-1)
        mean = feature.mean(dim=-1, keepdim=True)
        std = feature.std(dim=-1, keepdim=True)
        return (feature - mean) / (std + eps)
    elif feature.dim() == 3:
        # Shape: (B, F, T) -> compute stats across T (dim=-1)
        mean = feature.mean(dim=-1, keepdim=True)
        std = feature.std(dim=-1, keepdim=True)
        return (feature - mean) / (std + eps)
    else:
        raise ValueError(f"Expected 2D or 3D feature tensor, got {feature.shape}")


class FeatureNormalizer(nn.Module):
    """Global Feature Normalizer maintaining train-set mean and standard deviation.

    Guarantees strict absence of data leakage:
    Statistics are computed exclusively on the official training split and
    persisted to `checkpoints/normalization_stats.json`. The identical statistics
    are loaded and applied during dev/eval and real-time production inference.
    """

    def __init__(self, feature_dim: Optional[int] = None, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.feature_dim = feature_dim

        # Initial dummy buffers; updated when fitted or loaded
        if feature_dim is not None:
            self.register_buffer("mean", torch.zeros(feature_dim, 1))
            self.register_buffer("std", torch.ones(feature_dim, 1))
        else:
            self.register_buffer("mean", None)
            self.register_buffer("std", None)
        self.is_fitted = False

    def fit_from_tensors(self, feature_list: list[torch.Tensor]) -> None:
        """Fits mean and std across a list or generator of training feature matrices (F, T)."""
        all_frames = []
        for feat in feature_list:
            if feat.dim() == 2:
                all_frames.append(feat)  # (F, T)
            elif feat.dim() == 3:
                for b in range(feat.size(0)):
                    all_frames.append(feat[b])

        # Concatenate along time: (F, Total_T)
        concatenated = torch.cat(all_frames, dim=1)
        mean = concatenated.mean(dim=1, keepdim=True)  # (F, 1)
        std = concatenated.std(dim=1, keepdim=True)    # (F, 1)
        # Clamp std to avoid division by near-zero
        std = torch.clamp(std, min=1e-5)

        self.mean = mean
        self.std = std
        self.feature_dim = mean.size(0)
        self.is_fitted = True

        logger.info(
            "Fitted global normalization statistics on %d features across %d total frames.",
            self.feature_dim,
            concatenated.size(1),
        )

    def save(self, path: Path | str) -> None:
        """Saves fitted mean and std to a JSON file."""
        if not self.is_fitted or self.mean is None:
            raise RuntimeError("Cannot save unfitted FeatureNormalizer.")
        data = {
            "feature_dim": int(self.feature_dim),
            "mean": self.mean.squeeze().tolist(),
            "std": self.std.squeeze().tolist(),
        }
        save_json(data, path)
        logger.info("Saved normalization statistics to %s", path)

    def load(self, path: Path | str) -> "FeatureNormalizer":
        """Loads mean and std from saved JSON file."""
        data = load_json(path)
        mean = torch.tensor(data["mean"], dtype=torch.float32).unsqueeze(-1)
        std = torch.tensor(data["std"], dtype=torch.float32).unsqueeze(-1)
        self.mean = mean
        self.std = std
        self.feature_dim = data["feature_dim"]
        self.is_fitted = True
        logger.info("Loaded normalization statistics from %s (dim: %d)", path, self.feature_dim)
        return self

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Applies global normalization: (feature - mean) / (std + eps).

        Args:
            feature: Tensor of shape (F, T) or (B, F, T).

        Returns:
            Normalized tensor of identical shape.
        """
        if not self.is_fitted or self.mean is None:
            # Fallback to per-utterance CMVN if not fitted
            return per_utterance_cmvn(feature, eps=self.eps)

        mean = self.mean.to(feature.device)
        std = self.std.to(feature.device)

        if feature.dim() == 2:
            return (feature - mean) / (std + self.eps)
        elif feature.dim() == 3:
            # feature shape (B, F, T), mean shape (F, 1)
            return (feature - mean.unsqueeze(0)) / (std.unsqueeze(0) + self.eps)
        else:
            return feature
