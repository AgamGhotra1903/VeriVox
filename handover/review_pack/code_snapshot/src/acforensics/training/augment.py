"""Data augmentation for audio waveforms and spectrogram feature representations.

Author: Agam Ghotra (IIT2024226)
"""

import random
import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """SpecAugment (Park et al.): Frequency Masking and Time Masking.

    Applied directly on 2D spectrogram representations (F, T) during training
    to prevent overfitting to specific formant frequencies or temporal artifacts.
    """

    def __init__(
        self,
        freq_mask_max: int = 8,
        time_mask_max: int = 20,
        freq_masks_num: int = 2,
        time_masks_num: int = 2,
    ):
        super().__init__()
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.freq_masks_num = freq_masks_num
        self.time_masks_num = time_masks_num

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Applies SpecAugment in-place on feature tensor of shape (B, 1, F, T) or (B, F, T)."""
        if not self.training:
            return feature

        cloned = feature.clone()

        is_4d = cloned.dim() == 4
        if not is_4d:
            cloned = cloned.unsqueeze(1)

        B, C, F, T = cloned.shape

        for b in range(B):
            # Apply frequency masks
            for _ in range(self.freq_masks_num):
                f_width = random.randint(0, min(self.freq_mask_max, F))
                f_start = random.randint(0, F - f_width)
                cloned[b, :, f_start : f_start + f_width, :] = 0.0

            # Apply time masks
            for _ in range(self.time_masks_num):
                t_width = random.randint(0, min(self.time_mask_max, T))
                t_start = random.randint(0, T - t_width)
                cloned[b, :, :, t_start : t_start + t_width] = 0.0

        return cloned if is_4d else cloned.squeeze(1)
