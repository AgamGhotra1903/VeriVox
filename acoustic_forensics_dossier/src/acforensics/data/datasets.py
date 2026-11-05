"""PyTorch Dataset implementations, collate functions, and balanced samplers.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    TARGET_SAMPLES,
    SAMPLE_RATE,
)
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


class ASVspoofDataset(Dataset):
    """PyTorch Dataset for official ASVspoof 2019 Logical Access (LA) splits."""

    def __init__(
        self,
        protocol_df: pd.DataFrame,
        target_samples: int = TARGET_SAMPLES,
        split: str = "train",
        crop_mode: Optional[str] = None,
        feature_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        audio_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_samples: Optional[int] = None,
    ):
        """Initializes ASVspoofDataset.

        Args:
            protocol_df: DataFrame parsed by load_asvspoof_protocol.
            target_samples: Target length in audio samples (default: 64,000 = 4s).
            split: 'train', 'dev', or 'eval'.
            crop_mode: 'random' for training, 'center' for dev/eval. If None, chosen based on split.
            feature_transform: Optional callable transforming waveform -> 2D feature matrix.
            audio_transform: Optional callable applying waveform augmentation.
            max_samples: Subsample dataset for quick debugging or smoke tests.
        """
        self.split = split
        self.target_samples = target_samples
        self.feature_transform = feature_transform
        self.audio_transform = audio_transform

        if crop_mode is not None:
            self.crop_mode = crop_mode
        else:
            self.crop_mode = "random" if split == "train" else "center"

        # Optional debug subsampling (stratified when possible)
        if max_samples is not None and max_samples < len(protocol_df):
            # Stratified subsampling
            bonafide_df = protocol_df[protocol_df["label"] == BONAFIDE]
            spoof_df = protocol_df[protocol_df["label"] == SPOOF]

            ratio = len(bonafide_df) / len(protocol_df)
            n_bonafide = max(1, int(max_samples * ratio))
            n_spoof = max(1, max_samples - n_bonafide)

            sub_bf = bonafide_df.head(n_bonafide)
            sub_sp = spoof_df.head(n_spoof)
            self.df = pd.concat([sub_bf, sub_sp]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            logger.info("Subsampled %s split to %d samples for debug/smoke mode.", split, len(self.df))
        else:
            self.df = protocol_df.reset_index(drop=True)

        self.labels = self.df["label"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label = int(row["label"])
        file_id = str(row["file_id"])
        speaker_id = str(row["speaker_id"])
        attack_id = str(row.get("attack_id", "-"))

        # Load and validate audio (16 kHz mono)
        res = load_and_validate_audio(audio_path, target_sr=SAMPLE_RATE, log_rejections=True)
        if not res.is_valid or res.waveform is None:
            # Fallback for corrupted audio to prevent pipeline crashing
            logger.warning("Corrupt or invalid audio encountered at %s: %s", audio_path, res.error_reason)
            waveform = torch.zeros(self.target_samples, dtype=torch.float32)
        else:
            waveform = res.waveform

        # Segment to fixed 4.0s (crop or repeat-pad)
        waveform = fixed_length_crop_or_pad(
            waveform,
            target_samples=self.target_samples,
            mode=self.crop_mode,
        )
        assert waveform.shape == (self.target_samples,), (
            f"Expected waveform of shape ({self.target_samples},), got {tuple(waveform.shape)}"
        )

        # Optional waveform augmentation
        if self.audio_transform is not None:
            waveform = self.audio_transform(waveform)
            assert waveform.shape == (self.target_samples,), "Augmentation altered waveform length"

        item = {
            "waveform": waveform,
            "label": torch.tensor(label, dtype=torch.long),
            "file_id": file_id,
            "speaker_id": speaker_id,
            "attack_id": attack_id,
        }

        # Optional feature extraction (returns 2D or 3D tensor: (F, T) or (1, F, T))
        if self.feature_transform is not None:
            feature = self.feature_transform(waveform)
            assert feature.ndim in (2, 3), (
                f"Feature extractor must return 2D (F, T) or 3D (1, F, T) tensor, got {tuple(feature.shape)}"
            )
            item["feature"] = feature

        return item


class InTheWildDataset(Dataset):
    """PyTorch Dataset for In-The-Wild cross-domain evaluation."""

    def __init__(
        self,
        protocol_df: pd.DataFrame,
        target_samples: int = TARGET_SAMPLES,
        feature_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_samples: Optional[int] = None,
    ):
        self.target_samples = target_samples
        self.feature_transform = feature_transform

        if max_samples is not None and max_samples < len(protocol_df):
            self.df = protocol_df.head(max_samples).reset_index(drop=True)
        else:
            self.df = protocol_df.reset_index(drop=True)

        self.labels = self.df["label"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label = int(row["label"])
        file_id = str(row["file_id"])
        speaker_id = str(row.get("speaker_id", "unknown"))

        res = load_and_validate_audio(audio_path, target_sr=SAMPLE_RATE, log_rejections=True)
        if not res.is_valid or res.waveform is None:
            waveform = torch.zeros(self.target_samples, dtype=torch.float32)
        else:
            waveform = res.waveform

        waveform = fixed_length_crop_or_pad(waveform, target_samples=self.target_samples, mode="center")
        assert waveform.shape == (self.target_samples,), (
            f"Expected waveform of shape ({self.target_samples},), got {tuple(waveform.shape)}"
        )

        item = {
            "waveform": waveform,
            "label": torch.tensor(label, dtype=torch.long),
            "file_id": file_id,
            "speaker_id": speaker_id,
            "attack_id": "itw",
        }

        if self.feature_transform is not None:
            feature = self.feature_transform(waveform)
            assert feature.ndim in (2, 3), (
                f"Feature extractor must return 2D or 3D tensor, got {tuple(feature.shape)}"
            )
            item["feature"] = feature

        return item


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates a list of dataset items into batch tensors.

    Batches waveforms (B, T_samples) and labels (B,); preserves metadata lists for evaluation tracking.
    Features, if present, are batched to (B, 1, F, T).
    """
    assert len(batch) > 0, "Cannot collate empty batch"
    waveforms = torch.stack([b["waveform"] for b in batch], dim=0)  # Shape: (B, TARGET_SAMPLES)
    labels = torch.stack([b["label"] for b in batch], dim=0)        # Shape: (B,)

    assert waveforms.ndim == 2, f"Expected 2D waveforms tensor (B, T), got {waveforms.shape}"
    assert labels.ndim == 1, f"Expected 1D labels tensor (B,), got {labels.shape}"

    collated = {
        "waveform": waveforms,
        "label": labels,
        "file_id": [b["file_id"] for b in batch],
        "speaker_id": [b["speaker_id"] for b in batch],
        "attack_id": [b["attack_id"] for b in batch],
    }

    if "feature" in batch[0]:
        features = torch.stack([b["feature"] for b in batch], dim=0)
        # Ensure 4D batched feature tensor: (B, 1, F, T)
        if features.ndim == 3:
            features = features.unsqueeze(1)
        assert features.ndim == 4, f"Expected 4D batched feature tensor (B, 1, F, T), got {features.shape}"
        collated["feature"] = features

    return collated


def get_balanced_sampler(dataset: ASVspoofDataset) -> WeightedRandomSampler:
    """Creates a WeightedRandomSampler that samples bona fide and spoof uniformly.

    Used as an ablation against class-weighted cross-entropy loss.
    """
    labels = dataset.labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)

    sample_weights = class_weights[labels]
    sample_weights_tensor = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(dataset),
        replacement=True,
    )
    return sampler
