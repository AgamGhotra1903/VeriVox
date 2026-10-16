"""Unit tests for PyTorch Dataset implementations, collate_fn, and balanced sampler.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

import pytest
import soundfile as sf
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from acforensics.constants import BONAFIDE, SPOOF, TARGET_SAMPLES
from acforensics.data.datasets import (
    ASVspoofDataset,
    collate_fn,
    get_balanced_sampler,
)


@pytest.fixture
def mock_dataset_files(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)

    records = []
    for i in range(10):
        file_id = f"test_{i}"
        audio_file = audio_dir / f"{file_id}.flac"
        sig = np.random.uniform(-0.5, 0.5, 32000).astype(np.float32)  # 2.0s
        sf.write(str(audio_file), sig, 16000)

        label = BONAFIDE if i < 2 else SPOOF
        records.append({
            "speaker_id": f"spk_{i}",
            "file_id": file_id,
            "env": "-",
            "attack_id": "-" if label == BONAFIDE else "A01",
            "key": "bonafide" if label == BONAFIDE else "spoof",
            "label": label,
            "split": "train",
            "audio_path": str(audio_file),
        })

    df = pd.DataFrame(records)
    return df


def test_asvspoof_dataset_len_and_item(mock_dataset_files):
    df = mock_dataset_files
    dataset = ASVspoofDataset(protocol_df=df, target_samples=64000, split="train")

    assert len(dataset) == 10
    item = dataset[0]

    assert "waveform" in item
    assert "label" in item
    assert "file_id" in item
    assert "speaker_id" in item
    assert "attack_id" in item
    assert item["waveform"].shape == (64000,)
    assert isinstance(item["label"], torch.Tensor)


def test_collate_fn(mock_dataset_files):
    df = mock_dataset_files
    dataset = ASVspoofDataset(protocol_df=df, target_samples=64000, split="train")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    batch = next(iter(dataloader))
    assert batch["waveform"].shape == (4, 64000)
    assert batch["label"].shape == (4,)
    assert len(batch["file_id"]) == 4


def test_balanced_sampler(mock_dataset_files):
    df = mock_dataset_files
    dataset = ASVspoofDataset(protocol_df=df, target_samples=64000, split="train")
    sampler = get_balanced_sampler(dataset)

    assert sampler.num_samples == len(dataset)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    assert batch["waveform"].shape == (4, 64000)
