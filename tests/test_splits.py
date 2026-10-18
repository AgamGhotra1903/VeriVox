"""Unit tests for dataset splitting and speaker disjointness isolation ("AF-Split v1").

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

import pandas as pd
import pytest

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.data.splits import (
    split_in_the_wild_by_speaker,
    split_extension_by_generator,
    summarize_splits,
)


def test_split_in_the_wild_speaker_disjointness():
    """Verifies that ITW-CAL and ITW-TEST partitions have ZERO speaker overlap."""
    speakers = [f"speaker_{i:02d}" for i in range(20)]
    records = []
    for spk in speakers:
        for idx in range(10):
            records.append({
                "audio_path": f"dummy/{spk}_{idx}.wav",
                "speaker_id": spk,
                "label": BONAFIDE if idx % 2 == 0 else SPOOF,
            })
    df = pd.DataFrame(records)

    cal_df, test_df = split_in_the_wild_by_speaker(df, cal_ratio=0.20, seed=42)

    cal_speakers = set(cal_df["speaker_id"])
    test_speakers = set(test_df["speaker_id"])

    # Strict disjointness check
    assert len(cal_speakers.intersection(test_speakers)) == 0, "Speaker overlap detected between ITW-CAL and ITW-TEST!"
    # Check that ~20% of speakers were assigned to CAL (4 of 20)
    assert len(cal_speakers) == 4
    assert len(test_speakers) == 16
    assert len(cal_df) + len(test_df) == len(df)
    assert set(cal_df["split"].unique()) == {"itw_cal"}
    assert set(test_df["split"].unique()) == {"itw_test"}


def test_split_in_the_wild_single_speaker_fallback():
    """Ensures graceful handling when only a single speaker exists in the dataset."""
    records = [{"audio_path": f"clip_{i}.wav", "speaker_id": "spk_1", "label": BONAFIDE} for i in range(5)]
    df = pd.DataFrame(records)

    cal_df, test_df = split_in_the_wild_by_speaker(df, cal_ratio=0.15, seed=42)
    assert len(cal_df) == 0
    assert len(test_df) == len(df)


def test_split_extension_by_generator():
    """Verifies holding out unseen synthesis generators for multi-domain benchmark."""
    records = []
    # 4 generators
    for gen in ["vocoder_a", "vocoder_b", "diffusion_c", "flow_d"]:
        for i in range(15):
            records.append({
                "path": f"{gen}_{i}.wav",
                "attack_id": gen,
                "label": SPOOF,
                "speaker_id": f"spk_{i % 5}",
            })
    # Add some bona fide speech
    for spk in [f"real_spk_{i}" for i in range(6)]:
        for j in range(5):
            records.append({
                "path": f"{spk}_{j}.wav",
                "attack_id": "-",
                "label": BONAFIDE,
                "speaker_id": spk,
            })

    df = pd.DataFrame(records)

    train_x, val_x, test_x = split_extension_by_generator(
        df,
        test_generator="flow_d",
        val_generator="diffusion_c",
        generator_col="attack_id",
        speaker_col="speaker_id",
        seed=42,
    )

    # flow_d must appear ONLY in test_x
    assert (test_x[test_x["label"] == SPOOF]["attack_id"] == "flow_d").all()
    assert "flow_d" not in train_x["attack_id"].values
    assert "flow_d" not in val_x["attack_id"].values

    # diffusion_c must appear ONLY in val_x
    assert (val_x[val_x["label"] == SPOOF]["attack_id"] == "diffusion_c").all()
    assert "diffusion_c" not in train_x["attack_id"].values
    assert "diffusion_c" not in test_x["attack_id"].values


def test_summarize_splits():
    """Verifies that split summary table outputs correct counts and columns."""
    df_train = pd.DataFrame({
        "label": [BONAFIDE] * 10 + [SPOOF] * 40,
        "speaker_id": [f"spk_{i}" for i in range(50)],
        "attack_id": ["-"] * 10 + ["A01"] * 40,
    })
    summary = summarize_splits({"train": df_train})

    assert "total" in summary.columns
    assert "bonafide" in summary.columns
    assert "spoof" in summary.columns
    assert summary.iloc[0]["total"] == 50
    assert summary.iloc[0]["bonafide"] == 10
    assert summary.iloc[0]["spoof"] == 40
