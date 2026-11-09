"""Unit tests for standardized Parquet manifest creation, validation, and storage.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

from pathlib import Path
import pandas as pd
import pytest

from acforensics.constants import BONAFIDE, SPOOF, MANIFEST_COLUMNS
from acforensics.data.manifests import (
    build_manifest_from_records,
    save_manifest,
    load_manifest,
    get_manifest_stats,
)


def test_build_manifest_schema_and_columns():
    """Verifies that build_manifest_from_records produces exactly the required columns."""
    raw_records = [
        {"audio_path": "audio/clip1.flac", "speaker_id": "LA_0001", "label": BONAFIDE, "attack_id": "-"},
        {"audio_path": "audio/clip2.flac", "speaker_id": "LA_0002", "label": SPOOF, "attack_id": "A01"},
    ]

    manifest = build_manifest_from_records(
        raw_records,
        dataset_name="asvspoof2019_la",
        split_name="train",
        probe_audio=False,
    )

    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert len(manifest) == 2
    assert manifest.iloc[0]["dataset"] == "asvspoof2019_la"
    assert manifest.iloc[0]["split"] == "train"
    assert manifest.iloc[0]["speaker"] == "LA_0001"
    assert manifest.iloc[0]["label"] == BONAFIDE
    assert manifest.iloc[1]["label"] == SPOOF
    assert manifest.iloc[1]["attack_id"] == "A01"


def test_save_and_load_manifest_parquet(tmp_path: Path):
    """Verifies lossless round-trip serialization to Parquet and CSV mirror."""
    raw_records = [
        {"audio_path": f"audio/test_{i}.flac", "speaker_id": f"spk_{i}", "label": i % 2, "attack_id": "A02"}
        for i in range(25)
    ]
    manifest = build_manifest_from_records(raw_records, dataset_name="test_ds", split_name="dev")

    parquet_file = tmp_path / "dev.parquet"
    save_manifest(manifest, parquet_file, save_csv_mirror=True)

    assert parquet_file.exists()
    assert (tmp_path / "dev.csv").exists()

    loaded = load_manifest(parquet_file)
    assert len(loaded) == len(manifest)
    assert list(loaded.columns) == MANIFEST_COLUMNS
    assert (loaded["label"].values == manifest["label"].values).all()


def test_get_manifest_stats():
    """Verifies descriptive statistics extraction for manifest DataFrames."""
    df = pd.DataFrame({
        "path": ["a.flac", "b.flac", "c.flac"],
        "dataset": ["ds"] * 3,
        "split": ["train"] * 3,
        "speaker": ["s1", "s2", "s1"],
        "label": [BONAFIDE, SPOOF, SPOOF],
        "attack_id": ["-", "A01", "A02"],
        "duration": [3.5, 4.0, 2.5],
        "codec": ["flac"] * 3,
        "bitrate": [256000] * 3,
        "gender": ["M", "F", "M"],
    })

    stats = get_manifest_stats(df)
    assert stats["total"] == 3
    assert stats["bonafide"] == 1
    assert stats["spoof"] == 2
    assert stats["unique_speakers"] == 2
    assert pytest.approx(stats["mean_duration_sec"], 0.01) == 3.333
