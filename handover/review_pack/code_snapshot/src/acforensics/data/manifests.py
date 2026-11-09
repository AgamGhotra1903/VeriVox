"""Unified manifest generation, validation, and storage (Parquet / Pandas).

Produces standardized manifests matching the AF-Split v1 specification:
Columns: path, dataset, split, speaker, label, attack_id, duration, codec, bitrate, gender

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

from pathlib import Path
from typing import Any, Optional
import soundfile as sf
import pandas as pd

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    MANIFEST_COLUMNS,
    DEFAULT_MANIFESTS_DIR,
)
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def probe_audio_file(file_path: Path | str) -> dict[str, Any]:
    """Inspects audio metadata from file header without loading entire audio waveform.

    Returns:
        dict with duration (sec), sample_rate, channels, codec/format, bitrate.
    """
    path = Path(file_path)
    if not path.exists():
        return {
            "duration": 0.0,
            "sample_rate": 16000,
            "channels": 1,
            "codec": path.suffix.lstrip(".").lower() or "unknown",
            "bitrate": 0,
        }

    try:
        info = sf.info(str(path))
        return {
            "duration": float(info.duration),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "codec": str(info.format).lower(),
            "bitrate": int(info.samplerate * info.channels * 16),  # nominal PCM bitrate
        }
    except Exception as exc:
        logger.debug("Failed to read header info for %s: %s", path, exc)
        return {
            "duration": 0.0,
            "sample_rate": 16000,
            "channels": 1,
            "codec": path.suffix.lstrip(".").lower() or "unknown",
            "bitrate": 0,
        }


def build_manifest_from_records(
    records: list[dict[str, Any]] | pd.DataFrame,
    dataset_name: str,
    split_name: str,
    probe_audio: bool = False,
) -> pd.DataFrame:
    """Builds a standardized DataFrame adhering to the project manifest schema.

    Schema:
    - path: str (absolute or relative audio path)
    - dataset: str ('asvspoof2019_la', 'in_the_wild', 'indic_mini', etc.)
    - split: str ('train', 'dev', 'eval', 'itw_cal', 'itw_test', etc.)
    - speaker: str (speaker identifier)
    - label: int (1 = bonafide, 0 = spoof)
    - attack_id: str ('-' for bonafide, 'A01'..'A19', 'vocoder_name', etc.)
    - duration: float (duration in seconds)
    - codec: str ('flac', 'wav', 'mp3', etc.)
    - bitrate: int (bitrate in bps, or 0 if unprobed)
    - gender: str ('M', 'F', or 'unknown')

    Args:
        records: Raw records list or DataFrame.
        dataset_name: Dataset identifier.
        split_name: Split partition identifier.
        probe_audio: If True, reads header info from existing audio files.

    Returns:
        Standardized pandas DataFrame.
    """
    df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()

    # Normalize standard field names
    col_mapping = {
        "audio_path": "path",
        "file_path": "path",
        "speaker_id": "speaker",
        "key": "raw_label",
    }
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # Required columns defaults
    if "dataset" not in df.columns:
        df["dataset"] = dataset_name
    if "split" not in df.columns:
        df["split"] = split_name
    if "speaker" not in df.columns:
        df["speaker"] = "unknown"
    if "attack_id" not in df.columns:
        df["attack_id"] = "-"
    if "gender" not in df.columns:
        df["gender"] = "unknown"
    if "codec" not in df.columns:
        df["codec"] = df["path"].apply(lambda p: Path(str(p)).suffix.lstrip(".").lower() or "flac")
    if "duration" not in df.columns:
        df["duration"] = 0.0
    if "bitrate" not in df.columns:
        df["bitrate"] = 0

    # Ensure label is strictly integer (1 or 0)
    df["label"] = df["label"].astype(int)

    # Optional header probing
    if probe_audio and len(df) > 0:
        logger.info("Probing audio headers for %d files in %s/%s...", len(df), dataset_name, split_name)
        durations = []
        codecs = []
        bitrates = []
        for p in df["path"]:
            info = probe_audio_file(p)
            durations.append(info["duration"])
            codecs.append(info["codec"])
            bitrates.append(info["bitrate"])
        df["duration"] = durations
        df["codec"] = codecs
        df["bitrate"] = bitrates

    # Filter and reorder strictly to MANIFEST_COLUMNS
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in ("path", "dataset", "split", "speaker", "attack_id", "codec", "gender") else 0

    manifest_df = df[MANIFEST_COLUMNS].copy()
    manifest_df["speaker"] = manifest_df["speaker"].astype(str)
    manifest_df["attack_id"] = manifest_df["attack_id"].astype(str)
    manifest_df["path"] = manifest_df["path"].astype(str)
    manifest_df["dataset"] = manifest_df["dataset"].astype(str)
    manifest_df["split"] = manifest_df["split"].astype(str)
    manifest_df["gender"] = manifest_df["gender"].astype(str)
    manifest_df["codec"] = manifest_df["codec"].astype(str)
    manifest_df["duration"] = manifest_df["duration"].astype(float)
    manifest_df["bitrate"] = manifest_df["bitrate"].astype(int)
    manifest_df["label"] = manifest_df["label"].astype(int)

    return manifest_df.reset_index(drop=True)


def save_manifest(
    df: pd.DataFrame,
    output_path: Path | str,
    save_csv_mirror: bool = True,
) -> Path:
    """Saves standardized manifest DataFrame to Parquet format (with optional CSV mirror).

    Args:
        df: Manifest DataFrame.
        output_path: Path to target file (e.g. data/manifests/train.parquet).
        save_csv_mirror: Also saves a .csv for human inspection.

    Returns:
        Path to the saved Parquet file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix != ".parquet":
        parquet_path = path.with_suffix(".parquet")
    else:
        parquet_path = path

    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Saved manifest (%d records) to: %s", len(df), parquet_path)

    if save_csv_mirror:
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.debug("Saved CSV mirror to: %s", csv_path)

    return parquet_path


def load_manifest(manifest_path: Path | str) -> pd.DataFrame:
    """Loads a standardized manifest DataFrame from Parquet or CSV.

    Args:
        manifest_path: Path to Parquet or CSV file.

    Returns:
        Standardized pandas DataFrame.
    """
    path = Path(manifest_path)
    if not path.exists():
        # Check if alternative format exists
        if path.suffix == ".parquet" and path.with_suffix(".csv").exists():
            path = path.with_suffix(".csv")
        elif path.suffix == ".csv" and path.with_suffix(".parquet").exists():
            path = path.with_suffix(".parquet")
        else:
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path, engine="pyarrow")
    else:
        df = pd.read_csv(path)

    # Validate essential columns
    for col in ["path", "label", "speaker"]:
        if col not in df.columns:
            raise ValueError(f"Manifest {path.name} is missing mandatory column: '{col}'")

    return df


def get_manifest_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Calculates comprehensive descriptive statistics for a manifest partition."""
    total = len(df)
    if total == 0:
        return {"total": 0, "bonafide": 0, "spoof": 0}

    bonafide = int((df["label"] == BONAFIDE).sum())
    spoof = int((df["label"] == SPOOF).sum())
    unique_speakers = int(df["speaker"].nunique())
    attacks = df[df["label"] == SPOOF]["attack_id"].value_counts().to_dict()
    durations = df[df["duration"] > 0]["duration"]

    mean_duration = float(durations.mean()) if len(durations) > 0 else 0.0
    total_hours = float(durations.sum()) / 3600.0 if len(durations) > 0 else 0.0

    return {
        "total": total,
        "bonafide": bonafide,
        "bonafide_ratio": bonafide / total,
        "spoof": spoof,
        "spoof_ratio": spoof / total,
        "unique_speakers": unique_speakers,
        "mean_duration_sec": mean_duration,
        "total_hours": total_hours,
        "attack_breakdown": {str(k): int(v) for k, v in attacks.items()},
    }
