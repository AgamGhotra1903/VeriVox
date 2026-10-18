"""Protocol parsers and metadata handlers for ASVspoof 2019 LA and In-The-Wild.

Author: Paras Bansal (IIT2024221)
"""

from pathlib import Path
from typing import Any, Optional
import pandas as pd

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    LABEL_STR_MAP,
    KNOWN_ATTACKS,
    UNSEEN_ATTACKS,
)
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def parse_asvspoof_line(line: str) -> Optional[dict[str, Any]]:
    """Parses a single line from an ASVspoof 2019 LA protocol file.

    Official ASVspoof 2019 LA format:
    SPEAKER_ID FILE_ID - ATTACK_ID KEY
    Example bonafide: LA_0079 LA_D_1047731 - - bonafide
    Example spoof:    LA_0079 LA_D_1105538 - A06 spoof

    Returns:
        Dictionary with parsed fields, or None if the line is invalid/empty.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    speaker_id = parts[0]
    file_id = parts[1]
    env = parts[2]
    attack_id = parts[3]
    key = parts[4].lower()

    if key not in LABEL_STR_MAP:
        logger.warning("Unrecognized key '%s' in protocol line: %s", key, line)
        return None

    label = LABEL_STR_MAP[key]

    return {
        "speaker_id": speaker_id,
        "file_id": file_id,
        "env": env,
        "attack_id": attack_id,
        "key": key,
        "label": label,
    }


def load_asvspoof_protocol(
    protocol_path: Path | str,
    audio_dir: Optional[Path | str] = None,
    split_name: str = "train",
) -> pd.DataFrame:
    """Loads and standardizes an official ASVspoof 2019 LA protocol file.

    Args:
        protocol_path: Path to the protocol text file (e.g. ASVspoof2019.LA.cm.train.trn.txt).
        audio_dir: Optional directory containing audio files (FLAC or WAV).
            If provided, verifies existence or sets 'audio_path'.
        split_name: Name of split ('train', 'dev', or 'eval').

    Returns:
        Standardized pandas DataFrame containing columns:
        ['speaker_id', 'file_id', 'env', 'attack_id', 'key', 'label', 'split', 'audio_path']

    Raises:
        FileNotFoundError: If the protocol file is not found.
    """
    path = Path(protocol_path)
    if not path.exists():
        raise FileNotFoundError(f"ASVspoof protocol file not found at: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_asvspoof_line(line)
            if parsed is not None:
                parsed["split"] = split_name
                # Determine audio file path if audio_dir is provided
                if audio_dir is not None:
                    audio_root = Path(audio_dir)
                    # ASVspoof audio files are typically .flac, but support .wav fallback
                    candidate_flac = audio_root / f"{parsed['file_id']}.flac"
                    candidate_wav = audio_root / f"{parsed['file_id']}.wav"
                    if candidate_flac.exists():
                        parsed["audio_path"] = str(candidate_flac)
                    elif candidate_wav.exists():
                        parsed["audio_path"] = str(candidate_wav)
                    else:
                        parsed["audio_path"] = str(candidate_flac)
                else:
                    parsed["audio_path"] = ""
                records.append(parsed)

    df = pd.DataFrame(records)
    logger.info(
        "Loaded %s protocol (%s): %d total records (Bona fide: %d, Spoof: %d)",
        split_name,
        path.name,
        len(df),
        (df["label"] == BONAFIDE).sum(),
        (df["label"] == SPOOF).sum(),
    )
    return df


def load_in_the_wild_protocol(
    meta_path: Path | str,
    audio_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Loads and normalizes the In-The-Wild deepfake audio metadata (Müller et al. 2022).

    Expected CSV columns in meta.csv:
    file, speaker, label (values: 'bona-fide', 'spoof', etc.)

    Args:
        meta_path: Path to meta.csv.
        audio_dir: Base directory containing audio files.

    Returns:
        Standardized pandas DataFrame with project conventions (bonafide=1, spoof=0).
    """
    path = Path(meta_path)
    if not path.exists():
        raise FileNotFoundError(f"In-The-Wild metadata file not found at: {path}")

    raw_df = pd.read_csv(path)

    # Normalize column names
    col_map = {c: c.strip().lower() for c in raw_df.columns}
    raw_df = raw_df.rename(columns=col_map)

    # Determine speaker and file columns
    file_col = "file" if "file" in raw_df.columns else raw_df.columns[0]
    speaker_col = "speaker" if "speaker" in raw_df.columns else "speaker_id"
    label_col = "label" if "label" in raw_df.columns else "key"

    records = []
    for _, row in raw_df.iterrows():
        raw_key = str(row[label_col]).strip().lower()
        if raw_key in LABEL_STR_MAP:
            label = LABEL_STR_MAP[raw_key]
        else:
            logger.warning("Unknown label in ITW metadata: %s", raw_key)
            continue

        filename = str(row[file_col])
        file_id = Path(filename).stem
        speaker_id = str(row[speaker_col]) if speaker_col in row else "unknown"

        if audio_dir is not None:
            audio_path = str(Path(audio_dir) / filename)
        else:
            audio_path = filename

        records.append({
            "speaker_id": speaker_id,
            "file_id": file_id,
            "env": "in_the_wild",
            "attack_id": "unseen_itw",
            "key": "bonafide" if label == BONAFIDE else "spoof",
            "label": label,
            "split": "cross_domain",
            "audio_path": audio_path,
        })

    df = pd.DataFrame(records)
    logger.info(
        "Loaded In-The-Wild protocol: %d utterances (Bona fide: %d, Spoof: %d)",
        len(df),
        (df["label"] == BONAFIDE).sum(),
        (df["label"] == SPOOF).sum(),
    )
    return df


def get_protocol_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Computes comprehensive summary statistics for a dataset split.

    Args:
        df: DataFrame containing protocol records.

    Returns:
        Dictionary with count, percentages, speakers count, and attack breakdown.
    """
    total = len(df)
    if total == 0:
        return {"total": 0, "bonafide_count": 0, "spoof_count": 0}

    bonafide_count = int((df["label"] == BONAFIDE).sum())
    spoof_count = int((df["label"] == SPOOF).sum())
    unique_speakers = int(df["speaker_id"].nunique())

    # Attack breakdown
    attack_counts = df[df["label"] == SPOOF]["attack_id"].value_counts().to_dict()

    return {
        "total": total,
        "bonafide_count": bonafide_count,
        "bonafide_ratio": bonafide_count / total,
        "spoof_count": spoof_count,
        "spoof_ratio": spoof_count / total,
        "unique_speakers": unique_speakers,
        "attack_counts": {str(k): int(v) for k, v in attack_counts.items()},
    }
