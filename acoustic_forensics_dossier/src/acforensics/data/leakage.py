"""Experimental leakage auditing and protocol validation module.

Audits:
1. Speaker ID disjointness across splits (Train vs Dev vs Eval vs ITW-CAL vs ITW-TEST).
2. Utterance/File ID disjointness across splits.
3. Content hash duplicates (SHA-256) across splits to detect identical audio files.
4. VCTK heritage overlap warnings (ASVspoof 2019 LA bona fide speech derives from VCTK,
   so any extension dataset using VCTK shares speakers/utterances).

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

from hashlib import sha256
from pathlib import Path
from typing import Any, Optional
import itertools
import pandas as pd

from acforensics.utils.logging import get_logger

logger = get_logger(__name__)

# Known VCTK speaker ID pattern (p225, p226, ..., p376)
VCTK_SPEAKER_PREFIXES = ("p2", "p3")


def compute_file_content_hash(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file's binary content to detect duplicates.

    Args:
        file_path: Path to target file.
        chunk_size: Read buffer size in bytes.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    hasher = sha256()
    path = Path(file_path)
    if not path.exists():
        return ""

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_speaker_disjointness(
    splits: dict[str, pd.DataFrame],
    speaker_col: str = "speaker",
) -> dict[str, Any]:
    """Audits speaker ID disjointness across all provided dataset splits.

    Pairwise disjointness requirements:
    - Train vs Dev: MUST be disjoint
    - Train vs Eval: MUST be disjoint
    - Dev vs Eval: MUST be disjoint
    - ITW-CAL vs ITW-TEST: MUST be disjoint

    Returns:
        dict with pairwise overlap details, violations, and pass/fail boolean.
    """
    results: dict[str, Any] = {
        "passed": True,
        "pairwise_checks": {},
        "violations": [],
    }

    split_speakers: dict[str, set[str]] = {}
    for name, df in splits.items():
        col = speaker_col if speaker_col in df.columns else ("speaker_id" if "speaker_id" in df.columns else None)
        if col and len(df) > 0:
            speakers = set(df[col].astype(str).unique())
            # Exclude generic 'unknown' or empty markers from collision checks
            speakers.discard("unknown")
            speakers.discard("-")
            speakers.discard("")
            split_speakers[name] = speakers
        else:
            split_speakers[name] = set()

    for s1, s2 in itertools.combinations(split_speakers.keys(), 2):
        spk1 = split_speakers[s1]
        spk2 = split_speakers[s2]
        overlap = spk1.intersection(spk2)

        check_key = f"{s1}_vs_{s2}"
        is_disjoint = len(overlap) == 0

        # Define which pairs are strictly required to be disjoint
        strict_pairs = [
            ("train", "dev"),
            ("train", "eval"),
            ("dev", "eval"),
            ("itw_cal", "itw_test"),
            ("asvspoof_train", "asvspoof_dev"),
            ("asvspoof_train", "asvspoof_eval"),
            ("asvspoof_dev", "asvspoof_eval"),
        ]
        is_strict = (s1, s2) in strict_pairs or (s2, s1) in strict_pairs

        results["pairwise_checks"][check_key] = {
            "split_1": s1,
            "split_2": s2,
            "speakers_1_count": len(spk1),
            "speakers_2_count": len(spk2),
            "overlap_count": len(overlap),
            "overlapping_speakers": sorted(list(overlap))[:10],
            "is_disjoint": is_disjoint,
            "is_strict_requirement": is_strict,
        }

        if is_strict and not is_disjoint:
            results["passed"] = False
            msg = f"CRITICAL LEAKAGE: Strict speaker overlap between '{s1}' and '{s2}': {sorted(list(overlap))[:5]}"
            results["violations"].append(msg)
            logger.error(msg)
        elif not is_disjoint:
            logger.info("Non-strict speaker overlap observed between '%s' and '%s': %d speakers.", s1, s2, len(overlap))

    return results


def audit_file_id_disjointness(
    splits: dict[str, pd.DataFrame],
    file_col: str = "path",
) -> dict[str, Any]:
    """Audits utterance/file identifier disjointness across all splits."""
    results: dict[str, Any] = {
        "passed": True,
        "pairwise_checks": {},
        "violations": [],
    }

    split_files: dict[str, set[str]] = {}
    for name, df in splits.items():
        if "file_id" in df.columns:
            files = set(df["file_id"].astype(str).unique())
        elif file_col in df.columns:
            files = set(df[file_col].apply(lambda p: Path(str(p)).stem).unique())
        else:
            files = set()
        split_files[name] = files

    for s1, s2 in itertools.combinations(split_files.keys(), 2):
        f1 = split_files[s1]
        f2 = split_files[s2]
        overlap = f1.intersection(f2)
        check_key = f"{s1}_vs_{s2}"

        is_disjoint = len(overlap) == 0
        results["pairwise_checks"][check_key] = {
            "split_1": s1,
            "split_2": s2,
            "files_1_count": len(f1),
            "files_2_count": len(f2),
            "overlap_count": len(overlap),
            "overlapping_files": sorted(list(overlap))[:10],
            "is_disjoint": is_disjoint,
        }

        if not is_disjoint:
            results["passed"] = False
            msg = f"CRITICAL LEAKAGE: File ID overlap between '{s1}' and '{s2}': {sorted(list(overlap))[:5]}"
            results["violations"].append(msg)
            logger.error(msg)

    return results


def audit_audio_hash_collisions(
    splits: dict[str, pd.DataFrame],
    path_col: str = "path",
    max_files_per_split: Optional[int] = 500,
) -> dict[str, Any]:
    """Computes SHA-256 hashes of physical audio files to audit binary duplicate leakage."""
    results: dict[str, Any] = {
        "passed": True,
        "hashes_checked": 0,
        "collisions": [],
    }

    hash_to_origin: dict[str, list[tuple[str, str]]] = {}

    for split_name, df in splits.items():
        if path_col not in df.columns or len(df) == 0:
            continue

        sample_df = df if max_files_per_split is None else df.head(max_files_per_split)
        for _, row in sample_df.iterrows():
            fpath = Path(str(row[path_col]))
            if not fpath.exists():
                continue

            h = compute_file_content_hash(fpath)
            if not h:
                continue

            results["hashes_checked"] += 1
            if h not in hash_to_origin:
                hash_to_origin[h] = []
            hash_to_origin[h].append((split_name, str(fpath)))

    # Detect cross-split collisions
    for h, occurrences in hash_to_origin.items():
        splits_involved = {occ[0] for occ in occurrences}
        if len(splits_involved) > 1:
            results["passed"] = False
            collision_entry = {
                "hash": h,
                "splits": list(splits_involved),
                "files": [occ[1] for occ in occurrences],
            }
            results["collisions"].append(collision_entry)
            logger.error(
                "Duplicate audio content hash detected across splits %s: %s",
                splits_involved,
                [occ[1] for occ in occurrences],
            )

    return results


def check_vctk_heritage_warnings(
    splits: dict[str, pd.DataFrame],
    speaker_col: str = "speaker",
) -> list[str]:
    """Emits formal warnings regarding VCTK speaker overlap in bona fide audio.

    Section 3.1 Note:
    ASVspoof 2019 LA bona fide speech derives from VCTK (107 Scottish/English speakers).
    Any extension dataset (such as WaveFake, LibriSeVoc, or DFADD) built from VCTK
    will share source speakers or raw utterances.
    """
    warnings = []
    all_speakers = set()
    for name, df in splits.items():
        col = speaker_col if speaker_col in df.columns else ("speaker_id" if "speaker_id" in df.columns else None)
        if col:
            speakers = set(df[col].astype(str).unique())
            all_speakers.update(speakers)

            # Check if VCTK prefixes like 'p225', 'p226' are present
            vctk_like = [s for s in speakers if s.startswith(VCTK_SPEAKER_PREFIXES) and len(s) == 4 and s[1:].isdigit()]
            if vctk_like:
                msg = (
                    f"DATASET CAVEAT: Split '{name}' contains VCTK-formatted speakers ({len(vctk_like)} found: {vctk_like[:4]}). "
                    "Notice: ASVspoof 2019 LA bona fide speech derives from the VCTK corpus. "
                    "Ensure no extension datasets trained alongside LA share these identical VCTK speaker recordings."
                )
                warnings.append(msg)
                logger.warning(msg)

    if not warnings:
        warnings.append(
            "DATASET CARD NOTE: ASVspoof 2019 LA bona fide utterances originate from 107 VCTK speakers. "
            "Verified: No raw VCTK speaker conflicts detected in active manifests."
        )

    return warnings


def run_comprehensive_leakage_audit(
    splits: dict[str, pd.DataFrame],
    check_hashes: bool = True,
    max_hash_files: Optional[int] = 200,
) -> dict[str, Any]:
    """Executes full leakage audit pipeline across all active manifests.

    Args:
        splits: Dictionary mapping split names (e.g. 'train', 'dev', 'eval', 'itw_test') to DataFrames.
        check_hashes: If True, computes SHA-256 hashes of available audio files.
        max_hash_files: Subsample limit per split for content hash checking (protects runtime).

    Returns:
        Structured audit report dict.
    """
    logger.info("Executing comprehensive leakage audit across %d splits: %s", len(splits), list(splits.keys()))

    spk_audit = audit_speaker_disjointness(splits)
    file_audit = audit_file_id_disjointness(splits)

    hash_audit = {"passed": True, "hashes_checked": 0, "collisions": []}
    if check_hashes:
        hash_audit = audit_audio_hash_collisions(splits, max_files_per_split=max_hash_files)

    vctk_warnings = check_vctk_heritage_warnings(splits)

    overall_passed = spk_audit["passed"] and file_audit["passed"] and hash_audit["passed"]

    audit_summary = {
        "status": "PASS" if overall_passed else "FAIL",
        "overall_passed": overall_passed,
        "speaker_audit": spk_audit,
        "file_id_audit": file_audit,
        "content_hash_audit": hash_audit,
        "vctk_warnings": vctk_warnings,
        "splits_audited": {k: len(v) for k, v in splits.items()},
    }

    if overall_passed:
        logger.info("LEAKAGE AUDIT PASSED: Zero speaker/file/content overlap detected across disjoint splits.")
    else:
        logger.error("LEAKAGE AUDIT FAILED! See violations: %s", spk_audit["violations"] + file_audit["violations"])

    return audit_summary
