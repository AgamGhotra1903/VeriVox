"""CLI script to audit speaker and content leakage across all dataset splits.

Verifies:
1. Zero speaker overlap across Train, Dev, Eval, ITW-CAL, and ITW-TEST splits.
2. Zero utterance/file ID collisions.
3. Binary audio content hash duplicates (SHA-256).
4. VCTK heritage overlap warnings.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional

from acforensics.config import load_config
from acforensics.constants import DEFAULT_MANIFESTS_DIR
from acforensics.data.manifests import load_manifest
from acforensics.data.leakage import run_comprehensive_leakage_audit
from acforensics.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def audit_leakage(
    manifests_dir: Path | str = DEFAULT_MANIFESTS_DIR,
    output_json: Optional[Path | str] = None,
    check_hashes: bool = True,
    max_hash_files: int = 200,
) -> dict:
    """Loads all existing manifests and executes the comprehensive leakage audit.

    Args:
        manifests_dir: Path containing {split}.parquet files.
        output_json: Optional path to save full audit results JSON.
        check_hashes: Whether to compute SHA-256 binary audio hashes.
        max_hash_files: Subsample limit per split for content hash checking.

    Returns:
        Structured audit report dict.
    """
    m_dir = Path(manifests_dir)
    if not m_dir.exists():
        raise FileNotFoundError(f"Manifests directory not found: {m_dir}. Run scripts/prepare_manifests.py first.")

    manifest_files = list(m_dir.glob("*.parquet"))
    if not manifest_files:
        manifest_files = list(m_dir.glob("*.csv"))

    if not manifest_files:
        raise FileNotFoundError(f"No parquet or csv manifest files found in {m_dir}.")

    logger.info("Found %d manifest files to audit in %s:", len(manifest_files), m_dir)
    splits = {}
    for mf in manifest_files:
        split_name = mf.stem
        splits[split_name] = load_manifest(mf)
        logger.info("  Loaded split '%s': %d records", split_name, len(splits[split_name]))

    audit_result = run_comprehensive_leakage_audit(
        splits=splits,
        check_hashes=check_hashes,
        max_hash_files=max_hash_files,
    )

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(audit_result, f, indent=2)
        logger.info("Saved complete leakage audit report to: %s", out_p)

    return audit_result


def main():
    parser = argparse.ArgumentParser(description="Audit dataset splits for speaker, file, and content leakage.")
    parser.add_argument("--manifests-dir", type=str, default="data/manifests", help="Path to manifest parquet files")
    parser.add_argument("--output-json", type=str, default="results/audit/leakage_audit.json", help="Path to save audit JSON")
    parser.add_argument("--skip-hashes", action="store_true", help="Skip binary audio content hashing (faster)")
    parser.add_argument("--max-hash-files", type=int, default=200, help="Max files to hash per split")
    args = parser.parse_args()

    setup_logging()
    logger.info("Starting Leakage Audit...")

    try:
        report = audit_leakage(
            manifests_dir=args.manifests_dir,
            output_json=args.output_json,
            check_hashes=not args.skip_hashes,
            max_hash_files=args.max_hash_files,
        )

        print("\n" + "=" * 60)
        print(f"LEAKAGE AUDIT RESULT: {report['status']}")
        print("=" * 60)
        print("Pairwise Speaker Disjointness:")
        for pair_name, details in report["speaker_audit"]["pairwise_checks"].items():
            res_str = "PASS (Disjoint)" if details["is_disjoint"] else "FAIL (Overlap!)"
            print(f"  {pair_name:25s}: {res_str} - Overlap count: {details['overlap_count']}")

        print("\nPairwise File ID Disjointness:")
        for pair_name, details in report["file_id_audit"]["pairwise_checks"].items():
            res_str = "PASS (Disjoint)" if details["is_disjoint"] else "FAIL (Overlap!)"
            print(f"  {pair_name:25s}: {res_str} - Overlap count: {details['overlap_count']}")

        print("\nBinary Content Hash Duplicates:")
        print(f"  Files hashed: {report['content_hash_audit']['hashes_checked']}")
        print(f"  Collisions detected across splits: {len(report['content_hash_audit']['collisions'])}")

        print("\nDataset Heritage Warnings:")
        for w in report["vctk_warnings"]:
            print(f"  - {w}")
        print("=" * 60 + "\n")

        if not report["overall_passed"]:
            logger.error("Audit detected critical leakage! Exiting with error code 1.")
            sys.exit(1)

        logger.info("Leakage audit completed with zero violations.")
        sys.exit(0)

    except Exception as exc:
        logger.error("Audit failed with exception: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
