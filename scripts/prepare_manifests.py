"""CLI script to build and serialize standardized Parquet manifests for AF-Split v1.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

import argparse
from pathlib import Path
import pandas as pd

from acforensics.config import load_config
from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MANIFESTS_DIR,
    OFFICIAL_COUNTS,
)
from acforensics.data.protocols import (
    load_asvspoof_protocol,
    load_in_the_wild_protocol,
    get_protocol_statistics,
)
from acforensics.data.splits import (
    split_in_the_wild_by_speaker,
    summarize_splits,
)
from acforensics.data.manifests import (
    build_manifest_from_records,
    save_manifest,
    get_manifest_stats,
)
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def prepare_manifests(
    config,
    output_dir: Path | str = DEFAULT_MANIFESTS_DIR,
    debug: bool = False,
    probe_audio: bool = False,
) -> dict[str, pd.DataFrame]:
    """Generates standardized Parquet manifests for all partitions in AF-Split v1.

    Args:
        config: MainConfig instance.
        output_dir: Directory where .parquet files are saved.
        debug: If True, uses mock/smoke test data for quick validation.
        probe_audio: If True, probes physical audio headers for exact durations and bitrates.

    Returns:
        dict mapping split names to manifest DataFrames.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if debug:
        logger.info("Operating in --debug / smoke-test mode.")
        data_root = Path(config.smoke_test.data_dir)
        proto_dir = data_root / "ASVspoof2019_LA_cm_protocols"
        itw_dir = data_root / "in_the_wild"
        itw_meta = itw_dir / "meta.csv"
        itw_audio = itw_dir / "audio"
    else:
        data_root = Path(config.data.asvspoof_root)
        proto_dir = Path(config.data.protocols_dir)
        itw_meta = Path(config.data.in_the_wild_meta)
        itw_audio = Path(config.data.in_the_wild_root) / "audio"

    manifests: dict[str, pd.DataFrame] = {}

    # 1. ASVspoof 2019 LA Splits
    asv_specs = [
        ("train", config.data.train_protocol, config.data.train_audio_dir),
        ("dev", config.data.dev_protocol, config.data.dev_audio_dir),
        ("eval", config.data.eval_protocol, config.data.eval_audio_dir),
    ]

    for split_name, proto_file, audio_sub in asv_specs:
        proto_path = proto_dir / proto_file
        audio_dir = data_root / audio_sub

        if not proto_path.exists():
            if not debug:
                logger.warning(
                    "Protocol file not found: %s. Run with --debug or download official data (scripts/download_data.md).",
                    proto_path,
                )
            continue

        raw_df = load_asvspoof_protocol(proto_path, audio_dir=audio_dir, split_name=split_name)
        manifest_df = build_manifest_from_records(
            raw_df,
            dataset_name="asvspoof2019_la",
            split_name=split_name,
            probe_audio=probe_audio,
        )
        save_path = out_path / f"{split_name}.parquet"
        save_manifest(manifest_df, save_path)
        manifests[split_name] = manifest_df

    # 2. In-The-Wild Splits (Speaker-disjoint ITW-CAL and ITW-TEST)
    if itw_meta.exists():
        raw_itw_df = load_in_the_wild_protocol(itw_meta, audio_dir=itw_audio)
        cal_df, test_df = split_in_the_wild_by_speaker(
            raw_itw_df,
            cal_ratio=config.data.itw_cal_ratio,
            seed=config.seed,
        )

        manifest_cal = build_manifest_from_records(
            cal_df,
            dataset_name="in_the_wild",
            split_name="itw_cal",
            probe_audio=probe_audio,
        )
        manifest_test = build_manifest_from_records(
            test_df,
            dataset_name="in_the_wild",
            split_name="itw_test",
            probe_audio=probe_audio,
        )

        save_manifest(manifest_cal, out_path / "itw_cal.parquet")
        save_manifest(manifest_test, out_path / "itw_test.parquet")
        manifests["itw_cal"] = manifest_cal
        manifests["itw_test"] = manifest_test
    else:
        logger.info("In-The-Wild metadata not found at %s. Skipping ITW manifests.", itw_meta)

    # 3. Print Structured Summary Table
    summary_df = summarize_splits(manifests)
    logger.info("\n======================== AF-Split v1 Manifest Summary ========================\n%s\n", summary_df.to_string(index=False))

    # Compare against official ground truth if in production mode
    if not debug and "train" in manifests and "dev" in manifests and "eval" in manifests:
        logger.info("Verifying counts against official ASVspoof 2019 LA benchmark specifications:")
        for sp in ["train", "dev", "eval"]:
            expected = OFFICIAL_COUNTS[f"asvspoof_{sp}"]
            actual_bf = (manifests[sp]["label"] == BONAFIDE).sum()
            actual_sp = (manifests[sp]["label"] == SPOOF).sum()
            actual_tot = len(manifests[sp])
            status = "MATCH" if (actual_bf == expected["bonafide"] and actual_sp == expected["spoof"]) else "DISCREPANCY"
            logger.info("  [%s] %s: Bona fide %d (Exp: %d) | Spoof %d (Exp: %d) | Total %d (Exp: %d)",
                        status, sp.upper(), actual_bf, expected["bonafide"], actual_sp, expected["spoof"], actual_tot, expected["total"])

    return manifests


def main():
    parser = argparse.ArgumentParser(description="Build and save standardized Parquet manifests for Acoustic Forensics.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to default config YAML")
    parser.add_argument("--output-dir", type=str, default="data/manifests", help="Output directory for .parquet manifests")
    parser.add_argument("--debug", action="store_true", help="Use synthetic smoke test data for quick execution")
    parser.add_argument("--probe-audio", action="store_true", help="Probe actual audio files for duration and bitrate")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)
    config = load_config(args.config)

    logger.info("Starting manifest preparation...")
    manifests = prepare_manifests(
        config=config,
        output_dir=args.output_dir,
        debug=args.debug,
        probe_audio=args.probe_audio,
    )
    logger.info("Manifest preparation finished successfully! Generated %d manifests.", len(manifests))


if __name__ == "__main__":
    main()
