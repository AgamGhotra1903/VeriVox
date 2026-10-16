"""Data preparation, validation, and synthetic smoke-test dataset generator.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

import argparse
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from acforensics.config import load_config
from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    KNOWN_ATTACKS,
    UNSEEN_ATTACKS,
)
from acforensics.data.protocols import (
    load_asvspoof_protocol,
    load_in_the_wild_protocol,
    get_protocol_statistics,
)
from acforensics.data.datasets import ASVspoofDataset, collate_fn
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def generate_synthetic_audio(
    file_path: Path,
    duration_sec: float = 3.5,
    sample_rate: int = SAMPLE_RATE,
    is_spoof: bool = False,
) -> None:
    """Generates synthetic audio signal simulating human speech or vocoder artifacts.

    Args:
        file_path: Target output path (.flac or .wav).
        duration_sec: Duration in seconds.
        sample_rate: Sampling frequency in Hz.
        is_spoof: If True, adds phase jitter and high-frequency harmonics typical of vocoders.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)

    # Fundamental voice frequency (120 Hz to 240 Hz)
    f0 = random.uniform(120.0, 220.0)

    # Harmonic series simulating vowel formants
    signal = 0.5 * np.sin(2 * np.pi * f0 * t)
    signal += 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    signal += 0.10 * np.sin(2 * np.pi * 4 * f0 * t)

    # Low-frequency amplitude envelope simulating speech syllables
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    signal *= envelope

    if is_spoof:
        # Inject vocoder buzz and synthetic phase distortion
        buzz = 0.08 * np.sin(2 * np.pi * 3800.0 * t + np.random.uniform(0, np.pi))
        signal += buzz
        # Slight spectral roughness
        signal += 0.02 * np.random.normal(0, 1, size=len(t)).astype(np.float32)
    else:
        # Natural subtle room noise
        signal += 0.005 * np.random.normal(0, 1, size=len(t)).astype(np.float32)

    # Peak normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.95

    sf.write(str(file_path), signal, sample_rate, subtype="PCM_16", format="FLAC")


def generate_smoke_test_dataset(root_dir: Path | str) -> dict[str, Path]:
    """Generates a complete, lightweight mock dataset matching the ASVspoof 2019 layout.

    Allows all downstream training, baselines, and evaluation scripts to run end-to-end
    in minutes without requiring the full 35 GB ASVspoof download.

    Args:
        root_dir: Destination directory (e.g. 'data/smoke_test').

    Returns:
        Dictionary mapping partition names to protocol file paths.
    """
    root = Path(root_dir)
    protocols_dir = root / "ASVspoof2019_LA_cm_protocols"
    protocols_dir.mkdir(parents=True, exist_ok=True)
    asv_scores_dir = root / "ASVspoof2019_LA_asv_scores"
    asv_scores_dir.mkdir(parents=True, exist_ok=True)

    splits_spec = {
        "train": (200, KNOWN_ATTACKS, "ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train/flac"),
        "dev": (100, KNOWN_ATTACKS, "ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev/flac"),
        "eval": (100, UNSEEN_ATTACKS, "ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval/flac"),
    }

    protocol_paths = {}

    for split_name, (n_samples, attacks_pool, proto_filename, audio_subfolder) in splits_spec.items():
        audio_dir = root / audio_subfolder
        audio_dir.mkdir(parents=True, exist_ok=True)
        proto_path = protocols_dir / proto_filename
        protocol_paths[split_name] = proto_path

        asv_score_lines = []
        with open(proto_path, "w", encoding="utf-8") as pf:
            for i in range(n_samples):
                speaker_id = f"LA_{split_name[:2].upper()}_{1000 + (i % 20):04d}"
                file_id = f"LA_{split_name[0].upper()}_{1000000 + i}"
                audio_file = audio_dir / f"{file_id}.flac"

                # Imbalance: ~20% bona fide, ~80% spoof (matching ASVspoof distribution)
                is_bonafide = (i % 5 == 0)
                if is_bonafide:
                    key = "bonafide"
                    attack_id = "-"
                    is_spoof = False
                else:
                    key = "spoof"
                    attack_id = random.choice(attacks_pool)
                    is_spoof = True

                # Generate FLAC file
                duration = random.uniform(2.5, 4.5)
                generate_synthetic_audio(audio_file, duration_sec=duration, is_spoof=is_spoof)

                # Write protocol entry: SPEAKER_ID FILE_ID - ATTACK_ID KEY
                pf.write(f"{speaker_id} {file_id} - {attack_id} {key}\n")

                # Mock ASV score line for min-tDCF calculation
                # Format: SPEAKER_ID FILE_ID ASV_SCORE KEY
                mock_asv_score = random.gauss(2.5 if is_bonafide else -1.5, 1.0)
                asv_score_lines.append(f"{speaker_id} {file_id} target {mock_asv_score:.4f} {key}\n")

        # Save ASV score files for dev and eval
        if split_name in ("dev", "eval"):
            asv_score_file = asv_scores_dir / f"ASVspoof2019.LA.asv.{split_name}.gi.trl.scores.txt"
            with open(asv_score_file, "w", encoding="utf-8") as asv_f:
                asv_f.writelines(asv_score_lines)

    # Also generate mock In-The-Wild dataset
    itw_dir = root / "in_the_wild"
    itw_audio_dir = itw_dir / "audio"
    itw_audio_dir.mkdir(parents=True, exist_ok=True)
    meta_path = itw_dir / "meta.csv"

    with open(meta_path, "w", encoding="utf-8") as mf:
        mf.write("file,speaker,label\n")
        for i in range(50):
            file_name = f"itw_clip_{i:04d}.wav"
            speaker = f"public_speaker_{i % 5}"
            is_bf = (i % 3 == 0)
            label = "bona-fide" if is_bf else "spoof"
            audio_path = itw_audio_dir / file_name
            generate_synthetic_audio(audio_path, duration_sec=random.uniform(3.0, 5.0), is_spoof=not is_bf)
            mf.write(f"{file_name},{speaker},{label}\n")

    logger.info("Successfully generated mock smoke-test dataset at: %s", root)
    return protocol_paths


def print_protocol_summary(split_name: str, stats: dict) -> None:
    """Pretty prints protocol statistics table."""
    logger.info("--------------------------------------------------")
    logger.info(" Split: %s", split_name.upper())
    logger.info(" Total Utterances:   %d", stats["total"])
    logger.info(" Bona fide Count:    %d (%.2f%%)", stats["bonafide_count"], stats["bonafide_ratio"] * 100)
    logger.info(" Spoof Count:        %d (%.2f%%)", stats["spoof_count"], stats["spoof_ratio"] * 100)
    logger.info(" Unique Speakers:    %d", stats["unique_speakers"])
    logger.info(" Attack Breakdown:   %s", stats.get("attack_counts", {}))
    logger.info("--------------------------------------------------")


def verify_batch_loading(config, protocol_path: Path, audio_dir: Path) -> None:
    """Instantiates ASVspoofDataset, loads a batch through DataLoader, and verifies tensor dimensions."""
    logger.info("Verifying DataLoader batch loading with protocol: %s", protocol_path)
    df = load_asvspoof_protocol(protocol_path, audio_dir=audio_dir, split_name="train")
    dataset = ASVspoofDataset(
        protocol_df=df,
        target_samples=config.audio.target_length_samples,
        split="train",
        crop_mode="random",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    batch = next(iter(dataloader))
    waveforms = batch["waveform"]
    labels = batch["label"]

    logger.info("Batch successfully loaded!")
    logger.info("  Waveform shape: %s (Expected: (8, %d))", tuple(waveforms.shape), config.audio.target_length_samples)
    logger.info("  Labels shape:   %s (Values: %s)", tuple(labels.shape), labels.tolist())
    logger.info("  File IDs:       %s", batch["file_id"][:4])
    logger.info("  Attack IDs:     %s", batch["attack_id"][:4])


def main():
    parser = argparse.ArgumentParser(description="Acoustic Forensics Data Preparation & Validation CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--generate-smoke-data", action="store_true", help="Generate synthetic mock data")
    parser.add_argument("--smoke-dir", type=str, default="data/smoke_test", help="Directory for mock data")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing protocols")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)
    config = load_config(args.config)

    if args.generate_smoke_data:
        smoke_dir = Path(args.smoke_dir)
        protocol_paths = generate_smoke_test_dataset(smoke_dir)

        # Inspect and print summaries
        for split_name in ["train", "dev", "eval"]:
            p_path = protocol_paths[split_name]
            a_dir = smoke_dir / f"ASVspoof2019_LA_{split_name}" / "flac"
            df = load_asvspoof_protocol(p_path, audio_dir=a_dir, split_name=split_name)
            stats = get_protocol_statistics(df)
            print_protocol_summary(split_name, stats)

        # Verify batch loading
        verify_batch_loading(
            config,
            protocol_paths["train"],
            smoke_dir / "ASVspoof2019_LA_train" / "flac",
        )
        return

    # Check for real ASVspoof 2019 dataset
    asv_root = Path(config.data.asvspoof_root)
    proto_dir = Path(config.data.protocols_dir)

    if not proto_dir.exists():
        logger.warning(
            "Protocol directory not found at: %s.\n"
            "To download the official dataset, see: scripts/download_data.md\n"
            "To generate a synthetic smoke test dataset for instant verification, run:\n"
            "  python scripts/prepare_data.py --generate-smoke-data",
            proto_dir,
        )
        return

    for split_name, proto_file, audio_sub in [
        ("train", config.data.train_protocol, config.data.train_audio_dir),
        ("dev", config.data.dev_protocol, config.data.dev_audio_dir),
        ("eval", config.data.eval_protocol, config.data.eval_audio_dir),
    ]:
        p_path = proto_dir / proto_file
        a_dir = asv_root / audio_sub
        if p_path.exists():
            df = load_asvspoof_protocol(p_path, audio_dir=a_dir, split_name=split_name)
            stats = get_protocol_statistics(df)
            print_protocol_summary(split_name, stats)
        else:
            logger.warning("Split file %s does not exist.", p_path)


if __name__ == "__main__":
    main()
