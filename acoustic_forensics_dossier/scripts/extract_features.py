"""Feature extraction CLI, normalization fitter, and spectrogram visualizer.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from acforensics.config import load_config
from acforensics.constants import (
    SAMPLE_RATE,
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_RESULTS_DIR,
    BONAFIDE,
    SPOOF,
)
from acforensics.data.protocols import load_asvspoof_protocol
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.normalization import FeatureNormalizer
from acforensics.features.statistics import extract_utterance_statistics
from acforensics.features.cache import FeatureCache
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def plot_spectrogram_comparison(
    bf_wave: torch.Tensor,
    sp_wave: torch.Tensor,
    lfcc_extractor: LFCCExtractor,
    logmel_extractor: LogMelExtractor,
    output_dir: Path,
) -> None:
    """Generates comparison visualizations between bona fide and spoof speech."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        bf_lfcc = lfcc_extractor(bf_wave).cpu().numpy()
        sp_lfcc = lfcc_extractor(sp_wave).cpu().numpy()
        bf_logmel = logmel_extractor(bf_wave).cpu().numpy()
        sp_logmel = logmel_extractor(sp_wave).cpu().numpy()

    # 1. LFCC Heatmap Comparison
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    im0 = axes[0].imshow(bf_lfcc, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Bona Fide: LFCC (60 Coefficients with Deltas)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Coefficients")
    fig.colorbar(im0, ax=axes[0], format="%.1f")

    im1 = axes[1].imshow(sp_lfcc, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Spoof / Synthetic: LFCC (60 Coefficients with Deltas)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time Frames (~10ms hop)")
    axes[1].set_ylabel("Coefficients")
    fig.colorbar(im1, ax=axes[1], format="%.1f")
    plt.tight_layout()
    lfcc_plot_path = output_dir / "lfcc_comparison.png"
    plt.savefig(str(lfcc_plot_path), dpi=200)
    plt.close()
    logger.info("Saved LFCC comparison figure to %s", lfcc_plot_path)

    # 2. Log-Mel Spectrogram Comparison
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    im0 = axes[0].imshow(bf_logmel, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Bona Fide: Log-Mel Spectrogram (80 Mel Bands)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mel Frequency Bands")
    fig.colorbar(im0, ax=axes[0], format="%.1f dB")

    im1 = axes[1].imshow(sp_logmel, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("Spoof / Synthetic: Log-Mel Spectrogram (80 Mel Bands)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time Frames (~10ms hop)")
    axes[1].set_ylabel("Mel Frequency Bands")
    fig.colorbar(im1, ax=axes[1], format="%.1f dB")
    plt.tight_layout()
    logmel_plot_path = output_dir / "logmel_comparison.png"
    plt.savefig(str(logmel_plot_path), dpi=200)
    plt.close()
    logger.info("Saved Log-Mel comparison figure to %s", logmel_plot_path)


def main():
    parser = argparse.ArgumentParser(description="Feature Extraction & Normalization Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run on smoke test dataset")
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate spectrogram figures")
    parser.add_argument("--max-utterances", type=int, default=None, help="Limit number of utterances for debugging")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)
    config = load_config(args.config)

    # Determine data roots
    if args.smoke:
        base_dir = Path("data/smoke_test")
        proto_path = base_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
        audio_dir = base_dir / "ASVspoof2019_LA_train" / "flac"
    else:
        base_dir = Path(config.data.asvspoof_root)
        proto_path = Path(config.data.protocols_dir) / config.data.train_protocol
        audio_dir = base_dir / config.data.train_audio_dir
        if not proto_path.exists():
            logger.warning(
                "Training protocol not found at %s. Falling back to smoke test data at data/smoke_test.",
                proto_path,
            )
            base_dir = Path("data/smoke_test")
            proto_path = base_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
            audio_dir = base_dir / "ASVspoof2019_LA_train" / "flac"

    if not proto_path.exists():
        raise FileNotFoundError(f"Neither production nor smoke test protocol exists at {proto_path}. Run make smoke first.")

    train_df = load_asvspoof_protocol(proto_path, audio_dir=audio_dir, split_name="train")
    if args.max_utterances is not None:
        train_df = train_df.head(args.max_utterances)

    # Initialize feature extractors
    lfcc_ext = LFCCExtractor(
        sample_rate=config.audio.sample_rate,
        n_fft=config.features.lfcc.n_fft,
        win_length=config.features.lfcc.win_length,
        hop_length=config.features.lfcc.hop_length,
        n_filter=config.features.lfcc.n_filter,
        n_lfcc=config.features.lfcc.n_lfcc,
        with_deltas=config.features.lfcc.with_deltas,
    )
    logmel_ext = LogMelExtractor(
        sample_rate=config.audio.sample_rate,
        n_fft=config.features.logmel.n_fft,
        win_length=config.features.logmel.win_length,
        hop_length=config.features.logmel.hop_length,
        n_mels=config.features.logmel.n_mels,
        f_min=config.features.logmel.f_min,
        f_max=config.features.logmel.f_max,
        power=config.features.logmel.power,
        top_db=config.features.logmel.top_db,
    )

    logger.info("Extracting features from %d training utterances to compute global normalization stats...", len(train_df))
    extracted_features = []
    bf_wave = None
    sp_wave = None

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting Train Features"):
        audio_path = row["audio_path"]
        res = load_and_validate_audio(audio_path, target_sr=config.audio.sample_rate)
        if not res.is_valid or res.waveform is None:
            continue

        wave = fixed_length_crop_or_pad(res.waveform, target_samples=config.audio.target_length_samples, mode="center")

        if row["label"] == BONAFIDE and bf_wave is None:
            bf_wave = wave
        elif row["label"] == SPOOF and sp_wave is None:
            sp_wave = wave

        with torch.no_grad():
            feat = lfcc_ext(wave) if config.features.feature_type == "lfcc" else logmel_ext(wave)
            extracted_features.append(feat)

    # Fit train-only global normalization statistics
    normalizer = FeatureNormalizer()
    normalizer.fit_from_tensors(extracted_features)
    stats_out = Path(config.features.stats_file)
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    normalizer.save(stats_out)
    logger.info("Successfully persisted train-only normalization statistics to: %s", stats_out)

    # Optional visualizations
    if args.visualize and bf_wave is not None and sp_wave is not None:
        figures_dir = Path(config.paths.figures_dir)
        plot_spectrogram_comparison(bf_wave, sp_wave, lfcc_ext, logmel_ext, figures_dir)


if __name__ == "__main__":
    main()
