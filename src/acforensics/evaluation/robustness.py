"""Codec compression stress-testing suite (MP3, AAC, Opus) via ffmpeg.

Author: Chunit Bansal (IIT2024223)
"""

from pathlib import Path
from typing import Any, Optional
import subprocess
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from acforensics.constants import (
    SAMPLE_RATE,
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
)
from acforensics.evaluation.metrics import full_evaluation
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def check_ffmpeg_available() -> bool:
    """Verifies that the system ffmpeg binary is accessible."""
    return shutil.which("ffmpeg") is not None


def compress_and_decompress_audio(
    input_wav: Path | str,
    output_wav: Path | str,
    codec: str = "mp3",  # 'mp3', 'aac', 'opus'
    bitrate_kbps: int = 32,
    target_sr: int = SAMPLE_RATE,
) -> bool:
    """Encodes an audio clip using lossy codec and decodes back to 16 kHz mono WAV.

    Exact ffmpeg pipelines:
    - MP3:  ffmpeg -y -i in.wav -c:a libmp3lame -b:a {kbps}k temp.mp3 && ffmpeg -y -i temp.mp3 -ar 16000 -ac 1 out.wav
    - AAC:  ffmpeg -y -i in.wav -c:a aac -b:a {kbps}k temp.m4a && ffmpeg -y -i temp.m4a -ar 16000 -ac 1 out.wav
    - Opus: ffmpeg -y -i in.wav -c:a libopus -b:a {kbps}k temp.opus && ffmpeg -y -i temp.opus -ar 16000 -ac 1 out.wav

    Args:
        input_wav: Path to source audio.
        output_wav: Path to destination decoded WAV.
        codec: 'mp3', 'aac', or 'opus'.
        bitrate_kbps: Bitrate in kilobits per second (e.g. 16, 24, 32, 48, 64).
        target_sr: Target sample rate for decoded file (default: 16000).

    Returns:
        True if re-encoding succeeded, False otherwise.
    """
    if not check_ffmpeg_available():
        logger.warning("ffmpeg binary not found on system PATH.")
        return False

    in_path = Path(input_wav)
    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    codec_ext_map = {"mp3": ".mp3", "aac": ".m4a", "opus": ".opus"}
    ext = codec_ext_map.get(codec.lower(), ".mp3")
    temp_compressed = out_path.with_suffix(ext)

    encoder_map = {
        "mp3": "libmp3lame",
        "aac": "aac",
        "opus": "libopus",
    }
    encoder = encoder_map.get(codec.lower(), "libmp3lame")

    try:
        # Step 1: Encode to lossy format
        cmd_encode = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(in_path),
            "-c:a", encoder,
            "-b:a", f"{bitrate_kbps}k",
            str(temp_compressed),
        ]
        subprocess.run(cmd_encode, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        # Step 2: Decode back to 16 kHz mono WAV
        cmd_decode = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(temp_compressed),
            "-ar", str(target_sr),
            "-ac", "1",
            str(out_path),
        ]
        subprocess.run(cmd_decode, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        # Cleanup intermediate compressed file
        if temp_compressed.exists():
            temp_compressed.unlink()
        return True

    except Exception as e:
        logger.warning("ffmpeg codec simulation failed for %s (%s %dkbps): %s", in_path.name, codec, bitrate_kbps, e)
        if temp_compressed.exists():
            temp_compressed.unlink()
        return False


def plot_robustness_curves(
    results_dict: dict[str, dict[int, float]],
    clean_eer: float,
    output_path: Path | str,
) -> None:
    """Plots EER vs Bitrate across MP3, AAC, and Opus lossy compression codecs."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    codec_colors = {"mp3": "#EF4444", "aac": "#3B82F6", "opus": "#10B981"}
    codec_markers = {"mp3": "o", "aac": "s", "opus": "^"}

    for codec, bitrate_data in results_dict.items():
        bitrates = sorted(bitrate_data.keys())
        eers = [bitrate_data[b] for b in bitrates]
        ax.plot(
            bitrates,
            eers,
            label=f"{codec.upper()}",
            color=codec_colors.get(codec.lower(), "#2DD4BF"),
            marker=codec_markers.get(codec.lower(), "o"),
            lw=2,
            ms=6,
        )

    # Plot baseline clean uncompressed line
    ax.axhline(clean_eer, color="black", linestyle="--", lw=1.5, label=f"Clean Baseline ({clean_eer:.2f}%)")

    ax.set_xlabel("Bitrate (kbps)", fontsize=11)
    ax.set_ylabel("Equal Error Rate (EER %)", fontsize=11)
    ax.set_title("Lossy Codec Robustness Stress Test (ASVspoof 2019 Eval)", fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out))
    plt.savefig(str(out.with_suffix(".svg")))
    plt.close()
