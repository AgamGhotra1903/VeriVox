"""Robust audio input/output, validation, resampling, and normalization.

Author: Paras Bansal (IIT2024221)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import csv
import numpy as np
import soundfile as sf
import torch
import torchaudio

from acforensics.constants import (
    SAMPLE_RATE,
    MIN_DURATION_SEC,
    MIN_SAMPLES,
    DEFAULT_LOGS_DIR,
)
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AudioValidationResult:
    """Result container for audio loading and verification."""
    is_valid: bool
    waveform: Optional[torch.Tensor] = None  # 1D tensor shape (samples,)
    sample_rate: int = SAMPLE_RATE
    duration_sec: float = 0.0
    error_reason: Optional[str] = None
    orig_sample_rate: int = SAMPLE_RATE
    channels: int = 1
    bit_depth: str = "16-bit"
    bitrate_kbps: Optional[float] = None
    peak_level_db: float = 0.0
    snr_estimate_db: float = 30.0


def log_rejected_file(
    file_path: str | Path,
    reason: str,
    duration_sec: float = 0.0,
    log_csv_path: Optional[Path | str] = None,
) -> None:
    """Appends an entry to rejected_files.csv for error auditing.

    Args:
        file_path: Path of the audio file rejected.
        reason: Plain-text explanation of failure (e.g. 'too_short', 'nan_detected').
        duration_sec: Measured audio duration prior to rejection.
        log_csv_path: Path to rejected files CSV log.
    """
    csv_path = Path(log_csv_path) if log_csv_path else DEFAULT_LOGS_DIR / "rejected_files.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "file_path", "reason", "duration_sec"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            str(file_path),
            reason,
            f"{duration_sec:.4f}",
        ])


def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Resamples 1D audio waveform to target sample rate using torchaudio.

    Args:
        waveform: 1D torch.Tensor of shape (samples,).
        orig_sr: Source sampling rate.
        target_sr: Target sampling rate (default: 16000).

    Returns:
        Resampled 1D torch.Tensor.
    """
    if orig_sr == target_sr:
        return waveform

    # torchaudio.transforms.Resample expects shape (channels, samples)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler(waveform.unsqueeze(0)).squeeze(0)
    return resampled


def normalize_audio(
    waveform: torch.Tensor,
    method: str = "peak",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalizes audio waveform amplitude.

    Args:
        waveform: 1D torch.Tensor.
        method: Normalization technique:
            'peak' -> scales to [-1.0, 1.0] guarding against divide-by-zero.
            'rms'  -> scales root-mean-square to target energy (e.g. 0.1).
            'none' -> leaves waveform unchanged.
        eps: Small constant to avoid zero-division.

    Returns:
        Normalized 1D torch.Tensor.
    """
    if method == "peak":
        max_val = torch.max(torch.abs(waveform))
        if max_val > eps:
            return waveform / max_val
        return waveform
    elif method == "rms":
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > eps:
            target_rms = 0.1
            return waveform * (target_rms / (rms + eps))
        return waveform
    elif method == "none":
        return waveform
    else:
        logger.warning("Unknown normalization method '%s'; defaulting to peak.", method)
        max_val = torch.max(torch.abs(waveform))
        if max_val > eps:
            return waveform / max_val
        return waveform


def load_and_validate_audio(
    file_path: Path | str,
    target_sr: int = SAMPLE_RATE,
    min_duration_sec: float = MIN_DURATION_SEC,
    normalization_method: str = "peak",
    log_rejections: bool = True,
    rejected_log_path: Optional[Path | str] = None,
) -> AudioValidationResult:
    """Loads, validates, converts to mono, resamples, and normalizes an audio file.

    Performs comprehensive validation:
    1. File existence and readability check.
    2. Empty file rejection (0 bytes or 0 samples).
    3. Multichannel to mono conversion (channel averaging).
    4. Resampling to 16 kHz.
    5. Detection of NaN or infinite floating-point values.
    6. Minimum duration enforcement (>= 0.5s).
    7. Peak normalization with zero-division guard.

    Args:
        file_path: Path to FLAC, WAV, MP3, OGG, or M4A file.
        target_sr: Desired sampling rate (default: 16000 Hz).
        min_duration_sec: Minimum acceptable duration (default: 0.5s).
        normalization_method: 'peak', 'rms', or 'none'.
        log_rejections: Whether to write failed validations to rejected_files.csv.
        rejected_log_path: Optional custom path for rejected_files.csv.

    Returns:
        AudioValidationResult dataclass with validity status and waveform.
    """
    path = Path(file_path)

    # 1. Existence check
    if not path.exists():
        if log_rejections:
            log_rejected_file(path, "file_not_found", 0.0, rejected_log_path)
        return AudioValidationResult(
            is_valid=False,
            error_reason=f"File not found: {path}",
        )

    # 2. File read attempt (soundfile first, fallback to torchaudio, fallback to ffmpeg)
    bit_depth_str = "16-bit PCM"
    bitrate_kbps_val = None
    orig_channels = 1

    try:
        info = sf.info(str(path))
        orig_channels = info.channels
        if info.subtype:
            bit_depth_str = info.subtype
        if hasattr(info, "samplerate") and hasattr(info, "frames") and info.duration > 0:
            file_bytes = path.stat().st_size
            bitrate_kbps_val = round((file_bytes * 8) / (info.duration * 1000.0), 1)
    except Exception:
        pass

    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception as sf_err:
        try:
            waveform_torch, sr = torchaudio.load(str(path))
            data = waveform_torch.numpy().T
        except Exception as ta_err:
            try:
                import io
                import subprocess
                cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "wav", "-acodec", "pcm_s16le", "-"]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
            except Exception as ffmpeg_err:
                if log_rejections:
                    log_rejected_file(path, f"unreadable_codec_{sf_err}", 0.0, rejected_log_path)
                return AudioValidationResult(
                    is_valid=False,
                    error_reason=f"Corrupt or unreadable audio format: {sf_err} | {ta_err} | {ffmpeg_err}",
                )

    # Convert to 1D numpy array
    if data.ndim == 2:
        orig_channels = data.shape[1]
        # Multichannel: average channels down to mono
        data = np.mean(data, axis=1)
    elif data.ndim > 2:
        if log_rejections:
            log_rejected_file(path, "invalid_channel_dimensions", 0.0, rejected_log_path)
        return AudioValidationResult(
            is_valid=False,
            error_reason=f"Invalid channel shape: {data.shape}",
        )

    # 3. Empty file check
    if len(data) == 0:
        if log_rejections:
            log_rejected_file(path, "empty_audio_file", 0.0, rejected_log_path)
        return AudioValidationResult(
            is_valid=False,
            error_reason="Audio file contains 0 samples.",
        )

    # 4. NaN or Inf check
    if np.isnan(data).any() or np.isinf(data).any():
        if log_rejections:
            log_rejected_file(path, "nan_or_inf_detected", len(data) / sr, rejected_log_path)
        return AudioValidationResult(
            is_valid=False,
            duration_sec=len(data) / sr,
            error_reason="Audio contains NaN or Infinite amplitude values.",
        )

    # Compute physical acoustic properties before normalization
    peak_val = float(np.max(np.abs(data)))
    peak_level_db = round(20.0 * np.log10(max(peak_val, 1e-6)), 2)

    # SNR Estimation heuristic (frame RMS 90th percentile vs 10th percentile floor)
    frame_sz = min(1024, max(64, len(data) // 50))
    n_f = len(data) // frame_sz
    if n_f >= 4:
        f_rms = np.sqrt(np.mean(data[: n_f * frame_sz].reshape(n_f, frame_sz) ** 2, axis=1))
        f_rms = f_rms[f_rms > 1e-7]
        if len(f_rms) >= 4:
            p90 = np.percentile(f_rms, 90)
            p10 = np.percentile(f_rms, 10)
            snr_val = round(float(20.0 * np.log10((p90 + 1e-7) / (p10 + 1e-7))), 1)
        else:
            snr_val = 30.0
    else:
        snr_val = 30.0

    # Convert to torch tensor
    waveform = torch.from_numpy(data.astype(np.float32))

    # 5. Resample if necessary
    orig_sr = sr
    if sr != target_sr:
        try:
            waveform = resample_audio(waveform, orig_sr=sr, target_sr=target_sr)
        except Exception as resample_err:
            if log_rejections:
                log_rejected_file(path, f"resampling_failed_{resample_err}", len(data) / sr, rejected_log_path)
            return AudioValidationResult(
                is_valid=False,
                duration_sec=len(data) / sr,
                error_reason=f"Resampling failed: {resample_err}",
            )

    duration_sec = len(waveform) / target_sr

    # 6. Minimum duration check
    if duration_sec < min_duration_sec:
        if log_rejections:
            log_rejected_file(path, "duration_under_minimum", duration_sec, rejected_log_path)
        return AudioValidationResult(
            is_valid=False,
            duration_sec=duration_sec,
            error_reason=f"Duration ({duration_sec:.2f}s) is shorter than minimum allowed ({min_duration_sec:.2f}s).",
        )

    # 7. Normalization
    waveform = normalize_audio(waveform, method=normalization_method)

    return AudioValidationResult(
        is_valid=True,
        waveform=waveform,
        sample_rate=target_sr,
        duration_sec=duration_sec,
        error_reason=None,
        orig_sample_rate=orig_sr,
        channels=orig_channels,
        bit_depth=bit_depth_str,
        bitrate_kbps=bitrate_kbps_val,
        peak_level_db=peak_level_db,
        snr_estimate_db=snr_val,
    )
