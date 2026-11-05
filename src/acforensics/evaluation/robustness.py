from pathlib import Path
from typing import Any, Optional
import subprocess
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
import torch

from acforensics.constants import SAMPLE_RATE, DEFAULT_RESULTS_DIR
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)

def check_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None

def align_codec_delay(clean: np.ndarray, decoded: np.ndarray, max_search_samples: int = 4000) -> np.ndarray:
    if len(clean) == 0 or len(decoded) == 0:
        return decoded
    n = min(len(clean), len(decoded), 16000)
    c_sub = clean[:n]
    d_sub = decoded[:n]
    corr = scipy.signal.correlate(d_sub, c_sub, mode="full")
    lags = scipy.signal.correlation_lags(n, n, mode="full")
    search_mask = np.abs(lags) <= max_search_samples
    if not np.any(search_mask):
        return decoded
    best_lag = lags[search_mask][np.argmax(corr[search_mask])]
    if best_lag > 0 and best_lag < len(decoded):
        return decoded[best_lag:]
    elif best_lag < 0:
        pad = np.zeros(abs(best_lag), dtype=decoded.dtype)
        return np.concatenate([pad, decoded])
    return decoded

def compress_and_decompress_audio(
    input_wav: Path | str,
    output_wav: Path | str,
    codec: str = "mp3",
    bitrate_kbps: int = 32,
    target_sr: int = SAMPLE_RATE,
    compensate_delay: bool = True,
) -> bool:
    if not check_ffmpeg_available():
        return False
    in_path = Path(input_wav)
    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    codec_ext_map = {"mp3": ".mp3", "aac": ".m4a", "opus": ".opus"}
    ext = codec_ext_map.get(codec.lower(), ".mp3")
    temp_comp = out_path.with_suffix(ext)
    encoder_map = {"mp3": "libmp3lame", "aac": "aac", "opus": "libopus"}
    encoder = encoder_map.get(codec.lower(), "libmp3lame")
    try:
        cmd_enc = [
            "ffmpeg", "-y", "-nostdin", "-nostats", "-hide_banner",
            "-protocol_whitelist", "file,pipe",
            "-v", "error",
            "-i", str(in_path),
            "-c:a", encoder,
            "-b:a", f"{bitrate_kbps}k",
            str(temp_comp),
        ]
        subprocess.run(cmd_enc, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15.0)
        cmd_dec = [
            "ffmpeg", "-y", "-nostdin", "-nostats", "-hide_banner",
            "-protocol_whitelist", "file,pipe",
            "-v", "error",
            "-i", str(temp_comp),
            "-ar", str(target_sr),
            "-ac", "1",
            str(out_path),
        ]
        subprocess.run(cmd_dec, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15.0)
        if temp_comp.exists():
            temp_comp.unlink()
        if compensate_delay:
            c_sig, _ = sf.read(str(in_path))
            d_sig, sr = sf.read(str(out_path))
            aligned = align_codec_delay(c_sig, d_sig)
            sf.write(str(out_path), aligned, sr)
        return True
    except Exception:
        if temp_comp.exists():
            temp_comp.unlink()
        return False

def apply_telephony_channel(input_wav: Path | str, output_wav: Path | str, compand: str = "alaw") -> bool:
    if not check_ffmpeg_available():
        return False
    in_path = Path(input_wav)
    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_8k = out_path.with_suffix(".8k.wav")
    codec_pcm = "pcm_alaw" if compand.lower() == "alaw" else "pcm_mulaw"
    try:
        cmd1 = [
            "ffmpeg", "-y", "-nostdin", "-nostats", "-hide_banner",
            "-protocol_whitelist", "file,pipe",
            "-v", "error",
            "-i", str(in_path),
            "-ar", "8000",
            "-c:a", codec_pcm,
            str(temp_8k),
        ]
        subprocess.run(cmd1, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15.0)
        cmd2 = [
            "ffmpeg", "-y", "-nostdin", "-nostats", "-hide_banner",
            "-protocol_whitelist", "file,pipe",
            "-v", "error",
            "-i", str(temp_8k),
            "-ar", "16000",
            "-ac", "1",
            str(out_path),
        ]
        subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15.0)
        if temp_8k.exists():
            temp_8k.unlink()
        return True
    except Exception:
        if temp_8k.exists():
            temp_8k.unlink()
        return False

def apply_additive_noise(sig: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    p_sig = np.mean(sig ** 2)
    if p_sig <= 1e-12:
        return sig
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, np.sqrt(p_noise), size=sig.shape).astype(sig.dtype)
    return sig + noise

def apply_synthetic_reverb(sig: np.ndarray, rt60: float = 0.3, sr: int = 16000, seed: int = 42) -> np.ndarray:
    t_ir = max(0.05, min(1.0, rt60))
    n_ir = int(t_ir * sr)
    t = np.linspace(0, t_ir, n_ir, endpoint=False)
    decay_rate = 3.0 * np.log(10.0) / (rt60 + 1e-6)
    env = np.exp(-decay_rate * t)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, 1.0, size=n_ir)
    ir = (env * noise).astype(np.float32)
    ir = ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9)
    rev = scipy.signal.fftconvolve(sig, ir, mode="full")[: len(sig)]
    alpha = 0.35
    dry_wet = (1.0 - alpha) * sig + alpha * rev
    return dry_wet.astype(sig.dtype)

def apply_packet_loss(sig: np.ndarray, loss_rate: float = 0.10, packet_ms: int = 20, sr: int = 16000, seed: int = 42) -> np.ndarray:
    pkt_len = int((packet_ms / 1000.0) * sr)
    n_pkts = len(sig) // pkt_len
    if n_pkts < 2:
        return sig
    out = sig.copy()
    rng = np.random.RandomState(seed)
    drops = rng.rand(n_pkts) < loss_rate
    for i, drop in enumerate(drops):
        if drop:
            out[i * pkt_len : (i + 1) * pkt_len] = 0.0
    return out

def plot_robustness_curves(
    results_dict: dict[str, dict[int, float]],
    clean_eer: float,
    output_path: Path | str,
) -> None:
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
    ax.axhline(clean_eer, color="black", linestyle="--", lw=1.5, label=f"Clean Baseline ({clean_eer:.2f}%)")
    ax.set_xlabel("Bitrate (kbps)", fontsize=11)
    ax.set_ylabel("Equal Error Rate (EER %)", fontsize=11)
    ax.set_title("Codec Robustness Degradation (ASVspoof 2019 Eval)", fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out))
    plt.savefig(str(out.with_suffix(".svg")))
    plt.close()
