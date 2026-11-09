"""Acoustic input-quality descriptors and reliability audit module.

Measures deterministic audio properties for error analysis and the UI reliability panel:
- duration
- clipping ratio
- estimated SNR (percentile-based)
- spectral rolloff / effective bandwidth (Hz)
- silence ratio
- original sample rate

Author: Chunit Bansal (IIT2024223), Paras Bansal (IIT2024221)
"""

from typing import Any
import numpy as np
import scipy.signal

from acforensics.constants import SAMPLE_RATE


def compute_clipping_ratio(waveform: np.ndarray, threshold: float = 0.995) -> float:
    """Computes fraction of samples that are saturated or clipped."""
    if len(waveform) == 0:
        return 0.0
    clipped = np.sum(np.abs(waveform) >= threshold)
    return float(clipped / len(waveform))


def estimate_snr_db(waveform: np.ndarray, frame_size: int = 400) -> float:
    """Estimates signal-to-noise ratio in dB using 90th vs 10th percentile frame energy levels."""
    if len(waveform) < frame_size:
        return 30.0
    n_frames = len(waveform) // frame_size
    frames = waveform[: n_frames * frame_size].reshape(n_frames, frame_size)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    p90 = np.percentile(frame_rms, 90)
    p10 = np.percentile(frame_rms, 10)
    if p10 < 1e-5:
        p10 = 1e-5
    if p90 < 1e-5:
        return 0.0

    snr = 20.0 * np.log10(p90 / p10)
    return float(np.clip(snr, 0.0, 50.0))


def compute_spectral_bandwidth_hz(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    rolloff_percent: float = 0.85,
) -> float:
    """Computes spectral rolloff frequency (effective audio bandwidth in Hz).

    Audio with low bandwidth (< 4,000 Hz) indicates telephone narrowband,
    aggressive MP3/Opus compression, or heavy lowpass filtering.
    """
    if len(waveform) < 512:
        return float(sr / 2.0)

    # Compute FFT power spectrum
    freqs, psd = scipy.signal.welch(waveform, fs=sr, nperseg=min(512, len(waveform)))
    cum_energy = np.cumsum(psd)
    total_energy = cum_energy[-1]
    if total_energy <= 1e-12:
        return 0.0

    threshold = rolloff_percent * total_energy
    idx = np.where(cum_energy >= threshold)[0]
    if len(idx) > 0:
        return float(freqs[idx[0]])
    return float(freqs[-1])


def compute_silence_ratio(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: float = 25.0,
    silence_threshold_db: float = -42.0,
) -> float:
    """Computes fraction of audio duration consisting of background silence."""
    frame_length = int(sr * (frame_ms / 1000.0))
    if len(waveform) < frame_length:
        return 0.0

    # Frame-wise RMS energy
    n_frames = len(waveform) // frame_length
    frames = waveform[: n_frames * frame_length].reshape(n_frames, frame_length)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms / (np.max(rms) + 1e-9))

    silent_frames = np.sum(rms_db < silence_threshold_db)
    return float(silent_frames / n_frames)


def extract_audio_quality_profile(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    orig_sr: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """Computes complete deterministic audio quality profile.

    Returns dict with keys:
    duration, clipping, snr_db, bandwidth_hz, silence_ratio, orig_sr
    """
    duration = float(len(waveform) / sr)
    clipping = compute_clipping_ratio(waveform)
    snr_db = estimate_snr_db(waveform)
    bandwidth_hz = compute_spectral_bandwidth_hz(waveform, sr=sr)
    silence_ratio = compute_silence_ratio(waveform, sr=sr)

    return {
        "duration": round(duration, 2),
        "clipping": round(clipping, 4),
        "snr_db": round(snr_db, 1),
        "bandwidth_hz": int(bandwidth_hz),
        "silence_ratio": round(silence_ratio, 3),
        "orig_sr": int(orig_sr),
    }


def evaluate_reliability_checks(
    quality: dict[str, Any],
    per_model: dict[str, float] | None = None,
) -> list[dict[str, str]]:
    """Evaluates deterministic reliability criteria with traffic-light indicators.

    Returns list of check objects with keys:
    ['label', 'value_str', 'status', 'consequence']
    status: 'green' | 'amber' | 'red'
    """
    checks = []

    # 1. Duration check
    dur = quality.get("duration", 0.0)
    if dur < 1.0:
        checks.append({
            "label": "Clip Duration",
            "value_str": f"{dur:.1f}s (Severely short)",
            "status": "red",
            "consequence": "Clips under 1.0s lack sufficient phonetic context; false alarm rate rises sharply.",
        })
    elif dur < 2.0:
        checks.append({
            "label": "Clip Duration",
            "value_str": f"{dur:.1f}s (Short clip)",
            "status": "amber",
            "consequence": "Clips between 1.0s and 2.0s provide reduced temporal feature evidence.",
        })
    else:
        checks.append({
            "label": "Clip Duration",
            "value_str": f"{dur:.1f}s (Adequate)",
            "status": "green",
            "consequence": "Adequate duration (> 2.0s) for robust multi-frame feature extraction.",
        })

    # 2. Clipping check
    clip_ratio = quality.get("clipping", 0.0)
    if clip_ratio > 0.05:
        checks.append({
            "label": "Signal Clipping",
            "value_str": f"{clip_ratio * 100:.1f}% saturated",
            "status": "red",
            "consequence": "Harmonic distortion from clipping mimics vocoder high-frequency artifacts.",
        })
    elif clip_ratio > 0.01:
        checks.append({
            "label": "Signal Clipping",
            "value_str": f"{clip_ratio * 100:.1f}% saturated",
            "status": "amber",
            "consequence": "Mild saturation detected; inspect waveform peaks.",
        })
    else:
        checks.append({
            "label": "Signal Clipping",
            "value_str": "Clean (< 1% clipped)",
            "status": "green",
            "consequence": "No harmful analog or digital clipping detected.",
        })

    # 3. Estimated SNR check
    snr = quality.get("snr_db", 30.0)
    if snr < 12.0:
        checks.append({
            "label": "Signal-to-Noise Ratio",
            "value_str": f"{snr:.1f} dB (Noisy)",
            "status": "red",
            "consequence": "Heavy background noise obscures delicate acoustic cues.",
        })
    elif snr < 18.0:
        checks.append({
            "label": "Signal-to-Noise Ratio",
            "value_str": f"{snr:.1f} dB (Moderate noise)",
            "status": "amber",
            "consequence": "Moderate noise present; confidence should be interpreted with care.",
        })
    else:
        checks.append({
            "label": "Signal-to-Noise Ratio",
            "value_str": f"{snr:.1f} dB (High clarity)",
            "status": "green",
            "consequence": "Speech signal is clear with high dynamic range.",
        })

    # 4. Bandwidth / Rolloff check
    bw = quality.get("bandwidth_hz", 8000)
    if bw < 4200:
        checks.append({
            "label": "Effective Bandwidth",
            "value_str": f"{bw} Hz (Narrowband / Compressed)",
            "status": "red",
            "consequence": "High frequencies (>4 kHz) are missing; heavy compression or telephone channel detected.",
        })
    elif bw < 6000:
        checks.append({
            "label": "Effective Bandwidth",
            "value_str": f"{bw} Hz (Sub-8kHz cutoff)",
            "status": "amber",
            "consequence": "Bandwidth is mildly restricted; may indicate medium-bitrate MP3/Opus compression.",
        })
    else:
        checks.append({
            "label": "Effective Bandwidth",
            "value_str": f"{bw} Hz (Full wideband)",
            "status": "green",
            "consequence": "Full 8 kHz Nyquist spectrum available for forensic examination.",
        })

    # 5. Silence Ratio check
    sil = quality.get("silence_ratio", 0.0)
    if sil > 0.65:
        checks.append({
            "label": "Silence Ratio",
            "value_str": f"{sil * 100:.0f}% inactive frames",
            "status": "amber",
            "consequence": "Majority of clip contains background silence rather than active speech.",
        })
    else:
        checks.append({
            "label": "Silence Ratio",
            "value_str": f"{sil * 100:.0f}% inactive frames",
            "status": "green",
            "consequence": "Active speech occupies the majority of the recording.",
        })

    # 6. Model Agreement check (if available)
    if per_model and len(per_model) >= 2:
        probs = list(per_model.values())
        std_p = float(np.std(probs))
        if std_p > 0.30:
            checks.append({
                "label": "Model Ensemble Agreement",
                "value_str": f"High Disagreement (σ = {std_p:.2f})",
                "status": "red",
                "consequence": "Different neural and classical architectures reached opposing conclusions.",
            })
        elif std_p > 0.15:
            checks.append({
                "label": "Model Ensemble Agreement",
                "value_str": f"Moderate Variance (σ = {std_p:.2f})",
                "status": "amber",
                "consequence": "Minor disagreement between spectral and cepstral branches.",
            })
        else:
            checks.append({
                "label": "Model Ensemble Agreement",
                "value_str": f"Strong Consensus (σ = {std_p:.2f})",
                "status": "green",
                "consequence": "All evaluation models concur on the probability estimate.",
            })

    return checks
