"""Unit tests for UI helper functions, quality checks, verdict logic, and report generation.

Author: Chunit Bansal (IIT2024233)
"""

import numpy as np
import pytest

from acforensics.features.quality import (
    compute_clipping_ratio,
    estimate_snr_db,
    compute_spectral_bandwidth_hz,
    compute_silence_ratio,
    extract_audio_quality_profile,
    evaluate_reliability_checks,
)
from acforensics.inference.predictor import Prediction
from app.theme import get_verdict_presentation


def test_clipping_ratio_clean_and_clipped():
    clean_signal = np.sin(np.linspace(0, 10, 1000)) * 0.5
    assert compute_clipping_ratio(clean_signal) == 0.0

    clipped_signal = np.clip(np.sin(np.linspace(0, 10, 1000)) * 2.0, -1.0, 1.0)
    assert compute_clipping_ratio(clipped_signal) > 0.10


def test_estimate_snr_db():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    envelope = np.zeros_like(t)
    envelope[int(0.2 * sr) : int(0.8 * sr)] = 1.0
    speech_signal = 0.5 * np.sin(2 * np.pi * 440 * t) * envelope + 0.001 * np.random.normal(0, 1, len(t))
    snr_clean = estimate_snr_db(speech_signal)

    noisy = speech_signal + 0.1 * np.random.normal(0, 1, len(t))
    snr_noisy = estimate_snr_db(noisy)

    assert snr_clean > snr_noisy
    assert snr_clean >= 20.0


def test_compute_spectral_bandwidth():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    # Low frequency tone: 300 Hz
    low_tone = np.sin(2 * np.pi * 300 * t)
    bw_low = compute_spectral_bandwidth_hz(low_tone, sr=sr)
    assert bw_low < 2000.0

    # High frequency wideband noise
    white_noise = np.random.normal(0, 1, sr)
    bw_high = compute_spectral_bandwidth_hz(white_noise, sr=sr)
    assert bw_high > 5000.0


def test_verdict_presentation_and_color_independence():
    # Never rely on color alone: each must have an icon and a distinct title
    res_gen = get_verdict_presentation("genuine")
    assert res_gen["icon"] == "✔"
    assert "GENUINE" in res_gen["title"]

    res_syn = get_verdict_presentation("synthetic")
    assert res_syn["icon"] == "✖"
    assert "SYNTHETIC" in res_syn["title"]

    res_inc = get_verdict_presentation("inconclusive")
    assert res_inc["icon"] == "?"
    assert "INCONCLUSIVE" in res_inc["title"]


def test_evaluate_reliability_checks_triggers_warnings():
    # 1. Short clip triggers red warning
    poor_quality = {
        "duration": 0.8,
        "clipping": 0.08,
        "snr_db": 8.0,
        "bandwidth_hz": 3200,
        "silence_ratio": 0.75,
        "orig_sr": 8000,
    }
    checks = evaluate_reliability_checks(poor_quality)
    statuses = [c["status"] for c in checks]
    assert "red" in statuses

    # 2. Pristine clip gets all green
    good_quality = {
        "duration": 4.0,
        "clipping": 0.0,
        "snr_db": 35.0,
        "bandwidth_hz": 7500,
        "silence_ratio": 0.15,
        "orig_sr": 16000,
    }
    checks_good = evaluate_reliability_checks(good_quality)
    statuses_good = [c["status"] for c in checks_good]
    assert all(s == "green" for s in statuses_good)


def test_prediction_report_serialization():
    pred = Prediction(
        prob_genuine=0.88,
        verdict="genuine",
        threshold=0.50,
        window_scores=[0.85, 0.91],
        window_times=[(0.0, 2.0), (1.0, 3.0)],
        latency_ms=15.4,
        model_version="CRNN-v1.0",
        file_hash="abc123hash",
    )
    d = pred.to_dict()
    assert d["verdict"] == "genuine"
    assert d["prob_genuine"] == 0.88
    assert d["prob_synthetic"] == 0.12
    assert "explanation" in d
    assert len(d["window_analysis"]) == 2
