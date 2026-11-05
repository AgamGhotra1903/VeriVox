"""End-to-end integration tests for ForensicPredictor on synthetic audio.

Author: Agam Ghotra (IIT2024226)
"""

import pytest
from pathlib import Path
import soundfile as sf
import numpy as np

from acforensics.inference.predictor import ForensicPredictor, ForensicPrediction


def test_predictor_end_to_end_on_synthetic_wav(tmp_path: Path):
    # 1. Create a 3.5s synthetic audio WAV file
    audio_file = tmp_path / "test_sample.wav"
    sr = 16000
    duration = 3.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * 300.0 * t)
    sf.write(str(audio_file), signal, sr)

    # 2. Instantiate ForensicPredictor
    predictor = ForensicPredictor(
        checkpoint_path="checkpoints/crnn_best.pt",
        norm_stats_path="checkpoints/normalization_stats.json",
        calibrator_path="checkpoints/calibrator.joblib",
        device="cpu",
    )

    # 3. Run prediction
    pred = predictor.predict(audio_file)

    assert isinstance(pred, ForensicPrediction)
    assert pred.verdict in ("LIKELY GENUINE", "LIKELY SYNTHETIC", "UNCERTAIN")
    assert 0.0 <= pred.calibrated_probability_bonafide <= 1.0
    assert 0.0 <= pred.confidence_percentage <= 100.0
    assert pred.duration_sec == pytest.approx(duration, abs=0.1)
    assert pred.num_windows_analyzed >= 1
    assert pred.inference_latency_ms > 0.0
    assert len(pred.file_sha256) == 64
    assert len(pred.attention_weights) > 0
    assert len(pred.window_timeline) >= 1

    # Verify standardized forensic JSON output structure
    report = pred.to_dict()
    assert "metadata" in report
    assert "scores" in report
    assert "anomalies" in report
    assert "engine" in report
    assert "duration_sec" in report["metadata"]
    assert "sample_rate" in report["metadata"]
    assert "synthetic_audio_score" in report["scores"]
    assert "tampering_probability" in report["scores"]
    assert isinstance(report["anomalies"]["anomaly_timestamps"], list)


def test_predictor_classical_baseline_inference(tmp_path: Path):
    audio_file = tmp_path / "test_rf.wav"
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    signal = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(str(audio_file), signal, sr)

    predictor = ForensicPredictor(device="cpu")
    pred = predictor.predict(audio_file, model_name="Random Forest (LFCC)", feature_type="lfcc")

    assert isinstance(pred, ForensicPrediction)
    assert pred.verdict in ("LIKELY GENUINE", "LIKELY SYNTHETIC", "UNCERTAIN")
    assert 0.0 <= pred.calibrated_probability_bonafide <= 1.0
    assert "Random Forest" in pred.model_name

