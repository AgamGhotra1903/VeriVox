"""Integration and unit tests for the Acoustic Forensics FastAPI service.

Verifies:
- Health check & sample catalog endpoints
- File validation (rejects invalid extension, rejects empty/corrupt files)
- Server-Sent Events (SSE) emission in strict pipeline order:
  validating -> resampling -> feature_extraction -> model_scores -> ensemble -> done
- Complete result retrieval (verdict, probabilities, attention, waveform peaks, downsampled spectrograms)
- Codec stress-testing endpoint with ffmpeg transcoding matrix
- Dynamic benchmark loader and missing file fallback behavior

"""

import io
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

from acforensics.api.main import app
from acforensics.constants import SAMPLE_RATE


@pytest.fixture(scope="module")
def client():
    """Provides FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_wav_bytes():
    """Generates synthetic 2-second 16 kHz sine wave for testing."""
    t = np.linspace(0, 2.0, int(SAMPLE_RATE * 2.0), endpoint=False)
    wav = 0.4 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()


def test_health_endpoint(client):
    """GET /api/health returns 200 with operational device and models."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "device" in data
    assert len(data["models_loaded"]) == 7
    assert data["memory_rss_mb"] > 0


def test_samples_endpoint(client):
    """GET /api/samples returns reference bundled files."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 3
    names = [s["filename"] for s in samples]
    assert "sample_bonafide_human_speech.wav" in names
    assert "sample_spoof_neural_tts.wav" in names
    assert "sample_spoof_vocoder_vc.wav" in names


def test_stream_sample_audio(client):
    """GET /api/samples/{filename} streams audio file."""
    response = client.get("/api/samples/sample_bonafide_human_speech.wav")
    assert response.status_code == 200
    assert response.headers["content-type"] in ["audio/wav", "audio/x-wav"]
    assert len(response.content) > 10000

    resp_404 = client.get("/api/samples/non_existent.wav")
    assert resp_404.status_code == 404


def test_upload_validation_rejects_invalid_extension(client):
    """POST /api/analyze rejects unsupported extensions."""
    files = {"audio": ("malicious.exe", b"invalid binary content", "application/octet-stream")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]


def test_upload_validation_rejects_empty_file(client):
    """POST /api/analyze rejects too-small or empty files."""
    files = {"audio": ("empty.wav", b"too small", "audio/wav")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "too small" in response.json()["detail"]


def test_upload_and_stream_events_order(client, sample_wav_bytes):
    """POST /api/analyze and GET /api/analyze/{id}/events emits stages in strict order."""
    files = {"audio": ("test_clip.wav", sample_wav_bytes, "audio/wav")}
    resp_upload = client.post("/api/analyze", files=files)
    assert resp_upload.status_code == 200
    upload_data = resp_upload.json()
    job_id = upload_data["job_id"]
    assert job_id is not None
    assert upload_data["status"] == "processing"

    resp_stream = client.get(f"/api/analyze/{job_id}/events")
    assert resp_stream.status_code == 200
    assert "text/event-stream" in resp_stream.headers["content-type"]

    events = []
    for line in resp_stream.text.strip().split("\n"):
        if line.startswith("data: "):
            ev_data = json.loads(line[len("data: ") :])
            events.append(ev_data)

    stages = [ev["stage"] for ev in events]
    assert "validating" in stages
    assert "resampling" in stages
    assert "feature_extraction" in stages
    assert "model_scores" in stages
    assert "ensemble" in stages
    assert "done" in stages

    idx_val = stages.index("validating")
    idx_res = stages.index("resampling")
    idx_feat = stages.index("feature_extraction")
    idx_mod = stages.index("model_scores")
    idx_ens = stages.index("ensemble")
    idx_done = stages.index("done")

    assert idx_val < idx_res < idx_feat < idx_mod < idx_ens < idx_done

    models_emitted = [ev["model"] for ev in events if ev["stage"] == "model_scores"]
    assert "CRNN-LFCC" in models_emitted
    assert "CRNN-LogMel" in models_emitted
    assert "GMM" in models_emitted
    assert "SVM" in models_emitted
    assert "Random Forest" in models_emitted
    assert "Logistic Regression" in models_emitted


def test_get_result_endpoint(client, sample_wav_bytes):
    """GET /api/result/{job_id} returns full forensic report and downsampled matrices."""
    files = {"audio": ("test_result.wav", sample_wav_bytes, "audio/wav")}
    job_id = client.post("/api/analyze", files=files).json()["job_id"]

    response = client.get(f"/api/result/{job_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["job_id"] == job_id
    assert data["verdict"] in ["BONA FIDE (GENUINE)", "SPOOF (SYNTHETIC)", "INCONCLUSIVE"]
    assert 0.0 <= data["probability_bonafide"] <= 1.0
    assert "decision_threshold" in data
    assert len(data["models"]) == 6
    assert len(data["waveform_peaks"]) > 100
    assert len(data["attention_weights"]) > 10
    assert len(data["lfcc_downsampled"]) == 60
    assert len(data["logmel_downsampled"]) == 80
    assert len(data["time_axis_sec"]) == len(data["lfcc_downsampled"][0])
    assert data["quality"]["sample_rate"] == 16000
    assert data["quality"]["duration_sec"] > 0


def test_stress_test_endpoint(client, sample_wav_bytes):
    """POST /api/stress-test re-encodes across MP3, AAC, and Opus."""
    files = {"audio": ("test_stress.wav", sample_wav_bytes, "audio/wav")}
    job_id = client.post("/api/analyze", files=files).json()["job_id"]

    resp_stress = client.post("/api/stress-test", data={"job_id": job_id})
    assert resp_stress.status_code == 200
    data = resp_stress.json()
    assert data["job_id"] == job_id
    assert len(data["items"]) == 9

    codecs = {item["codec"] for item in data["items"]}
    assert "MP3" in codecs
    assert "AAC" in codecs
    assert "OPUS" in codecs


def test_benchmarks_endpoint(client):
    """GET /api/benchmarks returns empirical evaluation results dynamically."""
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    bench = response.json()

    assert bench["status"] == "available"
    assert len(bench["master_table"]) >= 8
    systems = [r["system"] for r in bench["master_table"]]
    assert "Single: CRNN-LFCC" in systems
    assert "Stacked Ensemble (Proposed)" in systems

    assert len(bench["meta_learner_weights"]) >= 6
    assert len(bench["feature_ablation"]) >= 3
    assert len(bench["architecture_ablation"]) >= 3
    assert len(bench["regularization_ablation"]) >= 2
    assert len(bench["per_attack_eer"]) >= 13
    assert len(bench["codec_robustness"]) == 9
    assert len(bench["crossdomain_itw"]) >= 1
    assert "Stacked Ensemble" in bench["roc_curves"]

