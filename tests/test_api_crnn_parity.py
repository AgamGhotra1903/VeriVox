from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from acforensics.constants import SAMPLE_RATE, TARGET_SAMPLES, BONAFIDE, SPOOF
from acforensics.api.main import app
from acforensics.api.service import get_forensic_service
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import per_utterance_cmvn


def test_api_health_endpoint():
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["primary_system"] == "CRNN-LFCC"
    assert "crnn_lfcc_best.pt" in data["primary_checkpoint"]
    assert data["primary_checkpoint_sha256"] != "unknown"
    assert data["ensemble_status"] == "EXPERIMENTAL ABLATION"


def test_api_crnn_score_parity_50_files():
    eval_parquet = Path("data/manifests/eval.parquet")
    if not eval_parquet.exists():
        pytest.skip("Eval manifest not found for API parity test.")

    df = pd.read_parquet(eval_parquet)
    n_files = min(50, len(df))
    subset_df = df.iloc[:n_files]

    service = get_forensic_service()
    device = service.device
    client = TestClient(app)

    extractor = LFCCExtractor(sample_rate=SAMPLE_RATE).to(device)
    model = service.predictor.crnn_lfcc

    diffs = []
    for _, row in subset_df.iterrows():
        audio_path = row["path"]
        res = load_and_validate_audio(audio_path, target_sr=SAMPLE_RATE, normalization_method="none")
        assert res.is_valid and res.waveform is not None

        wave_tensor = res.waveform
        n_samples = wave_tensor.size(0)
        if n_samples >= TARGET_SAMPLES:
            start_i = (n_samples - TARGET_SAMPLES) // 2
            aligned_wave = wave_tensor[start_i : start_i + TARGET_SAMPLES]
        else:
            p_pad = TARGET_SAMPLES - n_samples
            l_pad = p_pad // 2
            r_pad = p_pad - l_pad
            aligned_wave = torch.nn.functional.pad(wave_tensor, (l_pad, r_pad), mode="constant")
        aligned_wave = aligned_wave.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            raw_lfcc = extractor(aligned_wave)
            norm_lfcc = per_utterance_cmvn(raw_lfcc).unsqueeze(0).unsqueeze(1).to(device=device, dtype=torch.float32)
            logits, _ = model.forward(norm_lfcc)
            score_offline = float((logits[:, BONAFIDE] - logits[:, SPOOF]).item())

        with open(audio_path, "rb") as f:
            post_resp = client.post(
                "/api/analyze",
                files={"audio": (Path(audio_path).name, f, "audio/wav")},
            )
        assert post_resp.status_code == 200
        job_id = post_resp.json()["job_id"]

        get_resp = client.get(f"/api/result/{job_id}")
        assert get_resp.status_code == 200
        result_json = get_resp.json()

        score_api = result_json["crnn_score"]
        assert result_json["primary_system"] == "CRNN-LFCC"
        assert result_json["ensemble_status"] == "EXPERIMENTAL ABLATION"
        assert result_json["primary_checkpoint_sha256"] == service.primary_checkpoint_sha256

        diff = abs(score_api - score_offline)
        diffs.append(diff)

    max_diff = max(diffs)
    assert max_diff < 1e-4, f"API CRNN parity violated: max diff = {max_diff:.2e} >= 1e-4 across {n_files} files"
