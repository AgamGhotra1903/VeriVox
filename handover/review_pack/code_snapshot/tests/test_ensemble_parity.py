from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from acforensics.constants import SAMPLE_RATE, TARGET_SAMPLES, BONAFIDE, SPOOF
from acforensics.ensemble.predictor import EnsemblePredictor
from acforensics.features.normalization import per_utterance_cmvn


@pytest.mark.slow
def test_ensemble_raw_waveform_vs_offline_matrix_parity():
    eval_parquet = Path("data/manifests/eval.parquet")
    eval_npz_path = Path("data/cache/lfcc_eval.npz")
    bundle_path = Path("checkpoints/ensemble/classical_bundle.joblib")
    ensemble_path = Path("checkpoints/ensemble/stacked_ensemble.joblib")

    if not (eval_parquet.exists() and eval_npz_path.exists() and bundle_path.exists() and ensemble_path.exists()):
        pytest.skip("Required manifests, caches, or checkpoints missing for parity test.")

    df = pd.read_parquet(eval_parquet)
    bundle = joblib.load(bundle_path)
    stacker = joblib.load(ensemble_path)

    n_test = min(200, len(df))
    ev_npz = np.load(eval_npz_path)
    features_slice = ev_npz["features"][:n_test]
    pooled_slice = ev_npz["pooled"][:n_test]
    paths_slice = df["path"].iloc[:n_test].tolist()

    device = torch.device("cpu")
    predictor = EnsemblePredictor(device=device)

    max_diff = 0.0

    for i in range(n_test):
        feat_np = features_slice[i]
        pooled_np = pooled_slice[i:i+1]

        with torch.no_grad():
            t_feat = per_utterance_cmvn(torch.from_numpy(feat_np).to(dtype=torch.float32)).unsqueeze(0).unsqueeze(1)
            crnn_log, _ = predictor.crnn_lfcc(t_feat)
            s_crnn_offline = float((crnn_log[:, BONAFIDE] - crnn_log[:, SPOOF]).item())

        t_f = feat_np.astype(np.float64).T
        s_gmm_offline = float(np.mean(bundle["gmm_bf"].score_samples(t_f)) - np.mean(bundle["gmm_sp"].score_samples(t_f)))

        p_scaled = bundle["scaler_pooled"].transform(pooled_np)
        s_rf_offline = float(bundle["rf"].predict_proba(p_scaled)[0, BONAFIDE])

        p_pca = bundle["pca"].transform(p_scaled)
        s_svm_offline = float(bundle["svm"].predict_proba(p_pca)[0, BONAFIDE])
        s_lr_offline = float(bundle["lr"].predict_proba(p_pca)[0, BONAFIDE])

        mat_offline = np.array([[s_crnn_offline, s_gmm_offline, s_rf_offline, s_svm_offline, s_lr_offline]], dtype=np.float64)
        score_offline = float(stacker.predict_score(mat_offline)[0])

        audio_path = paths_slice[i]
        res = predictor.predict_file(audio_path)
        score_wave = res["fused_score"]

        diff = abs(score_wave - score_offline)
        if diff > max_diff:
            max_diff = diff

    assert max_diff < 1e-4, f"Parity mismatch: max difference {max_diff:.2e} >= 1e-4 across {n_test} trials!"
