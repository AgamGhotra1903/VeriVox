"""EXPERIMENTAL ABLATION ONLY: Calibrated Stacked Forensic Ensemble.

NOTE: This ensemble is an experimental ablation study in Section 4.5 of the technical report.
The primary standalone system recommended for production and headline reporting is CRNN-LFCC.

Executes the ensemble models on an audio file or raw waveform using the exact feature
representations, scaling, and PCA transformations trained in Stages 3 and 4:
1. Base model continuous scores (higher = more bona fide).
2. Stacked ensemble continuous score via non-negative meta-learner.
3. Baseline mean fusion score.
4. Calibrated confidence probability P(bona fide) via meta-learner sigmoid.

"""

from pathlib import Path
import threading
from typing import Any, Optional, Union
import joblib
import numpy as np
import torch

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    LFCC_DEFAULT_DIM,
    LOGMEL_DEFAULT_DIM,
)
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics
from acforensics.models.gmm import GMMForensics
from acforensics.ensemble.stacking import StackedEnsemble, NonNegativeStackedEnsemble
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


class EnsemblePredictor:
    """Unified predictor running full ensemble inference and meta-learner score fusion."""

    def __init__(
        self,
        crnn_lfcc_path: Path | str = "checkpoints/crnn_lfcc_best.pt",
        crnn_logmel_path: Path | str = "checkpoints/crnn_logmel_best.pt",
        classical_bundle_path: Path | str = "checkpoints/ensemble/classical_bundle.joblib",
        ensemble_path: Path | str = "checkpoints/ensemble/stacked_ensemble.joblib",
        device: Optional[torch.device] = None,
    ):
        if device is None:
            self.device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
        self._lock = threading.Lock()

        self.lfcc_ext = LFCCExtractor(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_lfcc=20,
            with_deltas=True,
        ).to(self.device)

        self.logmel_ext = LogMelExtractor(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80,
        ).to(self.device)

        p_lfcc = Path(crnn_lfcc_path)
        if not p_lfcc.exists():
            p_lfcc = Path("checkpoints/crnn_best.pt")
        ckpt_lfcc = torch.load(str(p_lfcc), map_location=self.device, weights_only=False)
        self.crnn_lfcc = CRNNForensics(
            in_channels=1,
            in_freq_dim=LFCC_DEFAULT_DIM,
            features="lfcc",
        )
        self.crnn_lfcc.load_state_dict(ckpt_lfcc["model_state_dict"] if "model_state_dict" in ckpt_lfcc else ckpt_lfcc)
        self.crnn_lfcc.to(self.device)
        self.crnn_lfcc.eval()

        self.crnn_logmel = None
        p_logmel = Path(crnn_logmel_path)
        if p_logmel.exists():
            try:
                ckpt_logmel = torch.load(str(p_logmel), map_location=self.device, weights_only=False)
                st = ckpt_logmel["model_state_dict"] if "model_state_dict" in ckpt_logmel else ckpt_logmel
                self.crnn_logmel = CRNNForensics(
                    in_channels=1,
                    in_freq_dim=80,
                    features="logmel",
                )
                self.crnn_logmel.load_state_dict(st)
                self.crnn_logmel.to(self.device)
                self.crnn_logmel.eval()
            except Exception as e:
                logger.warning("Could not load optional CRNN-LogMel: %s", e)

        cb_path = Path(classical_bundle_path)
        if cb_path.exists():
            bundle = joblib.load(str(cb_path))
            self.scaler_pooled = bundle["scaler_pooled"]
            self.pca = bundle["pca"]
            self.rf = bundle["rf"]
            self.svm = bundle["svm"]
            self.lr = bundle["lr"]
            self.gmm_bf = bundle["gmm_bf"]
            self.gmm_sp = bundle["gmm_sp"]
        else:
            raise FileNotFoundError(f"Classical bundle not found at {cb_path}")

        p_ens = Path(ensemble_path)
        if p_ens.exists():
            self.stacker = joblib.load(str(p_ens))
            self.base_model_names = self.stacker.model_names
        else:
            raise FileNotFoundError(f"Stacked ensemble not found at {p_ens}")

    def predict_waveform(self, waveform: torch.Tensor) -> dict[str, Any]:
        """Runs the entire ensemble on a raw 1D audio waveform."""
        assert waveform.ndim == 1, f"Expected 1D waveform tensor, got shape {tuple(waveform.shape)}"
        n = waveform.size(0)
        if n >= TARGET_SAMPLES:
            start = (n - TARGET_SAMPLES) // 2
            wave = waveform[start : start + TARGET_SAMPLES]
        else:
            pad = TARGET_SAMPLES - n
            left = pad // 2
            right = pad - left
            wave = torch.nn.functional.pad(waveform, (left, right), mode="constant")

        with self._lock:
            wave = wave.to(device=self.device, dtype=torch.float32)

            with torch.no_grad():
                raw_lfcc = self.lfcc_ext(wave)
                norm_lfcc = per_utterance_cmvn(raw_lfcc).unsqueeze(0).unsqueeze(1).to(device=self.device, dtype=torch.float32)
                logits_lfcc, _ = self.crnn_lfcc(norm_lfcc)
                s_crnn_lfcc = float((logits_lfcc[:, BONAFIDE] - logits_lfcc[:, SPOOF]).item())

                if self.crnn_logmel is not None:
                    raw_logmel = self.logmel_ext(wave)
                    norm_logmel = per_utterance_cmvn(raw_logmel).unsqueeze(0).unsqueeze(1).to(device=self.device, dtype=torch.float32)
                    logits_logmel, _ = self.crnn_logmel(norm_logmel)
                    s_crnn_logmel = float((logits_logmel[:, BONAFIDE] - logits_logmel[:, SPOOF]).item())
                else:
                    s_crnn_logmel = s_crnn_lfcc

                frames_t = raw_lfcc.detach().cpu().numpy().astype(np.float64).T
                mean_v = torch.mean(raw_lfcc, dim=1)
                std_v = torch.std(raw_lfcc, dim=1)
                pooled_vec = torch.cat([mean_v, std_v], dim=0).detach().cpu().numpy().reshape(1, -1)

                if self.device.type == "mps":
                    torch.mps.synchronize()

        ll_bf = np.mean(self.gmm_bf.score_samples(frames_t))
        ll_sp = np.mean(self.gmm_sp.score_samples(frames_t))
        s_gmm = float(ll_bf - ll_sp)

        scaled_pooled = self.scaler_pooled.transform(pooled_vec)
        s_rf = float(self.rf.predict_proba(scaled_pooled)[0, BONAFIDE])

        pca_pooled = self.pca.transform(scaled_pooled)
        s_svm = float(self.svm.predict_proba(pca_pooled)[0, BONAFIDE])
        s_lr = float(self.lr.predict_proba(pca_pooled)[0, BONAFIDE])

        scores_by_name = {
            "CRNN-LFCC": s_crnn_lfcc,
            "CRNN-LogMel": s_crnn_logmel,
            "GMM": s_gmm,
            "SVM": s_svm,
            "Random Forest": s_rf,
            "Logistic Regression": s_lr,
        }

        alias_map = {
            "GMM-LFCC": "GMM",
            "SVM-LFCC": "SVM",
            "RandomForest-LFCC": "Random Forest",
            "LogisticRegression-LFCC": "Logistic Regression",
        }

        scores_vec = np.array([[scores_by_name[alias_map.get(name, name)] for name in self.base_model_names]], dtype=np.float64)

        fused_score = float(self.stacker.predict_score(scores_vec)[0])
        baseline_mean = float(self.stacker.predict_baseline_mean(scores_vec)[0])
        proba_bonafide = float(self.stacker.predict_proba(scores_vec)[0])

        verdict = "BONA FIDE (GENUINE)" if proba_bonafide >= 0.5 else "SPOOF (SYNTHETIC)"

        return {
            "base_scores": scores_by_name,
            "fused_score": fused_score,
            "baseline_mean_score": baseline_mean,
            "probability_bonafide": proba_bonafide,
            "verdict": verdict,
        }

    def predict_file(self, audio_path: Path | str) -> dict[str, Any]:
        """Runs the entire ensemble on an audio file on disk."""
        res = load_and_validate_audio(str(audio_path), target_sr=SAMPLE_RATE, normalization_method="none")
        if not res.is_valid or res.waveform is None:
            raise ValueError(f"Corrupt or invalid audio file: {audio_path}")
        return self.predict_waveform(res.waveform)
