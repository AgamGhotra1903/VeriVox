"""Unified Inference Engine for the Stacked Forensic Ensemble.

Executes all 6 base models (CRNN-LFCC, CRNN-LogMel, GMM, SVM, Random Forest, Logistic Regression)
on an audio file or raw waveform, and applies the frozen meta-learner to yield:
1. Base model continuous scores (higher = more bona fide).
2. Stacked ensemble continuous score.
3. Baseline mean fusion score.
4. Calibrated confidence probability P(bona fide) via sigmoid of the meta-learner output.

Author: Agam Ghotra (IIT2024226)
"""

from pathlib import Path
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
from acforensics.features.normalization import FeatureNormalizer, per_utterance_cmvn
from acforensics.models.classical import (
    extract_mean_std_features,
    predict_classical_score,
)
from acforensics.models.crnn import CRNNForensics
from acforensics.models.gmm import GMMForensics
from acforensics.ensemble.stacking import StackedEnsemble, NonNegativeStackedEnsemble
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


class EnsemblePredictor:
    """Unified predictor running full 6-model inference and meta-learner score fusion."""

    def __init__(
        self,
        crnn_lfcc_path: Path | str = "checkpoints/crnn_lfcc_best.pt",
        crnn_logmel_path: Path | str = "checkpoints/crnn_logmel_best.pt",
        gmm_path: Path | str = "checkpoints/classical/gmm_lfcc_baseline.joblib",
        svm_path: Path | str = "checkpoints/classical/rbf_svm_lfcc.joblib",
        rf_path: Path | str = "checkpoints/classical/random_forest_lfcc.joblib",
        lr_path: Path | str = "checkpoints/classical/logistic_regression_lfcc.joblib",
        ensemble_path: Path | str = "checkpoints/ensemble/stacked_ensemble.joblib",
        device: Optional[torch.device] = None,
    ):
        if device is None:
            self.device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

        # 1. Feature extractors
        self.lfcc_ext = LFCCExtractor(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_lfcc=20,
            with_deltas=True,
        )
        self.logmel_ext = LogMelExtractor(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80,
        )

        self.norm_lfcc = FeatureNormalizer()
        self.norm_logmel = FeatureNormalizer()

        # 2. Base Models
        # CRNN-LFCC
        p_lfcc = Path(crnn_lfcc_path)
        if not p_lfcc.exists():
            p_lfcc = Path("checkpoints/crnn_best.pt")
        ckpt_lfcc = torch.load(str(p_lfcc), map_location=self.device, weights_only=False)
        self.crnn_lfcc = CRNNForensics(
            in_channels=1,
            in_freq_dim=LFCC_DEFAULT_DIM,
            features="lfcc",
        )
        self.crnn_lfcc.load_state_dict(ckpt_lfcc["model_state_dict"])
        self.crnn_lfcc.to(self.device)
        self.crnn_lfcc.eval()

        # CRNN-LogMel
        p_logmel = Path(crnn_logmel_path)
        if not p_logmel.exists():
            p_logmel = Path("checkpoints/crnn_best.pt")
        ckpt_logmel = torch.load(str(p_logmel), map_location=self.device, weights_only=False)
        self.crnn_logmel = CRNNForensics(
            in_channels=1,
            in_freq_dim=LOGMEL_DEFAULT_DIM if "in_freq_dim" not in ckpt_logmel else ckpt_logmel["model_state_dict"]["cnn.b1.block.0.weight"].shape[1],
            features="logmel",
        )
        try:
            self.crnn_logmel.load_state_dict(ckpt_logmel["model_state_dict"])
        except Exception:
            self.crnn_logmel = CRNNForensics(in_channels=1, in_freq_dim=80, features="logmel")
            self.crnn_logmel.load_state_dict(ckpt_logmel["model_state_dict"])
        self.crnn_logmel.to(self.device)
        self.crnn_logmel.eval()

        # GMM
        p_gmm = Path(gmm_path)
        if not p_gmm.exists():
            p_gmm = Path("checkpoints/classical/gmm_lfcc.joblib")
        self.gmm = GMMForensics.load(p_gmm)

        # Classical Sklearn Pipelines
        self.svm = joblib.load(str(svm_path))
        self.rf = joblib.load(str(rf_path))
        self.lr = joblib.load(str(lr_path))

        # Meta-learner Stacked Ensemble
        self.stacker = joblib.load(str(ensemble_path))
        self.base_model_names = self.stacker.model_names

    def predict_waveform(self, waveform: torch.Tensor) -> dict[str, Any]:
        """Runs the entire 6-model ensemble on a raw 1D audio waveform.

        Args:
            waveform: 1D torch.Tensor audio samples at 16 kHz.

        Returns:
            Dictionary containing:
            - base_scores: dict mapping model name to continuous score
            - fused_score: continuous score of stacked ensemble
            - baseline_mean_score: unweighted average of z-scores
            - probability_bonafide: sigmoid confidence in [0, 1]
            - verdict: 'BONA FIDE (GENUINE)' or 'SPOOF (SYNTHETIC)'
        """
        assert waveform.ndim == 1, f"Expected 1D waveform tensor, got shape {tuple(waveform.shape)}"
        wave = fixed_length_crop_or_pad(waveform, target_samples=TARGET_SAMPLES, mode="center")

        with torch.no_grad():
            # 1. Feature representations
            raw_lfcc = self.lfcc_ext(wave)
            raw_lfcc_t = raw_lfcc.T.detach().cpu().numpy().astype(np.float64)

            # Standardized Per-Utterance CMVN for CRNNs
            feat_lfcc = per_utterance_cmvn(raw_lfcc)
            raw_logmel = self.logmel_ext(wave)
            feat_logmel = per_utterance_cmvn(raw_logmel)

            # 2. Base model evaluations
            s_crnn_lfcc = self.crnn_lfcc.predict_score(feat_lfcc.to(self.device))
            s_crnn_logmel = self.crnn_logmel.predict_score(feat_logmel.to(self.device))
            s_gmm = self.gmm.predict_score(raw_lfcc_t)

            mean_std = extract_mean_std_features(raw_lfcc_t)
            s_svm = predict_classical_score(self.svm, mean_std)[0]
            s_rf = predict_classical_score(self.rf, mean_std)[0]
            s_lr = predict_classical_score(self.lr, mean_std)[0]

        scores_by_name = {
            "CRNN-LFCC": float(s_crnn_lfcc),
            "CRNN-LogMel": float(s_crnn_logmel),
            "GMM": float(s_gmm),
            "SVM": float(s_svm),
            "Random Forest": float(s_rf),
            "Logistic Regression": float(s_lr),
        }

        alias_map = {
            "GMM-LFCC": "GMM",
            "SVM-LFCC": "SVM",
            "RandomForest-LFCC": "Random Forest",
            "LogisticRegression-LFCC": "Logistic Regression",
        }
        scores_vec = np.array([[scores_by_name[alias_map.get(name, name)] for name in self.base_model_names]], dtype=np.float64)

        # 3. Meta-learner predictions
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
        """Runs the entire 6-model ensemble on an audio file on disk."""
        res = load_and_validate_audio(str(audio_path), target_sr=SAMPLE_RATE)
        if not res.is_valid or res.waveform is None:
            raise ValueError(f"Corrupt or invalid audio file: {audio_path}")
        return self.predict_waveform(res.waveform)
