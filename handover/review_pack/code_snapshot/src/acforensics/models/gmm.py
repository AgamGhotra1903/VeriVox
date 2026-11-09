"""Gaussian Mixture Model (GMM) Baseline for Acoustic Forensics.

Mirrors the official ASVspoof 2019 baseline:
- Two 64-component diagonal-covariance GMMs:
  1. Lambda_bonafide fitted on genuine speech LFCC frames.
  2. Lambda_spoof fitted on synthetic/spoof speech LFCC frames.
- Scoring function: frame-level Log-Likelihood Ratio (LLR), averaged over frames:
  LLR(X) = (1 / T) * sum_{t=1}^T [ log p(x_t | Lambda_bonafide) - log p(x_t | Lambda_spoof) ]
- High score indicates authentic human speech; low score indicates spoofed speech.

Author: Agam Ghotra (IIT2024226), Jashan Goyal (IIT2024223)
"""

from pathlib import Path
from typing import Any, Optional, Union
import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


class GMMForensics:
    """ASVspoof 2019 Official GMM Baseline Implementation."""

    def __init__(
        self,
        n_components: int = 64,
        covariance_type: str = "diag",
        max_iter: int = 100,
        reg_covar: float = 1e-3,
        random_state: int = 42,
        max_frames_per_class: int = 250000,
    ):
        """Initializes bona fide and spoof GMM models.

        Args:
            n_components: Number of Gaussian mixture components (default: 64).
            covariance_type: Covariance matrix constraint ('diag' as in official baseline).
            max_iter: Maximum EM iterations.
            reg_covar: Non-negative regularization added to the diagonal of covariance matrices.
            random_state: RNG seed for reproducible initialization.
            max_frames_per_class: Maximum number of frames to subsample for EM training.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.max_frames_per_class = max_frames_per_class

        self.gmm_bonafide = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        self.gmm_spoof = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(
        self,
        bonafide_or_utterances: Union[np.ndarray, list[np.ndarray]],
        spoof_or_labels: Union[np.ndarray, list[int]],
    ) -> "GMMForensics":
        """Fits separate GMMs on bona fide and spoof spectral frames.

        Supports two calling conventions:
        1. Two separate 2D frame arrays: fit(bonafide_frames_2d, spoof_frames_2d)
        2. Per-utterance frames list + labels: fit(utterances_frames_list, labels)

        Args:
            bonafide_or_utterances: Either 2D array of bona fide frames, or list of utterance frame arrays.
            spoof_or_labels: Either 2D array of spoof frames, or 1D array/list of binary labels (1=bonafide, 0=spoof).

        Returns:
            self
        """
        is_utterance_list = isinstance(bonafide_or_utterances, list) or (
            hasattr(spoof_or_labels, "ndim") and spoof_or_labels.ndim == 1
        ) or (isinstance(spoof_or_labels, list) and not isinstance(spoof_or_labels[0], (list, np.ndarray)))

        if is_utterance_list:
            labels = np.asarray(spoof_or_labels)
            utterances = bonafide_or_utterances
            assert len(utterances) == len(labels), "Length mismatch between utterances and labels"

            def align_frames(arr: np.ndarray) -> np.ndarray:
                arr = np.asarray(arr, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[0] in (20, 60, 80) and arr.shape[1] > arr.shape[0]:
                    return arr.T
                return arr

            bf_list = [align_frames(u) for u, l in zip(utterances, labels) if l == BONAFIDE and len(u) > 0]
            sp_list = [align_frames(u) for u, l in zip(utterances, labels) if l == SPOOF and len(u) > 0]

            assert len(bf_list) > 0, "No bona fide frames available for GMM fitting"
            assert len(sp_list) > 0, "No spoof frames available for GMM fitting"

            bonafide_frames = np.vstack(bf_list)
            spoof_frames = np.vstack(sp_list)
        else:
            bonafide_frames = np.asarray(bonafide_or_utterances, dtype=np.float64)
            spoof_frames = np.asarray(spoof_or_labels, dtype=np.float64)

        assert bonafide_frames.ndim == 2, f"Expected 2D array (N, F), got {bonafide_frames.shape}"
        assert spoof_frames.ndim == 2, f"Expected 2D array (N, F), got {spoof_frames.shape}"
        assert bonafide_frames.shape[1] == spoof_frames.shape[1], (
            f"Feature dimension mismatch: {bonafide_frames.shape[1]} vs {spoof_frames.shape[1]}"
        )

        bf_clean = np.asarray(bonafide_frames, dtype=np.float64)
        sp_clean = np.asarray(spoof_frames, dtype=np.float64)

        rng = np.random.RandomState(self.random_state)

        # Frame subsampling if pool exceeds memory/time thresholds
        if len(bf_clean) > self.max_frames_per_class:
            logger.info(
                "Subsampling bona fide frames from %d to %d for GMM EM training",
                len(bf_clean), self.max_frames_per_class,
            )
            idx_bf = rng.choice(len(bf_clean), size=self.max_frames_per_class, replace=False)
            bonafide_train = bf_clean[idx_bf]
        else:
            bonafide_train = bf_clean

        if len(sp_clean) > self.max_frames_per_class:
            logger.info(
                "Subsampling spoof frames from %d to %d for GMM EM training",
                len(sp_clean), self.max_frames_per_class,
            )
            idx_sp = rng.choice(len(sp_clean), size=self.max_frames_per_class, replace=False)
            spoof_train = sp_clean[idx_sp]
        else:
            spoof_train = sp_clean

        # Handle small smoke test sample edge cases (n_components cannot exceed n_samples)
        n_bf = len(bonafide_train)
        n_sp = len(spoof_train)
        effective_k = min(self.n_components, n_bf, n_sp)
        if effective_k < self.n_components:
            logger.warning("Reducing GMM components from %d to %d due to small sample size", self.n_components, effective_k)
            self.gmm_bonafide.set_params(n_components=effective_k)
            self.gmm_spoof.set_params(n_components=effective_k)

        logger.info("Fitting GMM_bonafide on %d frames (K=%d)...", len(bonafide_train), self.gmm_bonafide.n_components)
        self.gmm_bonafide.fit(bonafide_train)

        logger.info("Fitting GMM_spoof on %d frames (K=%d)...", len(spoof_train), self.gmm_spoof.n_components)
        self.gmm_spoof.fit(spoof_train)

        self.is_fitted = True
        return self

    def score_utterance(self, frames: np.ndarray) -> float:
        """Computes frame-averaged Log-Likelihood Ratio (LLR) for a single utterance.

        Args:
            frames: 2D array of shape (T, F_dim) or (F_dim, T).

        Returns:
            Scalar float score where higher value indicates bona fide (genuine).
        """
        assert self.is_fitted, "GMMForensics must be fitted before scoring."
        if frames.ndim != 2:
            raise ValueError(f"Expected 2D frames matrix, got shape {frames.shape}")

        # Ensure shape is (T, F_dim)
        if frames.shape[0] < frames.shape[1] and frames.shape[0] in (20, 60, 80):
            frames = frames.T

        frames_clean = np.asarray(frames, dtype=np.float64)
        T, F_dim = frames_clean.shape
        if T == 0:
            return 0.0

        # score_samples computes per-frame log-likelihood: shape (T,)
        ll_bf = self.gmm_bonafide.score_samples(frames_clean)
        ll_sp = self.gmm_spoof.score_samples(frames_clean)

        llr = float(np.mean(ll_bf - ll_sp))
        return llr

    def predict_score(self, utterances_frames: Union[list[np.ndarray], np.ndarray]) -> np.ndarray:
        """Scores one utterance or batch of utterances.

        Args:
            utterances_frames: Either a single 2D array (T, F) / (F, T),
                               or a list/array of 2D frame feature arrays.

        Returns:
            1D array of LLR scores.
        """
        if isinstance(utterances_frames, np.ndarray) and utterances_frames.ndim == 2:
            return np.array([self.score_utterance(utterances_frames)], dtype=np.float64)

        return np.array([self.score_utterance(frames) for frames in utterances_frames], dtype=np.float64)

    def save(self, output_path: Path | str) -> None:
        """Serializes fitted GMM models to disk via joblib."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(out))
        logger.info("Saved GMM baseline model to %s", out)

    @classmethod
    def load(cls, model_path: Path | str) -> "GMMForensics":
        """Loads serialized GMMForensics model from disk."""
        obj = joblib.load(str(model_path))
        if isinstance(obj, dict) and "gmm_bf" in obj and "gmm_sp" in obj:
            instance = cls(n_components=getattr(obj["gmm_bf"], "n_components", 64))
            instance.gmm_bonafide = obj["gmm_bf"]
            instance.gmm_spoof = obj["gmm_sp"]
            instance.is_fitted = True
            return instance
        return obj
