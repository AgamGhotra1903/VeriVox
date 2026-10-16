"""Probability calibration via Platt Scaling fitted exclusively on validation (dev) scores.

Author: Agam Ghotra (IIT2024226)
"""

from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) across probability bins.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1

    n_samples = len(y_true)
    ece = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask] == BONAFIDE)
            bin_conf = np.mean(y_prob[mask])
            bin_weight = np.sum(mask) / n_samples
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    return float(ece)


class PlattCalibrator:
    """Platt Scaling: Fits univariate Logistic Regression on validation continuous scores/logits.

    Maps uncalibrated logit scores L in (-inf, +inf) to calibrated probabilities P(bona fide) in [0, 1].
    Strict Rule: Must be fitted solely on DEV split; eval split is never used for fitting.
    """

    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs", max_iter=500)
        self.is_fitted = False
        self.dev_ece: float = 0.0

    def fit(self, dev_scores: np.ndarray, dev_labels: np.ndarray) -> "PlattCalibrator":
        """Fits sigmoid mapping on dev scores and dev labels."""
        X = dev_scores.reshape(-1, 1)
        y = dev_labels.astype(int)

        self.model.fit(X, y)
        self.is_fitted = True

        # Calculate dev ECE
        probs = self.predict_proba(dev_scores)
        self.dev_ece = compute_expected_calibration_error(dev_labels, probs)
        logger.info(
            "Fitted Platt Calibrator on %d dev scores. Dev ECE: %.4f (Slope: %.4f, Intercept: %.4f)",
            len(dev_scores),
            self.dev_ece,
            self.model.coef_[0][0],
            self.model.intercept_[0],
        )
        return self

    def predict_proba(self, scores: np.ndarray | torch.Tensor) -> np.ndarray:
        """Converts raw scores to calibrated probability of bona fide."""
        if not self.is_fitted:
            # Fallback to standard sigmoid if unfitted
            if isinstance(scores, torch.Tensor):
                return torch.sigmoid(scores).detach().cpu().numpy()
            return 1.0 / (1.0 + np.exp(-np.clip(scores, -20.0, 20.0)))

        if isinstance(scores, torch.Tensor):
            s = scores.detach().cpu().numpy()
        else:
            s = np.asarray(scores)

        X = s.reshape(-1, 1)
        # Class 1 is BONAFIDE
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path | str) -> None:
        """Serializes fitted calibrator using joblib."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(dest))
        logger.info("Saved Platt Calibrator to %s", dest)

    @classmethod
    def load(cls, path: Path | str) -> "PlattCalibrator":
        """Loads serialized calibrator from disk."""
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Calibrator file not found at: {src}")
        return joblib.load(str(src))
