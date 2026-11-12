"""Stacked Ensemble (Score-Level Fusion) and Meta-Learner.

Leakage Architecture Rules:
1. Base models are trained on TRAIN split only.
2. The meta-learner (L2 Logistic Regression) and its StandardScaler are fitted on DEV split only.
3. EVAL and In-The-Wild splits are evaluated frozen (no refitting, no threshold retuning).
4. Enforces zero utterance ID overlap between train, dev, and eval with explicit assertions.

Outputs:
- Continuous fused score (higher = more bona fide).
- Calibrated confidence probability P(bona fide) via sigmoid of meta-learner.
- Baseline mean fusion (mean of z-normalized scores) for benchmark comparison.

"""

from pathlib import Path
from typing import Any, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def assert_zero_data_leakage(
    train_ids: list[str] | set[str],
    dev_ids: list[str] | set[str],
    eval_ids: list[str] | set[str],
    itw_ids: Optional[list[str] | set[str]] = None,
) -> None:
    """Enforces strict data partition disjointness with mathematical assertions.

    Raises:
        AssertionError: If any utterance ID appears in more than one partition.
    """
    s_train = set(train_ids)
    s_dev = set(dev_ids)
    s_eval = set(eval_ids)

    inter_train_dev = s_train.intersection(s_dev)
    assert len(inter_train_dev) == 0, (
        f"CRITICAL LEAKAGE DETECTED: {len(inter_train_dev)} IDs shared between TRAIN and DEV! Samples: {list(inter_train_dev)[:5]}"
    )

    inter_train_eval = s_train.intersection(s_eval)
    assert len(inter_train_eval) == 0, (
        f"CRITICAL LEAKAGE DETECTED: {len(inter_train_eval)} IDs shared between TRAIN and EVAL! Samples: {list(inter_train_eval)[:5]}"
    )

    inter_dev_eval = s_dev.intersection(s_eval)
    assert len(inter_dev_eval) == 0, (
        f"CRITICAL LEAKAGE DETECTED: {len(inter_dev_eval)} IDs shared between DEV and EVAL! Samples: {list(inter_dev_eval)[:5]}"
    )

    if itw_ids is not None:
        s_itw = set(itw_ids)
        inter_train_itw = s_train.intersection(s_itw)
        assert len(inter_train_itw) == 0, (
            f"CRITICAL LEAKAGE DETECTED: {len(inter_train_itw)} IDs shared between TRAIN and IN-THE-WILD!"
        )

    logger.info("Zero-leakage verification passed: TRAIN (%d), DEV (%d), EVAL (%d) are completely disjoint.",
                len(s_train), len(s_dev), len(s_eval))


class StackedEnsemble:
    """Score-level stacked ensemble using an L2-regularized Logistic Regression meta-learner."""

    def __init__(
        self,
        model_names: Optional[list[str]] = None,
        seed: int = 42,
    ):
        """Initializes StackedEnsemble.

        Args:
            model_names: List of base model names corresponding to score matrix columns.
            seed: RNG seed for reproducible cross-validation.
        """
        self.model_names = model_names or []
        self.seed = seed
        self.scaler = StandardScaler()
        self.meta_learner: Optional[LogisticRegressionCV] = None
        self.is_fitted = False
        self.best_c: float = 1.0

    def fit(
        self,
        dev_scores: np.ndarray,
        dev_labels: np.ndarray,
        model_names: Optional[list[str]] = None,
    ) -> "StackedEnsemble":
        """Fits StandardScaler and LogisticRegressionCV meta-learner strictly on DEV scores only.

        Args:
            dev_scores: (N_dev, M_models) continuous scores matrix.
            dev_labels: (N_dev,) binary ground truth labels.
            model_names: Optional override for model names list.

        Returns:
            self
        """
        assert dev_scores.ndim == 2, f"Expected 2D score matrix (N, M), got shape {dev_scores.shape}"
        assert len(dev_scores) == len(dev_labels), "Sample count mismatch between scores and labels"
        assert len(np.unique(dev_labels)) >= 2, "Dev split must contain both bona fide and spoof instances"

        if model_names is not None:
            self.model_names = model_names
        elif not self.model_names:
            self.model_names = [f"Model_{i}" for i in range(dev_scores.shape[1])]

        assert dev_scores.shape[1] == len(self.model_names), (
            f"Model count mismatch: matrix has {dev_scores.shape[1]} columns but {len(self.model_names)} names provided"
        )

        logger.info(
            "Fitting StackedEnsemble meta-learner on DEV score matrix (%d utterances, %d base models)...",
            dev_scores.shape[0], dev_scores.shape[1],
        )

        z_dev = self.scaler.fit_transform(dev_scores)

        n_min_class = int(np.min(np.bincount(dev_labels)))
        cv_folds = max(2, min(5, n_min_class))

        self.meta_learner = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv=cv_folds,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            scoring="roc_auc",
            random_state=self.seed,
            n_jobs=-1,
        )
        self.meta_learner.fit(z_dev, dev_labels)
        self.best_c = float(self.meta_learner.C_[0])
        self.is_fitted = True

        coefs = self.get_coefficients()
        logger.info("Meta-learner fitted successfully. Selected C: %.4f | Intercept: %.4f",
                    self.best_c, float(self.meta_learner.intercept_[0]))
        for m_name, weight in coefs.items():
            logger.info("  -> Base Model '%s': weight = %+.4f", m_name, weight)

        return self

    def predict_score(self, scores: np.ndarray) -> np.ndarray:
        """Computes fused continuous countermeasure score where higher value = more bona fide.

        Evaluates decision_function of the meta-learner on z-normalized base scores:
            Fused_Score = w^T * z(scores) + b
        """
        assert self.is_fitted and self.meta_learner is not None, "StackedEnsemble must be fitted on DEV first."
        z_scores = self.scaler.transform(scores)
        return self.meta_learner.decision_function(z_scores)

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Computes calibrated probability of authentic speech P(bona fide) in [0, 1].

        Evaluates sigmoid of the meta-learner output:
            P(bona fide) = 1 / (1 + exp(-Fused_Score))
        """
        assert self.is_fitted and self.meta_learner is not None, "StackedEnsemble must be fitted on DEV first."
        z_scores = self.scaler.transform(scores)
        return self.meta_learner.predict_proba(z_scores)[:, BONAFIDE]

    def predict_baseline_mean(self, scores: np.ndarray) -> np.ndarray:
        """Simple baseline fusion: unweighted mean of dev z-normalized base scores.

        Used for direct ablation comparison against the learned stacked ensemble.
        """
        assert self.is_fitted, "StackedEnsemble must be fitted on DEV to provide z-score scaling."
        z_scores = self.scaler.transform(scores)
        return np.mean(z_scores, axis=1)

    def get_coefficients(self) -> dict[str, float]:
        """Returns dictionary mapping base model names to their meta-learner coefficients."""
        assert self.is_fitted and self.meta_learner is not None, "StackedEnsemble not fitted."
        weights = self.meta_learner.coef_[0]
        return {name: float(w) for name, w in zip(self.model_names, weights)}

    def get_coefficients_table(self) -> pd.DataFrame:
        """Generates a structured DataFrame summarizing base model contributions."""
        coefs = self.get_coefficients()
        rows = []
        for name, weight in coefs.items():
            rows.append({
                "Base Model": name,
                "Meta-Learner Weight": round(weight, 4),
                "Absolute Contribution": round(abs(weight), 4),
            })
        df = pd.DataFrame(rows).sort_values(by="Absolute Contribution", ascending=False).reset_index(drop=True)
        return df

    def save(self, output_path: Path | str) -> None:
        """Serializes fitted ensemble (scaler, meta-learner, model names) to disk."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(out))
        logger.info("Saved StackedEnsemble model to %s", out)

    @classmethod
    def load(cls, model_path: Path | str) -> "StackedEnsemble":
        """Loads serialized StackedEnsemble from disk."""
        return joblib.load(str(model_path))

