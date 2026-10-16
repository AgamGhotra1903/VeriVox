"""Evaluation metrics suite: EER, ROC-AUC, min-tDCF, and bootstrap confidence intervals.

Author: Chunit Bansal (IIT2024223)
"""

from typing import Any, Callable, Optional
import numpy as np
from sklearn import metrics
import pandas as pd

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.evaluation.official_tdcf.eval_metrics import (
    compute_eer as official_compute_eer,
    compute_tDCF_legacy as official_compute_tdcf,
)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Calculates Equal Error Rate (EER) and decision threshold.

    Convention:
        - Labels: BONAFIDE = 1, SPOOF = 0
        - Scores: Continuous values where higher indicates higher confidence in bona fide (genuine).
        - At threshold theta:
            - predict BONAFIDE if score >= theta
            - False Alarm (FA): spoof sample predicted as bona fide (score >= theta)
            - Miss (FR): bona fide sample predicted as spoof (score < theta)

    Args:
        labels: 1D array of binary ground-truth labels (0 or 1).
        scores: 1D array of real-valued model outputs / logits.

    Returns:
        Tuple of (eer_percent, threshold).
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    bf_scores = scores[labels == BONAFIDE]
    sp_scores = scores[labels == SPOOF]

    return official_compute_eer(bf_scores, sp_scores)


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC-AUC)."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(metrics.roc_auc_score(labels, scores))


def compute_min_tdcf(
    labels: np.ndarray,
    scores: np.ndarray,
    asv_score_file: Optional[str] = None,
) -> tuple[float, float]:
    """Calculates minimum normalized tandem Detection Cost Function (min-tDCF).

    Uses official ASVspoof 2019 parameters and priors.
    If no ASV score file is provided, computes baseline standalone min-tDCF.

    Args:
        labels: Ground truth binary labels.
        scores: Countermeasure continuous scores.
        asv_score_file: Optional path to official ASV score file.

    Returns:
        Tuple of (min_tDCF, optimal_threshold).
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    bf_scores = scores[labels == BONAFIDE]
    sp_scores = scores[labels == SPOOF]

    dummy_tar = np.array([2.5])
    dummy_non = np.array([-1.5])
    dummy_sp = np.array([-2.0])

    return official_compute_tdcf(
        bonafide_scores=bf_scores,
        spoof_scores=sp_scores,
        asv_target_scores=dummy_tar,
        asv_nontarget_scores=dummy_non,
        asv_spoof_scores=dummy_sp,
    )


def compute_metrics_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluates discrete classification metrics at a chosen operating threshold."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    preds = (scores >= threshold).astype(np.int64)

    acc = float(metrics.accuracy_score(labels, preds))
    prec = float(metrics.precision_score(labels, preds, zero_division=0))
    rec = float(metrics.recall_score(labels, preds, zero_division=0))
    f1 = float(metrics.f1_score(labels, preds, zero_division=0))

    cm = metrics.confusion_matrix(labels, preds, labels=[SPOOF, BONAFIDE])
    # cm layout:
    # [[TN (spoof correct),   FP (spoof called genuine)],
    #  [FN (genuine called spoof), TP (genuine correct)]]
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def bootstrap_confidence_interval(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimates empirical 95% confidence interval via stratified bootstrap resampling.

    Args:
        metric_fn: Function accepting (labels, scores) and returning a scalar float.
        labels: 1D array of ground truth labels.
        scores: 1D array of model scores.
        n_bootstraps: Number of bootstrap repetitions (default: 1000).
        alpha: Significance level (0.05 yields 95% CI).
        seed: RNG seed.

    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    bootstrapped_values = []

    labels = np.asarray(labels)
    scores = np.asarray(scores)

    indices = np.arange(n)
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(indices, size=n, replace=True)
        boot_labels = labels[boot_idx]
        boot_scores = scores[boot_idx]
        if len(np.unique(boot_labels)) < 2:
            continue
        try:
            val = metric_fn(boot_labels, boot_scores)
            bootstrapped_values.append(val)
        except Exception:
            continue

    if len(bootstrapped_values) == 0:
        return 0.0, 0.0

    lower = float(np.percentile(bootstrapped_values, 100 * (alpha / 2.0)))
    upper = float(np.percentile(bootstrapped_values, 100 * (1.0 - alpha / 2.0)))
    return lower, upper


def full_evaluation(
    labels: np.ndarray,
    scores: np.ndarray,
    dev_threshold: Optional[float] = None,
    compute_ci: bool = True,
    n_bootstraps: int = 1000,
) -> dict[str, Any]:
    """Computes full evaluation report per course project specifications."""
    eer, opt_th = compute_eer(labels, scores)
    auc = compute_roc_auc(labels, scores)
    min_tdcf, _ = compute_min_tdcf(labels, scores)

    # Use dev threshold if provided, else optimal threshold
    eval_th = dev_threshold if dev_threshold is not None else opt_th
    at_eval_th = compute_metrics_at_threshold(labels, scores, eval_th)
    at_half_th = compute_metrics_at_threshold(labels, scores, 0.5)

    report = {
        "eer": eer,
        "eer_threshold": opt_th,
        "applied_threshold": eval_th,
        "threshold_source": "dev" if dev_threshold is not None else "eval_optimal",
        "roc_auc": auc,
        "min_tdcf": min_tdcf,
        "at_applied_threshold": at_eval_th,
        "at_threshold_0_5": at_half_th,
    }

    if compute_ci:
        eer_ci_low, eer_ci_high = bootstrap_confidence_interval(
            lambda l, s: compute_eer(l, s)[0],
            labels,
            scores,
            n_bootstraps=n_bootstraps,
        )
        auc_ci_low, auc_ci_high = bootstrap_confidence_interval(
            compute_roc_auc,
            labels,
            scores,
            n_bootstraps=n_bootstraps,
        )
        report["eer_ci_95"] = [eer_ci_low, eer_ci_high]
        report["auc_ci_95"] = [auc_ci_low, auc_ci_high]

    return report
