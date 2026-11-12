"""Unit tests for EER, ROC-AUC, min-tDCF, and bootstrap confidence intervals.

"""

from pathlib import Path
import pytest
import numpy as np

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_min_tdcf,
    compute_metrics_at_threshold,
    bootstrap_confidence_interval,
    full_evaluation,
)


def test_eer_perfect_separation():
    labels = np.array([BONAFIDE] * 50 + [SPOOF] * 50)
    scores = np.array([5.0 + i * 0.1 for i in range(50)] + [-5.0 - i * 0.1 for i in range(50)])

    eer, th = compute_eer(labels, scores)
    assert eer == pytest.approx(0.0, abs=1e-5)
    assert -5.0 < th <= 5.0


def test_eer_random_guess():
    np.random.seed(42)
    n = 2000
    labels = np.random.choice([BONAFIDE, SPOOF], size=n)
    scores = np.random.normal(0, 1, size=n)

    eer, _ = compute_eer(labels, scores)
    assert 45.0 <= eer <= 55.0


def test_eer_flipped_sign():
    """Verifies that inverted scores (spoof higher than bona fide) yields EER ~ 100%."""
    labels = np.array([BONAFIDE] * 50 + [SPOOF] * 50)
    scores = np.array([-5.0 - i * 0.1 for i in range(50)] + [5.0 + i * 0.1 for i in range(50)])

    eer, _ = compute_eer(labels, scores)
    assert eer >= 99.0


def test_roc_auc_perfect_and_inverted():
    labels = np.array([BONAFIDE, BONAFIDE, SPOOF, SPOOF])
    scores = np.array([10.0, 8.0, -1.0, -5.0])
    auc = compute_roc_auc(labels, scores)
    assert auc == 1.0

    inverted_scores = -scores
    auc_inv = compute_roc_auc(labels, inverted_scores)
    assert auc_inv == 0.0


def test_min_tdcf_computation():
    labels = np.array([BONAFIDE] * 20 + [SPOOF] * 80)
    scores = np.random.normal(1.0, 1.0, size=100)
    min_tdcf, opt_th = compute_min_tdcf(labels, scores)
    assert min_tdcf >= 0.0
    assert isinstance(opt_th, float)


def test_min_tdcf_with_asv_score_file(tmp_path: Path):
    asv_file = tmp_path / "mock_asv_scores.txt"
    with open(asv_file, "w", encoding="utf-8") as f:
        f.write("SPK1 F1 target 2.5 bonafide\n")
        f.write("SPK2 F2 nontarget -1.8 spoof\n")
        f.write("SPK3 F3 spoof -2.3 spoof\n")

    labels = np.array([BONAFIDE] * 10 + [SPOOF] * 10)
    scores = np.array([3.0] * 10 + [-3.0] * 10)
    min_tdcf, opt_th = compute_min_tdcf(labels, scores, asv_score_file=str(asv_file))
    assert min_tdcf >= 0.0


def test_bootstrap_confidence_interval():
    np.random.seed(42)
    labels = np.array([BONAFIDE] * 30 + [SPOOF] * 70)
    scores = np.concatenate([np.random.normal(2.0, 1.0, 30), np.random.normal(-1.0, 1.0, 70)])

    low, high = bootstrap_confidence_interval(
        compute_roc_auc,
        labels,
        scores,
        n_bootstraps=200,
        alpha=0.05,
        seed=42,
    )
    assert 0.5 <= low <= high <= 1.0


def test_full_evaluation_report():
    labels = np.array([BONAFIDE] * 20 + [SPOOF] * 80)
    scores = np.concatenate([np.random.normal(2.0, 1.0, 20), np.random.normal(-1.0, 1.0, 80)])

    report = full_evaluation(labels, scores, dev_threshold=0.5, compute_ci=False)
    assert "eer" in report
    assert "roc_auc" in report
    assert "min_tdcf" in report
    assert "at_applied_threshold" in report
    assert report["applied_threshold"] == 0.5
