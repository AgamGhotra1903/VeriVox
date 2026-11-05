"""Unit tests for EER, ROC-AUC, min-tDCF, and bootstrap confidence intervals.

Author: Chunit Bansal (IIT2024223)
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
    # 50 bona fide with high scores (e.g. 5.0 to 10.0)
    # 50 spoof with low scores (e.g. -10.0 to -5.0)
    labels = np.array([BONAFIDE] * 50 + [SPOOF] * 50)
    scores = np.array([5.0 + i * 0.1 for i in range(50)] + [-5.0 - i * 0.1 for i in range(50)])

    eer, th = compute_eer(labels, scores)
    assert eer == pytest.approx(0.0, abs=1e-5)
    # Threshold should separate negative and positive scores
    assert -5.0 < th <= 5.0


def test_eer_random_guess():
    np.random.seed(42)
    n = 2000
    labels = np.random.choice([BONAFIDE, SPOOF], size=n)
    scores = np.random.normal(0, 1, size=n)

    eer, _ = compute_eer(labels, scores)
    # Random guess EER should be close to 50%
    assert 45.0 <= eer <= 55.0


def test_eer_flipped_sign():
    """Verifies that inverted scores (spoof higher than bona fide) yields EER ~ 100%."""
    # Perfectly separated with wrong sign
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


def test_min_tdcf_missing_asv_file_raises():
    labels = np.array([BONAFIDE] * 20 + [SPOOF] * 80)
    scores = np.random.normal(1.0, 1.0, size=100)
    with pytest.raises(ValueError, match="Official ASV score file is required"):
        compute_min_tdcf(labels, scores)


def test_min_tdcf_with_asv_score_file(tmp_path: Path):
    asv_file = tmp_path / "mock_asv_scores.txt"
    with open(asv_file, "w", encoding="utf-8") as f:
        f.write("bonafide target 5.5\n")
        f.write("bonafide nontarget -2.8\n")
        f.write("A01 spoof -3.3\n")

    labels = np.array([BONAFIDE] * 10 + [SPOOF] * 10)
    scores = np.array([3.0] * 10 + [-3.0] * 10)
    min_tdcf, opt_th = compute_min_tdcf(labels, scores, asv_score_file=str(asv_file))
    assert min_tdcf >= 0.0
    assert isinstance(opt_th, float)


def test_min_tdcf_matches_official_formula_within_tolerance(tmp_path: Path):
    np.random.seed(42)
    n_tar = 1500
    n_non = 1500
    n_spoof_asv = 2000
    tar_scores = np.random.normal(6.0, 2.0, n_tar)
    non_scores = np.random.normal(-3.0, 2.0, n_non)
    spoof_asv_scores = np.random.normal(-1.0, 2.0, n_spoof_asv)

    asv_file = tmp_path / "asv_test_eval.txt"
    with open(asv_file, "w", encoding="utf-8") as f:
        for s in tar_scores:
            f.write(f"bonafide target {s:.6f}\n")
        for s in non_scores:
            f.write(f"bonafide nontarget {s:.6f}\n")
        for s in spoof_asv_scores:
            f.write(f"A07 spoof {s:.6f}\n")

    n_bona = 1000
    n_spoof = 3000
    labels = np.array([BONAFIDE] * n_bona + [SPOOF] * n_spoof)
    cm_scores = np.concatenate([
        np.random.normal(2.5, 1.2, n_bona),
        np.random.normal(-1.5, 1.2, n_spoof),
    ])

    min_tdcf, opt_th = compute_min_tdcf(labels, cm_scores, asv_score_file=str(asv_file))

    from acforensics.evaluation.official_tdcf.eval_metrics import (
        compute_det_curve,
    )
    asv_frr, asv_far, asv_th = compute_det_curve(tar_scores, non_scores)
    eer_idx = np.argmin(np.abs(asv_frr - asv_far))
    tau_asv = asv_th[eer_idx]
    p_miss_asv = np.mean(tar_scores < tau_asv)
    p_fa_asv = np.mean(non_scores >= tau_asv)
    p_fa_spoof_asv = np.mean(spoof_asv_scores >= tau_asv)

    c0 = 0.9405 * 1.0 * p_miss_asv + 0.0095 * 10.0 * p_fa_asv
    c1 = 0.9405 * 1.0 - c0
    c2 = 0.05 * 10.0 * p_fa_spoof_asv

    bona_cm = cm_scores[:n_bona]
    spoof_cm = cm_scores[n_bona:]
    cm_frr, cm_far, cm_th = compute_det_curve(bona_cm, spoof_cm)
    t_dcf = c0 + c1 * cm_frr + c2 * cm_far
    t_dcf_norm = t_dcf / (c0 + min(c1, c2))
    expected_min_tdcf = float(np.min(t_dcf_norm))

    assert abs(min_tdcf - expected_min_tdcf) < 1e-6


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
