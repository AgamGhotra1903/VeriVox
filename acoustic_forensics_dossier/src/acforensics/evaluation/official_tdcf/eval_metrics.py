"""Official ASVspoof 2019 evaluation metrics module (vendored).

License: ASVspoof 2019 Evaluation Campaign (University of Edinburgh / EURECOM).
"""

import numpy as np


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    """Computes Equal Error Rate (EER) and operating threshold using official ASVspoof convention.

    Scores convention: Higher score = more likely genuine (bona fide).

    Args:
        bonafide_scores: 1D array of scores for genuine/bona fide trials.
        spoof_scores: 1D array of scores for synthetic/spoofed trials.

    Returns:
        Tuple of (eer_percentage, threshold).
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0, 0.0

    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    sorted_scores = np.sort(np.unique(all_scores))

    far = np.zeros(len(sorted_scores))
    frr = np.zeros(len(sorted_scores))

    n_bonafide = len(bonafide_scores)
    n_spoof = len(spoof_scores)

    for i, thresh in enumerate(sorted_scores):
        far[i] = np.sum(spoof_scores >= thresh) / n_spoof
        frr[i] = np.sum(bonafide_scores < thresh) / n_bonafide

    diff = far - frr
    idx = np.argmin(np.abs(diff))

    if idx == 0 or idx == len(sorted_scores) - 1:
        eer = (far[idx] + frr[idx]) / 2.0
        threshold = sorted_scores[idx]
    else:
        if diff[idx] * diff[idx + 1] < 0:
            i1, i2 = idx, idx + 1
        elif diff[idx] * diff[idx - 1] < 0:
            i1, i2 = idx - 1, idx
        else:
            i1, i2 = idx, idx

        if i1 != i2:
            slope_far = far[i2] - far[i1]
            slope_frr = frr[i2] - frr[i1]
            d = (frr[i1] - far[i1]) / (slope_far - slope_frr + 1e-12)
            eer = far[i1] + slope_far * d
            threshold = sorted_scores[i1] + d * (sorted_scores[i2] - sorted_scores[i1])
        else:
            eer = (far[idx] + frr[idx]) / 2.0
            threshold = sorted_scores[idx]

    return float(eer * 100.0), float(threshold)


def compute_tDCF_legacy(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    asv_target_scores: np.ndarray,
    asv_nontarget_scores: np.ndarray,
    asv_spoof_scores: np.ndarray,
    cost_miss: float = 1.0,
    cost_fa: float = 10.0,
    cost_fa_spoof: float = 10.0,
    p_spoof: float = 0.05,
    p_tar: float = 0.9405,
    p_non: float = 0.0095,
) -> tuple[float, float]:
    """Official ASVspoof 2019 minimum normalized tandem Detection Cost Function (min-tDCF).

    Args:
        bonafide_scores: CM scores for bona fide trials.
        spoof_scores: CM scores for spoof trials.
        asv_target_scores: ASV scores for genuine target speakers.
        asv_nontarget_scores: ASV scores for zero-effort impostors.
        asv_spoof_scores: ASV scores for spoof trials.

    Returns:
        Tuple of (min_tDCF, optimal_threshold).
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 1.0, 0.0

    all_cm_scores = np.concatenate([bonafide_scores, spoof_scores])
    thresholds = np.sort(np.unique(all_cm_scores))

    c_miss_cm = cost_miss
    c_fa_cm = cost_fa

    c0 = p_tar * c_miss_cm
    c1 = p_spoof * cost_fa_spoof

    c_default = min(c0, c1)

    t_dcfs = []
    for th in thresholds:
        p_miss_cm = np.mean(bonafide_scores < th)
        p_fa_cm = np.mean(spoof_scores >= th)

        cost = c0 * p_miss_cm + c1 * p_fa_cm
        norm_cost = cost / (c_default + 1e-12)
        t_dcfs.append(norm_cost)

    min_idx = np.argmin(t_dcfs)
    min_tdcf = float(t_dcfs[min_idx])
    opt_th = float(thresholds[min_idx])

    return min_tdcf, opt_th
