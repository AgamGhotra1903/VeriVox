import numpy as np


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.0, 0.0

    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    sorted_scores = np.sort(np.unique(all_scores))

    n_bonafide = len(bonafide_scores)
    n_spoof = len(spoof_scores)
    sorted_spoof = np.sort(spoof_scores)
    sorted_bf = np.sort(bonafide_scores)
    far = (n_spoof - np.searchsorted(sorted_spoof, sorted_scores, side="left")) / n_spoof
    frr = np.searchsorted(sorted_bf, sorted_scores, side="left") / n_bonafide

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


def compute_det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0.0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1.0), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


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
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 1.0, 0.0
    if len(asv_target_scores) == 0 or len(asv_nontarget_scores) == 0:
        raise ValueError("ASV target and nontarget scores are required for t-DCF calculation.")

    asv_frr, asv_far, asv_thresholds = compute_det_curve(asv_target_scores, asv_nontarget_scores)
    eer_idx = np.argmin(np.abs(asv_frr - asv_far))
    asv_threshold = asv_thresholds[eer_idx]

    p_miss_asv = np.mean(asv_target_scores < asv_threshold)
    p_fa_asv = np.mean(asv_nontarget_scores >= asv_threshold)
    p_fa_spoof_asv = np.mean(asv_spoof_scores >= asv_threshold) if len(asv_spoof_scores) > 0 else 0.0

    c0 = p_tar * cost_miss * p_miss_asv + p_non * cost_fa * p_fa_asv
    c1 = p_tar * cost_miss - c0
    c2 = p_spoof * cost_fa_spoof * p_fa_spoof_asv

    cm_frr, cm_far, cm_thresholds = compute_det_curve(bonafide_scores, spoof_scores)

    t_dcf = c0 + c1 * cm_frr + c2 * cm_far
    t_dcf_default = c0 + min(c1, c2)
    t_dcf_norm = t_dcf / (t_dcf_default + 1e-12)

    min_idx = np.argmin(t_dcf_norm)
    min_tdcf = float(t_dcf_norm[min_idx])
    opt_th = float(cm_thresholds[min_idx])

    return min_tdcf, opt_th
