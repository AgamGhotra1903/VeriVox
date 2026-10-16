"""In-The-Wild Cross-Domain Evaluation Suite (Müller et al., Interspeech 2022).

Author: Chunit Bansal (IIT2024223)
"""

from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import torch

from acforensics.constants import BONAFIDE, SPOOF, SAMPLE_RATE, TARGET_SAMPLES
from acforensics.data.protocols import load_in_the_wild_protocol
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import generate_multi_crop_windows
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_metrics_at_threshold,
    bootstrap_confidence_interval,
)
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_in_the_wild(
    model: torch.nn.Module,
    meta_csv: Path | str,
    audio_dir: Optional[Path | str] = None,
    dev_threshold: Optional[float] = None,
    feature_extractor: Optional[Any] = None,
    device: Optional[torch.device] = None,
    max_clips: Optional[int] = None,
) -> dict[str, Any]:
    """Runs zero-shot cross-domain evaluation on In-The-Wild Deepfake dataset.

    Methodology:
    - Never trained on ITW.
    - Variable-length audio: uses multi-crop inference with 50% window overlap.
    - Window continuous scores (logit[BONAFIDE] - logit[SPOOF]) are averaged per utterance.
    - min-tDCF is NOT computed (no paired Automatic Speaker Verification trials exist).
    - Reports:
        (1) In-domain EER at ITW optimal threshold.
        (2) Cross-domain Accuracy & Performance at the threshold selected on ASVspoof dev split.
            The performance gap demonstrates the magnitude of domain and calibration shift.

    Returns:
        Dictionary containing cross-domain evaluation metrics and domain gap analysis.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    df = load_in_the_wild_protocol(meta_csv, audio_dir=audio_dir)
    if max_clips is not None:
        df = df.head(max_clips)

    all_scores = []
    all_labels = []

    logger.info("Evaluating %d In-The-Wild clips via multi-crop sliding window inference...", len(df))

    for _, row in df.iterrows():
        audio_path = row["audio_path"]
        res = load_and_validate_audio(audio_path, target_sr=SAMPLE_RATE)
        if not res.is_valid or res.waveform is None:
            continue

        # Generate overlapping 4.0s windows
        windows = generate_multi_crop_windows(res.waveform, window_samples=TARGET_SAMPLES, hop_ratio=0.5)

        window_scores = []
        with torch.no_grad():
            for w in windows:
                wave_chunk = w["waveform"]
                if feature_extractor is not None:
                    feat = feature_extractor(wave_chunk)
                    if feat.dim() == 2:
                        feat = feat.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
                    elif feat.dim() == 3:
                        feat = feat.unsqueeze(1)
                    feat = feat.to(device)
                    logits, _ = model(feat)
                else:
                    wave_input = wave_chunk.unsqueeze(0).to(device)
                    logits, _ = model(wave_input)

                score = (logits[:, BONAFIDE] - logits[:, SPOOF]).item()
                window_scores.append(score)

        # Average window logits
        mean_score = float(np.mean(window_scores))
        all_scores.append(mean_score)
        all_labels.append(row["label"])

    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # 1. Optimal ITW threshold
    itw_eer, itw_opt_thresh = compute_eer(labels_arr, scores_arr)
    itw_auc = compute_roc_auc(labels_arr, scores_arr)

    itw_eer_ci = bootstrap_confidence_interval(
        lambda l, s: compute_eer(l, s)[0],
        labels_arr,
        scores_arr,
        n_bootstraps=500,
    )

    # 2. Performance at ASVspoof Dev threshold
    dev_th = dev_threshold if dev_threshold is not None else itw_opt_thresh
    dev_th_metrics = compute_metrics_at_threshold(labels_arr, scores_arr, dev_th)

    results = {
        "dataset": "In-the-Wild Deepfake Audio (Müller et al., 2022)",
        "num_evaluated_clips": len(all_labels),
        "bonafide_count": int(np.sum(labels_arr == BONAFIDE)),
        "spoof_count": int(np.sum(labels_arr == SPOOF)),
        "multi_crop_inference": True,
        "optimal_threshold_metrics": {
            "itw_optimal_eer": itw_eer,
            "itw_optimal_eer_ci_95": list(itw_eer_ci),
            "itw_optimal_threshold": itw_opt_thresh,
            "roc_auc": itw_auc,
        },
        "dev_threshold_metrics": {
            "asvspoof_dev_threshold": dev_th,
            "accuracy": dev_th_metrics["accuracy"],
            "precision": dev_th_metrics["precision"],
            "recall": dev_th_metrics["recall"],
            "f1": dev_th_metrics["f1"],
            "confusion_matrix": {
                "tp": dev_th_metrics["tp"],
                "fp": dev_th_metrics["fp"],
                "tn": dev_th_metrics["tn"],
                "fn": dev_th_metrics["fn"],
            },
        },
        "domain_shift_analysis": (
            f"At its own optimal threshold ({itw_opt_thresh:.4f}), the model achieves {itw_eer:.2f}% EER on real-world clips. "
            f"When using the threshold calibrated on ASVspoof dev ({dev_th:.4f}), accuracy is {dev_th_metrics['accuracy']*100:.2f}%. "
            "This difference exposes the distribution shift between studio speech and unconstrained acoustic environments."
        ),
        "min_tdcf_note": "min-tDCF is undefined on In-The-Wild as the corpus lacks paired ASV speaker verification trials.",
    }

    return results
