from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import torch

from acforensics.constants import BONAFIDE, SPOOF, SAMPLE_RATE, TARGET_SAMPLES
from acforensics.data.protocols import load_in_the_wild_protocol
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import generate_multi_crop_windows, repeat_pad
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_metrics_at_threshold,
    bootstrap_confidence_interval,
)
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)

def aggregate_window_scores(window_scores: list[float], strategy: str = "mean", top_k_ratio: float = 0.5) -> float:
    if not window_scores:
        return 0.0
    arr = np.array(window_scores, dtype=np.float64)
    if strategy == "max":
        return float(arr[np.argmax(np.abs(arr))])
    elif strategy == "top_k":
        k = max(1, int(len(arr) * top_k_ratio))
        sorted_asc = np.sort(arr)
        return float(np.mean(sorted_asc[:k]))
    return float(np.mean(arr))

def evaluate_in_the_wild(
    model: Any,
    meta_csv: Path | str,
    audio_dir: Optional[Path | str] = None,
    dev_threshold: Optional[float] = None,
    feature_extractor: Optional[Any] = None,
    device: Optional[torch.device] = None,
    max_clips: Optional[int] = None,
    aggregation: str = "mean",
    abstain_margin: float = 0.05,
    seed: int = 42,
    n_bootstraps: int = 1000,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    if hasattr(model, "to"):
        model = model.to(device)
        model.eval()

    if str(meta_csv).endswith(".parquet"):
        df = pd.read_parquet(meta_csv)
    else:
        df = load_in_the_wild_protocol(meta_csv, audio_dir=audio_dir)
    if max_clips is not None:
        df = df.head(max_clips)

    all_scores = []
    all_labels = []
    all_durations = []
    clip_categories = []

    logger.info("Evaluating %d In-The-Wild clips via multi-crop sliding window inference (agg=%s)...", len(df), aggregation)

    path_col = "path" if "path" in df.columns else "audio_path"
    for _, row in df.iterrows():
        audio_path = row[path_col]
        res = load_and_validate_audio(audio_path, target_sr=SAMPLE_RATE)
        if not res.is_valid or res.waveform is None:
            continue

        dur = float(len(res.waveform) / SAMPLE_RATE)
        all_durations.append(dur)
        if dur < 1.0:
            clip_categories.append("short")
        elif dur > 4.0:
            clip_categories.append("long")
        else:
            clip_categories.append("medium")

        windows = generate_multi_crop_windows(res.waveform, window_samples=TARGET_SAMPLES, hop_ratio=0.5)
        window_scores = []

        if hasattr(model, "predict_waveform"):
            for w in windows:
                pred = model.predict_waveform(w["waveform"])
                window_scores.append(float(pred["fused_score"]))
        else:
            with torch.no_grad():
                if feature_extractor is not None:
                    raw_feats = [feature_extractor(w["waveform"]) for w in windows]
                    norm_feats = [per_utterance_cmvn(f) for f in raw_feats]
                    batch_feats = torch.stack(norm_feats)
                    if batch_feats.dim() == 3:
                        batch_feats = batch_feats.unsqueeze(1)
                    batch_feats = batch_feats.to(device)
                    logits, _ = model(batch_feats)
                    scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
                    window_scores = [float(s) for s in scores]
                else:
                    batch_waves = torch.stack([w["waveform"] for w in windows]).to(device)
                    logits, _ = model(batch_waves)
                    scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
                    window_scores = [float(s) for s in scores]

        agg_score = aggregate_window_scores(window_scores, strategy=aggregation)
        all_scores.append(agg_score)
        all_labels.append(row["label"])

    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int64)
    cats_arr = np.array(clip_categories)

    itw_eer, itw_opt_thresh = compute_eer(labels_arr, scores_arr)
    itw_auc = compute_roc_auc(labels_arr, scores_arr)
    itw_eer_ci = bootstrap_confidence_interval(
        lambda l, s: compute_eer(l, s)[0],
        labels_arr,
        scores_arr,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    dev_th = dev_threshold if dev_threshold is not None else itw_opt_thresh
    dev_th_metrics = compute_metrics_at_threshold(labels_arr, scores_arr, dev_th)

    probs = 1.0 / (1.0 + np.exp(-np.clip(scores_arr - dev_th, -30.0, 30.0)))
    abstain_mask = np.abs(probs - 0.5) < abstain_margin
    n_abstain = int(np.sum(abstain_mask))
    coverage = float((len(labels_arr) - n_abstain) / max(1, len(labels_arr)) * 100.0)

    if np.any(~abstain_mask) and len(np.unique(labels_arr[~abstain_mask])) >= 2:
        non_ab_metrics = compute_metrics_at_threshold(labels_arr[~abstain_mask], scores_arr[~abstain_mask], dev_th)
        non_ab_eer, _ = compute_eer(labels_arr[~abstain_mask], scores_arr[~abstain_mask])
        non_ab_acc = non_ab_metrics["accuracy"] * 100.0
    else:
        non_ab_eer = itw_eer
        non_ab_acc = dev_th_metrics["accuracy"] * 100.0

    length_breakdown = {}
    for cat in ["short", "long"]:
        m_cat = cats_arr == cat
        if np.sum(m_cat) >= 10 and len(np.unique(labels_arr[m_cat])) >= 2:
            e_cat, _ = compute_eer(labels_arr[m_cat], scores_arr[m_cat])
            a_cat = compute_roc_auc(labels_arr[m_cat], scores_arr[m_cat])
            length_breakdown[cat] = {
                "count": int(np.sum(m_cat)),
                "eer": float(e_cat),
                "roc_auc": float(a_cat),
            }
        else:
            length_breakdown[cat] = {
                "count": int(np.sum(m_cat)),
                "eer": None,
                "roc_auc": None,
            }

    results = {
        "dataset": "In-the-Wild Deepfake Audio (Müller et al., 2022)",
        "num_evaluated_clips": len(all_labels),
        "bonafide_count": int(np.sum(labels_arr == BONAFIDE)),
        "spoof_count": int(np.sum(labels_arr == SPOOF)),
        "aggregation_strategy": aggregation,
        "optimal_threshold_metrics": {
            "itw_optimal_eer": float(itw_eer),
            "itw_optimal_eer_ci_95": [float(itw_eer_ci[0]), float(itw_eer_ci[1])],
            "itw_optimal_threshold": float(itw_opt_thresh),
            "roc_auc": float(itw_auc),
        },
        "dev_threshold_metrics": {
            "asvspoof_dev_threshold": float(dev_th),
            "accuracy": float(dev_th_metrics["accuracy"] * 100.0),
            "precision": float(dev_th_metrics["precision"]),
            "recall": float(dev_th_metrics["recall"]),
            "f1": float(dev_th_metrics["f1"]),
        },
        "abstain_analysis": {
            "abstain_margin": float(abstain_margin),
            "abstain_count": n_abstain,
            "coverage_pct": float(coverage),
            "non_abstained_eer": float(non_ab_eer),
            "non_abstained_accuracy": float(non_ab_acc),
        },
        "clip_length_breakdown": length_breakdown,
        "min_tdcf_note": "min-tDCF is undefined on In-The-Wild as the corpus lacks paired ASV speaker verification trials.",
    }
    return results
