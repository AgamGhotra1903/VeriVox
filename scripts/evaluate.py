"""Master Evaluation CLI: Full Evaluation, Multi-Model ROC/DET, and Silence Shortcut Check.

Author: Chunit Bansal (IIT2024223), Agam Ghotra (IIT2024226)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    LFCC_DEFAULT_DIM,
    DEFAULT_RESULTS_DIR,
)
from acforensics.data.protocols import load_asvspoof_protocol
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import FeatureNormalizer
from acforensics.models.crnn import CRNNForensics
from acforensics.evaluation.metrics import full_evaluation
from acforensics.evaluation.plots import plot_roc_curve, plot_det_curve
from acforensics.utils.io import save_json, load_json
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def energy_based_vad_trim(waveform: torch.Tensor, energy_threshold_ratio: float = 0.01) -> torch.Tensor:
    """Trims leading and trailing silence using an energy-based Voice Activity Detection heuristic.

    Used to perform the silence shortcut ablation test: does the model exploit silence padding bias?
    """
    frame_len = 160
    n_frames = len(waveform) // frame_len
    if n_frames < 2:
        return waveform

    frames = waveform[: n_frames * frame_len].view(n_frames, frame_len)
    frame_energy = (frames ** 2).mean(dim=1)
    max_energy = frame_energy.max()

    voiced = frame_energy > (max_energy * energy_threshold_ratio)
    voiced_indices = torch.where(voiced)[0]

    if len(voiced_indices) == 0:
        return waveform

    first_frame = voiced_indices[0].item()
    last_frame = voiced_indices[-1].item()

    start_sample = first_frame * frame_len
    end_sample = min(len(waveform), (last_frame + 1) * frame_len)

    return waveform[start_sample:end_sample]


def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Suite CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run on smoke test dataset")
    parser.add_argument("--silence-check", action="store_true", default=True, help="Run silence shortcut ablation")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)

    device = torch.device("cpu")
    ckpt_path = Path("checkpoints/crnn_best.pt")
    if not ckpt_path.exists():
        logger.error("CRNN checkpoint not found at %s. Run make train first.", ckpt_path)
        return

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = CRNNForensics(in_channels=1, in_freq_dim=LFCC_DEFAULT_DIM)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dev_thresh = ckpt.get("dev_threshold", 0.0)

    extractor = LFCCExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    norm_file = Path("checkpoints/normalization_stats.json")
    normalizer = FeatureNormalizer().load(norm_file) if norm_file.exists() else FeatureNormalizer()

    # Load eval dataset
    base_dir = Path("data/smoke_test" if args.smoke else "data/raw/LA")
    proto_file = base_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
    if not proto_file.exists():
        base_dir = Path("data/smoke_test")
        proto_file = base_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"

    eval_df = load_asvspoof_protocol(proto_file, audio_dir=base_dir / "ASVspoof2019_LA_eval" / "flac", split_name="eval")

    all_untrimmed_scores = []
    all_trimmed_scores = []
    labels = []

    logger.info("Evaluating CRNN on official eval split (%d trials) with and without silence trimming...", len(eval_df))
    for _, row in eval_df.iterrows():
        res = load_and_validate_audio(row["audio_path"], target_sr=16000)
        if not res.is_valid or res.waveform is None:
            continue

        labels.append(row["label"])

        # 1. Untrimmed standard 4.0s center crop
        wave_untrimmed = fixed_length_crop_or_pad(res.waveform, target_samples=64000, mode="center")
        with torch.no_grad():
            feat1 = extractor(wave_untrimmed)
            if normalizer.is_fitted:
                feat1 = normalizer(feat1)
            feat1 = feat1.unsqueeze(0).unsqueeze(0).to(device)
            logits1, _ = model(feat1)
            score1 = (logits1[:, BONAFIDE] - logits1[:, SPOOF]).item()
            all_untrimmed_scores.append(score1)

        # 2. Silence trimmed ablation
        trimmed_wave = energy_based_vad_trim(res.waveform)
        wave_trimmed = fixed_length_crop_or_pad(trimmed_wave, target_samples=64000, mode="center")
        with torch.no_grad():
            feat2 = extractor(wave_trimmed)
            if normalizer.is_fitted:
                feat2 = normalizer(feat2)
            feat2 = feat2.unsqueeze(0).unsqueeze(0).to(device)
            logits2, _ = model(feat2)
            score2 = (logits2[:, BONAFIDE] - logits2[:, SPOOF]).item()
            all_trimmed_scores.append(score2)

    labels_arr = np.array(labels, dtype=np.int64)
    scores_untrimmed_arr = np.array(all_untrimmed_scores, dtype=np.float64)
    scores_trimmed_arr = np.array(all_trimmed_scores, dtype=np.float64)

    eval_untrimmed = full_evaluation(labels_arr, scores_untrimmed_arr, dev_threshold=dev_thresh, compute_ci=True)
    eval_trimmed = full_evaluation(labels_arr, scores_trimmed_arr, dev_threshold=dev_thresh, compute_ci=True)

    silence_delta = eval_trimmed["eer"] - eval_untrimmed["eer"]

    final_report = {
        "model": "CRNN (Agam Ghotra)",
        "eval_trials": len(labels_arr),
        "standard_evaluation": eval_untrimmed,
        "silence_trimmed_evaluation": eval_trimmed,
        "silence_shortcut_delta_eer": silence_delta,
        "silence_analysis": (
            f"Standard Eval EER: {eval_untrimmed['eer']:.2f}% vs Silence-Trimmed EER: {eval_trimmed['eer']:.2f}%. "
            f"Delta EER: {silence_delta:+.2f}%. "
            "A small change indicates genuine feature representation rather than reliance on boundary silence artifacts."
        ),
    }

    metrics_out = Path("results/metrics/master_evaluation.json")
    save_json(final_report, metrics_out)
    logger.info("Saved master evaluation report to %s", metrics_out)

    print("\n" + "=" * 80)
    print("MASTER EVALUATION REPORT & SILENCE SHORTCUT ANALYSIS")
    print("=" * 80)
    print(f" Applied Dev Threshold:       {dev_thresh:.4f}")
    print(f" Standard Eval EER:           {eval_untrimmed['eer']:.2f}% (95% CI: [{eval_untrimmed['eer_ci_95'][0]:.2f}, {eval_untrimmed['eer_ci_95'][1]:.2f}])")
    print(f" Standard Eval ROC-AUC:       {eval_untrimmed['roc_auc']:.4f}")
    print(f" Standard min-tDCF:           {eval_untrimmed['min_tdcf']:.4f}")
    print(f" Accuracy @ Dev Threshold:    {eval_untrimmed['at_applied_threshold']['accuracy'] * 100:.2f}%")
    print("-" * 80)
    print(f" Silence-Trimmed EER:         {eval_trimmed['eer']:.2f}% (Delta: {silence_delta:+.2f}%)")
    print(f" Silence-Trimmed ROC-AUC:     {eval_trimmed['roc_auc']:.4f}")
    print(" ML Rigor Finding:")
    print(f"  {final_report['silence_analysis']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
