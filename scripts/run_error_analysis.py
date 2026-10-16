"""Forensic Error Breakdown CLI (Per-Attack, Per-Speaker, Worst-Cases).

Author: Chunit Bansal (IIT2024223)
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
from acforensics.evaluation.error_analysis import run_error_audit
from acforensics.utils.io import save_json
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Forensic Error Analysis CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run on smoke test dataset")
    parser.add_argument("--max-clips", type=int, default=100, help="Max clips to audit")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)

    device = torch.device("cpu")
    ckpt_path = Path("checkpoints/crnn_best.pt")
    if not ckpt_path.exists():
        logger.error("Model checkpoint not found at %s. Run make train first.", ckpt_path)
        return

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = CRNNForensics(in_channels=1, in_freq_dim=LFCC_DEFAULT_DIM)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dev_thresh = ckpt.get("dev_threshold", 0.0)

    # Feature extractor & normalizer
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
    if args.max_clips:
        eval_df = eval_df.head(args.max_clips)

    all_scores = []
    valid_rows = []

    logger.info("Computing predictions on %d evaluation trials for error auditing...", len(eval_df))
    for _, row in eval_df.iterrows():
        res = load_and_validate_audio(row["audio_path"], target_sr=16000)
        if not res.is_valid or res.waveform is None:
            continue
        wave = fixed_length_crop_or_pad(res.waveform, target_samples=64000, mode="center")
        with torch.no_grad():
            feat = extractor(wave)
            if normalizer.is_fitted:
                feat = normalizer(feat)
            feat = feat.unsqueeze(0).unsqueeze(0).to(device)
            logits, _ = model(feat)
            score = (logits[:, BONAFIDE] - logits[:, SPOOF]).item()

        all_scores.append(score)
        valid_rows.append(row)

    valid_df = pd.DataFrame(valid_rows)
    scores_arr = np.array(all_scores, dtype=np.float64)

    audit_summary = run_error_audit(
        eval_df=valid_df,
        scores=scores_arr,
        threshold=dev_thresh,
        output_dir="results",
    )

    metrics_out = Path("results/metrics/error_analysis_summary.json")
    save_json(audit_summary, metrics_out)
    logger.info("Saved error analysis summary to %s", metrics_out)

    print("\n" + "=" * 80)
    print("FORENSIC ERROR AUDIT SUMMARY (ASVspoof 2019 LA Eval)")
    print("=" * 80)
    print(f" Operating Decision Threshold: {dev_thresh:.4f} (Derived from Dev EER)")
    print(f" Total Evaluated Trials:       {audit_summary['total_trials']}")
    print(f" False Positives (Spoof -> Genuine): {audit_summary['false_positive_count']}")
    print(f" False Negatives (Genuine -> Spoof): {audit_summary['false_negative_count']}")
    print(f" Most Vulnerable Attack:       {audit_summary['most_vulnerable_attack']} (EER: {audit_summary['most_vulnerable_eer']:.2f}%)")
    print("\nKey Forensic Findings:")
    for f in audit_summary["plain_language_findings"]:
        print(f"  - {f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
