"""In-The-Wild Cross-Domain Evaluation CLI.

Author: Chunit Bansal (IIT2024223)
"""

import argparse
from pathlib import Path
import torch

from acforensics.constants import LFCC_DEFAULT_DIM, DEFAULT_RESULTS_DIR
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import FeatureNormalizer
from acforensics.models.crnn import CRNNForensics
from acforensics.evaluation.crossdomain import evaluate_in_the_wild
from acforensics.utils.io import save_json
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="In-The-Wild Cross-Domain Evaluation CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--meta", type=str, default="data/smoke_test/in_the_wild/meta.csv", help="Path to meta.csv")
    parser.add_argument("--audio-dir", type=str, default="data/smoke_test/in_the_wild/audio", help="Audio directory")
    parser.add_argument("--max-clips", type=int, default=50, help="Max clips to evaluate")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)

    meta_path = Path(args.meta)
    if not meta_path.exists():
        # Fallback to smoke test in the wild
        meta_path = Path("data/smoke_test/in_the_wild/meta.csv")
        audio_dir = Path("data/smoke_test/in_the_wild/audio")
    else:
        audio_dir = Path(args.audio_dir)

    if not meta_path.exists():
        logger.error("In-The-Wild metadata not found at %s. Run python scripts/prepare_data.py --generate-smoke-data first.", meta_path)
        return

    # Load CRNN model
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

    def feature_fn(wave: torch.Tensor) -> torch.Tensor:
        feat = extractor(wave)
        if normalizer.is_fitted:
            feat = normalizer(feat)
        return feat

    results = evaluate_in_the_wild(
        model=model,
        meta_csv=meta_path,
        audio_dir=audio_dir,
        dev_threshold=dev_thresh,
        feature_extractor=feature_fn,
        device=device,
        max_clips=args.max_clips,
    )

    metrics_out = Path("results/metrics/crossdomain_results.json")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, metrics_out)
    logger.info("Saved cross-domain evaluation results to %s", metrics_out)

    print("\n" + "=" * 80)
    print("IN-THE-WILD CROSS-DOMAIN ZERO-SHOT EVALUATION")
    print("=" * 80)
    print(f" Evaluated Clips:             {results['num_evaluated_clips']}")
    print(f" Genuine Clips:               {results['bonafide_count']}")
    print(f" Spoofed / Deepfake Clips:    {results['spoof_count']}")
    print(f" In-Domain Optimal EER:       {results['optimal_threshold_metrics']['itw_optimal_eer']:.2f}%")
    print(f" In-Domain Optimal Threshold: {results['optimal_threshold_metrics']['itw_optimal_threshold']:.4f}")
    print(f" In-Domain ROC-AUC:           {results['optimal_threshold_metrics']['roc_auc']:.4f}")
    print(f" Applied Dev Threshold:       {results['dev_threshold_metrics']['asvspoof_dev_threshold']:.4f} (from ASVspoof Dev)")
    print(f" Cross-Domain Accuracy:       {results['dev_threshold_metrics']['accuracy'] * 100:.2f}%")
    print("\nDomain Gap Analysis:")
    print(f"  {results['domain_shift_analysis']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
