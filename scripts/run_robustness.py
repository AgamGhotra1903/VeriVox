"""Codec Compression Stress-Testing CLI.

Author: Chunit Bansal (IIT2024223)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from acforensics.config import load_config
from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    LFCC_DEFAULT_DIM,
)
from acforensics.data.protocols import load_asvspoof_protocol
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import FeatureNormalizer
from acforensics.models.crnn import CRNNForensics
from acforensics.evaluation.robustness import (
    compress_and_decompress_audio,
    plot_robustness_curves,
    check_ffmpeg_available,
)
from acforensics.evaluation.metrics import compute_eer, full_evaluation
from acforensics.utils.io import save_json, load_json
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def evaluate_on_dataset(
    model: torch.nn.Module,
    file_list: list[tuple[str, int]],
    extractor: LFCCExtractor,
    normalizer: FeatureNormalizer,
    device: torch.device,
    dev_thresh: float,
) -> dict:
    """Evaluates model on audio file list and returns metric report."""
    all_scores = []
    all_labels = []

    for path_str, label in file_list:
        res = load_and_validate_audio(path_str, target_sr=16000)
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
        all_labels.append(label)

    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int64)

    return full_evaluation(labels_arr, scores_arr, dev_threshold=dev_thresh, compute_ci=False)


def main():
    parser = argparse.ArgumentParser(description="Codec Robustness Evaluation CLI")
    parser.add_argument("--config", type=str, default="configs/robustness.yaml", help="Path to robustness config")
    parser.add_argument("--smoke", action="store_true", help="Run on small sample for instant verification")
    parser.add_argument("--max-files", type=int, default=50, help="Number of files to compress and test")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)

    if not check_ffmpeg_available():
        logger.warning("ffmpeg is not installed or not in PATH. Codec stress test requires ffmpeg.")
        return

    # Load CRNN model
    device = torch.device("cpu")
    ckpt_path = Path("checkpoints/crnn_best.pt")
    if not ckpt_path.exists():
        logger.error("Trained CRNN checkpoint not found at %s. Run make train first.", ckpt_path)
        return

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = CRNNForensics(in_channels=1, in_freq_dim=LFCC_DEFAULT_DIM)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dev_thresh = ckpt.get("dev_threshold", 0.0)

    # Feature extractor and normalizer
    extractor = LFCCExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    norm_file = Path("checkpoints/normalization_stats.json")
    normalizer = FeatureNormalizer().load(norm_file) if norm_file.exists() else FeatureNormalizer()

    # Determine eval split
    base_dir = Path("data/smoke_test" if args.smoke else "data/raw/LA")
    proto_file = base_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
    if not proto_file.exists():
        base_dir = Path("data/smoke_test")
        proto_file = base_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"

    eval_df = load_asvspoof_protocol(proto_file, audio_dir=base_dir / "ASVspoof2019_LA_eval" / "flac", split_name="eval")
    test_subset = eval_df.head(args.max_files)

    clean_files = [(row["audio_path"], row["label"]) for _, row in test_subset.iterrows()]
    logger.info("Evaluating baseline performance on %d clean uncompressed files...", len(clean_files))
    clean_eval = evaluate_on_dataset(model, clean_files, extractor, normalizer, device, dev_thresh)
    clean_eer = clean_eval["eer"]

    codecs = ["mp3", "aac", "opus"]
    bitrates = [16, 24, 32, 48, 64] if not args.smoke else [16, 32, 64]

    cache_root = Path("data/compressed")
    results_by_codec = {c: {} for c in codecs}
    table_rows = []

    for codec in codecs:
        for kbps in bitrates:
            logger.info(">>> Processing %s @ %d kbps...", codec.upper(), kbps)
            comp_dir = cache_root / f"{codec}_{kbps}k"
            comp_files = []

            for _, row in test_subset.iterrows():
                in_path = Path(row["audio_path"])
                out_path = comp_dir / f"{in_path.stem}.wav"

                # Check cache before re-encoding
                if not out_path.exists():
                    compress_and_decompress_audio(in_path, out_path, codec=codec, bitrate_kbps=kbps)

                if out_path.exists():
                    comp_files.append((str(out_path), row["label"]))

            if len(comp_files) > 0:
                report = evaluate_on_dataset(model, comp_files, extractor, normalizer, device, dev_thresh)
                comp_eer = report["eer"]
                results_by_codec[codec][kbps] = comp_eer

                degradation_abs = comp_eer - clean_eer
                degradation_rel = (degradation_abs / (clean_eer + 1e-6)) * 100 if clean_eer > 0 else 0.0

                table_rows.append({
                    "Codec": codec.upper(),
                    "Bitrate (kbps)": kbps,
                    "Clean EER (%)": f"{clean_eer:.2f}",
                    "Compressed EER (%)": f"{comp_eer:.2f}",
                    "Abs Degradation (pts)": f"{degradation_abs:+.2f}",
                    "Accuracy (%)": f"{report['at_applied_threshold']['accuracy'] * 100:.2f}",
                })

    # Save summary table & figures
    results_dir = Path("results")
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"

    summary_df = pd.DataFrame(table_rows)
    csv_path = tables_dir / "robustness_degradation.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info("Saved robustness degradation table to %s", csv_path)

    metrics_out = metrics_dir / "robustness_results.json"
    save_json({
        "clean_baseline_eer": clean_eer,
        "results_by_codec": results_by_codec,
    }, metrics_out)

    plot_robustness_curves(results_by_codec, clean_eer, figures_dir / "robustness_eer_vs_bitrate.png")

    print("\n" + "=" * 80)
    print("CODEC COMPRESSION ROBUSTNESS BENCHMARK")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
