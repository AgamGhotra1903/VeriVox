"""CRNN Model Training and Evaluation CLI.

Author: Agam Ghotra (IIT2024226)
"""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from acforensics.config import load_config
from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    LFCC_DEFAULT_DIM,
    LOGMEL_DEFAULT_DIM,
)
from acforensics.data.protocols import load_asvspoof_protocol
from acforensics.data.datasets import ASVspoofDataset, collate_fn
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.normalization import FeatureNormalizer, per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics
from acforensics.models.calibration import PlattCalibrator
from acforensics.training.train_crnn import train_crnn
from acforensics.evaluation.metrics import full_evaluation
from acforensics.utils.io import save_json, load_json
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed
from acforensics.utils.timing import Timer

logger = get_logger(__name__)


def measure_inference_latency(model: torch.nn.Module, feature_dim: int, device: torch.device, n_trials: int = 50) -> float:
    """Measures average inference latency in milliseconds for a 4.0s clip (400 frames)."""
    model.eval()
    dummy = torch.randn(1, 1, feature_dim, 400).to(device)
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_trials):
            _ = model(dummy)
    elapsed = time.perf_counter() - start
    return (elapsed / n_trials) * 1000.0


def main():
    parser = argparse.ArgumentParser(description="CRNN Training and Benchmark CLI")
    parser.add_argument("--config", type=str, default="configs/crnn.yaml", help="Path to CRNN config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run quick 2-epoch smoke test on mock data")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('cuda', 'mps', 'cpu', 'auto')")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    master_config = load_config("configs/default.yaml")

    # Resolve device
    if args.device == "auto":
        device = master_config.get_resolved_device()
    else:
        device = torch.device(args.device)

    # Determine directories
    base_dir = Path("data/smoke_test" if args.smoke else master_config.data.asvspoof_root)
    if not (base_dir / "ASVspoof2019_LA_cm_protocols").exists():
        logger.warning("Dataset not found at %s. Falling back to data/smoke_test.", base_dir)
        base_dir = Path("data/smoke_test")

    proto_dir = base_dir / "ASVspoof2019_LA_cm_protocols"
    train_proto = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_proto = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_proto = proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"

    train_df = load_asvspoof_protocol(train_proto, audio_dir=base_dir / "ASVspoof2019_LA_train" / "flac", split_name="train")
    dev_df = load_asvspoof_protocol(dev_proto, audio_dir=base_dir / "ASVspoof2019_LA_dev" / "flac", split_name="dev")
    eval_df = load_asvspoof_protocol(eval_proto, audio_dir=base_dir / "ASVspoof2019_LA_eval" / "flac", split_name="eval")

    epochs = args.epochs if args.epochs is not None else (2 if args.smoke else 40)
    batch_size = 16 if args.smoke else 32

    # Feature extractor & normalizer
    feature_type = master_config.features.feature_type
    if feature_type == "lfcc":
        feat_extractor = LFCCExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
        in_dim = LFCC_DEFAULT_DIM
    else:
        feat_extractor = LogMelExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80)
        in_dim = LOGMEL_DEFAULT_DIM

    norm_stats_file = Path(master_config.features.stats_file)
    if norm_stats_file.exists():
        normalizer = FeatureNormalizer().load(norm_stats_file)
    else:
        normalizer = FeatureNormalizer()

    def transform_fn(waveform: torch.Tensor) -> torch.Tensor:
        feat = feat_extractor(waveform)
        if normalizer.is_fitted:
            feat = normalizer(feat)
        else:
            feat = per_utterance_cmvn(feat)
        return feat.unsqueeze(0)  # (1, F, T)

    train_dataset = ASVspoofDataset(train_df, split="train", crop_mode="random", feature_transform=transform_fn)
    dev_dataset = ASVspoofDataset(dev_df, split="dev", crop_mode="center", feature_transform=transform_fn)
    eval_dataset = ASVspoofDataset(eval_df, split="eval", crop_mode="center", feature_transform=transform_fn)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = CRNNForensics(in_channels=1, in_freq_dim=in_dim, features=feature_type)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("CRNN initialized with %d trainable parameters.", param_count)

    with Timer() as train_timer:
        trained_model, calibrator, summary = train_crnn(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            epochs=epochs,
            lr=1e-3,
            device=device,
            use_amp=(device.type == "cuda"),
            use_spec_augment=True,
            checkpoint_dir="checkpoints",
            results_dir="results",
            norm_stats=load_json(norm_stats_file) if norm_stats_file.exists() else None,
        )

    training_time_sec = train_timer.get_elapsed_sec()

    # Eval split evaluation using Dev EER threshold
    trained_model.eval()
    dev_threshold = summary["best_dev_threshold"]
    all_eval_scores = []
    all_eval_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            feats = batch["feature"].to(device)
            labels = batch["label"].to(device)
            logits, _ = trained_model(feats)
            scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
            all_eval_scores.extend(scores)
            all_eval_labels.extend(labels.cpu().numpy())

    eval_scores_arr = np.array(all_eval_scores)
    eval_labels_arr = np.array(all_eval_labels)

    eval_report = full_evaluation(eval_labels_arr, eval_scores_arr, dev_threshold=dev_threshold, compute_ci=True)

    # Latency measurement
    cpu_latency = measure_inference_latency(trained_model.cpu(), in_dim, torch.device("cpu"))
    gpu_latency = measure_inference_latency(trained_model.to(device), in_dim, device) if device.type != "cpu" else cpu_latency

    final_results = {
        "model": "CRNN",
        "parameters": param_count,
        "training_time_sec": training_time_sec,
        "inference_latency_ms": {
            "cpu": cpu_latency,
            "accelerator": gpu_latency,
            "device": str(device),
        },
        "dev": {
            "eer": summary["best_dev_eer"],
            "threshold": dev_threshold,
        },
        "eval": eval_report,
    }

    metrics_out = Path("results/metrics/crnn_results.json")
    save_json(final_results, metrics_out)
    logger.info("Saved final CRNN evaluation results to %s", metrics_out)

    print("\n" + "=" * 70)
    print("CRNN MODEL EVALUATION RESULTS (ASVspoof 2019 LA)")
    print("=" * 70)
    print(f" Trainable Parameters:    {param_count:,}")
    print(f" Total Training Time:     {training_time_sec:.2f} s")
    print(f" Inference Latency (CPU): {cpu_latency:.2f} ms / 4s clip")
    if device.type != "cpu":
        print(f" Inference Latency ({device}): {gpu_latency:.2f} ms / 4s clip")
    print(f" Best Dev EER:            {summary['best_dev_eer']:.2f}% (Epoch {summary['best_epoch']})")
    print(f" Applied Dev Threshold:   {dev_threshold:.4f}")
    print(f" Eval Split EER:          {eval_report['eer']:.2f}% (95% CI: [{eval_report['eer_ci_95'][0]:.2f}, {eval_report['eer_ci_95'][1]:.2f}])")
    print(f" Eval ROC-AUC:            {eval_report['roc_auc']:.4f}")
    print(f" Eval min-tDCF:           {eval_report['min_tdcf']:.4f}")
    print(f" Accuracy @ Dev Thresh:   {eval_report['at_applied_threshold']['accuracy'] * 100:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
