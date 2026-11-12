"""CRNN Training and Benchmark CLI: LFCC & Log-Mel Variants.

Features:
- Configurable feature front-end: LFCC (60-dim) or Log-Mel (80-dim).
- Adam / AdamW with weight decay.
- ReduceLROnPlateau (dev EER) or Cosine Annealing scheduler.
- Gradient clipping at 5.0.
- Early stopping monitoring Dev EER with patience=5 and best checkpoint restoration.
- Class-weighted cross entropy handling ~8:1 spoof:bonafide imbalance.
- Multi-seed training support reporting mean and std.
- Deterministic seed logging and per-run JSON logging.
- Strict data leakage boundary: eval split is never used for training or model selection.

"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Optional
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    LFCC_DEFAULT_DIM,
    LOGMEL_DEFAULT_DIM,
    TARGET_SAMPLES,
    SAMPLE_RATE,
)
from acforensics.data.protocols import load_asvspoof_protocol
from acforensics.data.datasets import ASVspoofDataset, collate_fn
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.normalization import FeatureNormalizer, per_utterance_cmvn
from acforensics.features.cache import FeatureCache
from acforensics.models.crnn import CRNNForensics
from acforensics.training.train_crnn import train_crnn
from acforensics.evaluation.metrics import full_evaluation
from acforensics.utils.io import save_json, load_json
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed
from acforensics.utils.timing import Timer

logger = get_logger(__name__)


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_feature_extractor_and_dim(
    feature_type: str,
    feature_cfg: dict[str, Any],
) -> tuple[torch.nn.Module, int]:
    """Instantiates the feature extraction module and determines input frequency dimension."""
    if feature_type == "lfcc":
        lfcc_params = feature_cfg.get("lfcc", {})
        extractor = LFCCExtractor(
            sample_rate=lfcc_params.get("sample_rate", SAMPLE_RATE),
            n_fft=lfcc_params.get("n_fft", 512),
            win_length=lfcc_params.get("win_length", 400),
            hop_length=lfcc_params.get("hop_length", 160),
            n_filter=lfcc_params.get("n_filter", 20),
            n_lfcc=lfcc_params.get("n_lfcc", 20),
            with_deltas=lfcc_params.get("with_deltas", True),
        )
        in_dim = LFCC_DEFAULT_DIM
    elif feature_type == "logmel":
        mel_params = feature_cfg.get("logmel", {})
        extractor = LogMelExtractor(
            sample_rate=mel_params.get("sample_rate", SAMPLE_RATE),
            n_fft=mel_params.get("n_fft", 512),
            win_length=mel_params.get("win_length", 400),
            hop_length=mel_params.get("hop_length", 160),
            n_mels=mel_params.get("n_mels", 80),
            f_min=mel_params.get("f_min", 0.0),
            f_max=mel_params.get("f_max", 8000.0),
            power=mel_params.get("power", 2.0),
            top_db=mel_params.get("top_db", 80.0),
        )
        in_dim = LOGMEL_DEFAULT_DIM
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    return extractor, in_dim


def run_single_crnn_training(
    cfg: dict[str, Any],
    feature_type: str,
    seed: int,
    smoke: bool = False,
    override_epochs: Optional[int] = None,
    override_device: str = "auto",
    ckpt_name: Optional[str] = None,
) -> dict[str, Any]:
    if not smoke and Path(f"data/cache/{feature_type}_train.npz").exists():
        from scripts.train_crnn_mps import run_training
        tr_cfg = cfg.get("training", {})
        ep = override_epochs if override_epochs is not None else tr_cfg.get("epochs", 15)
        return run_training(
            feature_type=feature_type,
            seed=seed,
            epochs=ep,
            batch_size=tr_cfg.get("batch_size", 32),
            lr=tr_cfg.get("learning_rate", 1e-3),
            weight_decay=tr_cfg.get("weight_decay", 1e-4),
            patience=tr_cfg.get("early_stopping", {}).get("patience", 3),
            device_name=override_device,
        )
    set_seed(seed=seed, deterministic=True)

    if override_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(override_device)
    logger.info("Using device: %s for training", device)

    base_dir = Path("data/smoke_test" if smoke else "data/raw/LA")
    if not (base_dir / "ASVspoof2019_LA_cm_protocols").exists():
        logger.warning("Dataset not found at %s. Falling back to data/smoke_test.", base_dir)
        base_dir = Path("data/smoke_test")

    proto_dir = base_dir / "ASVspoof2019_LA_cm_protocols"
    train_proto = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_proto = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_proto = proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"

    train_audio_dir = base_dir / "ASVspoof2019_LA_train" / "flac"
    dev_audio_dir = base_dir / "ASVspoof2019_LA_dev" / "flac"
    eval_audio_dir = base_dir / "ASVspoof2019_LA_eval" / "flac"

    train_df = load_asvspoof_protocol(train_proto, audio_dir=train_audio_dir, split_name="train")
    dev_df = load_asvspoof_protocol(dev_proto, audio_dir=dev_audio_dir, split_name="dev")
    eval_df = load_asvspoof_protocol(eval_proto, audio_dir=eval_audio_dir, split_name="eval")

    train_ids = set(train_df["file_id"])
    dev_ids = set(dev_df["file_id"])
    eval_ids = set(eval_df["file_id"])
    assert len(train_ids.intersection(dev_ids)) == 0, "DATA LEAKAGE: Overlapping IDs between train and dev!"
    assert len(train_ids.intersection(eval_ids)) == 0, "DATA LEAKAGE: Overlapping IDs between train and eval!"
    assert len(dev_ids.intersection(eval_ids)) == 0, "DATA LEAKAGE: Overlapping IDs between dev and eval!"

    feature_cfg = cfg.get("features", {})
    feat_extractor, in_dim = build_feature_extractor_and_dim(feature_type, feature_cfg)
    cache = FeatureCache(enabled=not smoke)

    norm_stats_file = Path(f"checkpoints/normalization_stats_{feature_type}.json")
    if norm_stats_file.exists():
        normalizer = FeatureNormalizer().load(norm_stats_file)
    elif Path("checkpoints/normalization_stats.json").exists() and feature_type == "lfcc":
        normalizer = FeatureNormalizer().load("checkpoints/normalization_stats.json")
    else:
        normalizer = FeatureNormalizer()

    def transform_fn(waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = feat_extractor(waveform)
            if normalizer.is_fitted:
                feat = normalizer(feat)
            else:
                feat = per_utterance_cmvn(feat)
            return feat.unsqueeze(0)

    training_cfg = cfg.get("training", {})
    batch_size = 16 if smoke else training_cfg.get("batch_size", 32)
    epochs = override_epochs if override_epochs is not None else (2 if smoke else training_cfg.get("epochs", 40))

    train_dataset = ASVspoofDataset(train_df, split="train", crop_mode="random", feature_transform=transform_fn)
    dev_dataset = ASVspoofDataset(dev_df, split="dev", crop_mode="center", feature_transform=transform_fn)
    eval_dataset = ASVspoofDataset(eval_df, split="eval", crop_mode="center", feature_transform=transform_fn)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model_cfg = cfg.get("model", {})
    gru_cfg = model_cfg.get("gru", {}) if isinstance(model_cfg.get("gru"), dict) else {}
    cnn_cfg = model_cfg.get("cnn", {}) if isinstance(model_cfg.get("cnn"), dict) else {}
    attn_cfg = model_cfg.get("attention", {}) if isinstance(model_cfg.get("attention"), dict) else {}
    clf_cfg = model_cfg.get("classifier", {}) if isinstance(model_cfg.get("classifier"), dict) else {}

    gru_hidden = model_cfg.get("gru_hidden") or gru_cfg.get("hidden_size", 128)
    num_gru_layers = model_cfg.get("num_gru_layers") or gru_cfg.get("num_layers", 2)
    gru_dropout = model_cfg.get("gru_dropout") if "gru_dropout" in model_cfg else gru_cfg.get("dropout", 0.3)
    conv_dropout = model_cfg.get("conv_dropout") if "conv_dropout" in model_cfg else cnn_cfg.get("dropout", 0.1)
    classifier_dropout = model_cfg.get("classifier_dropout") if "classifier_dropout" in model_cfg else clf_cfg.get("dropout", 0.3)
    num_attn_heads = model_cfg.get("num_attn_heads") or attn_cfg.get("num_heads", 4)

    model = CRNNForensics(
        in_channels=1,
        in_freq_dim=in_dim,
        features=feature_type,
        use_gru=model_cfg.get("use_gru", True),
        use_attention=model_cfg.get("use_attention", True),
        pooling=model_cfg.get("pooling", "attention"),
        proj_dim=model_cfg.get("proj_dim", 128),
        gru_hidden=gru_hidden,
        num_gru_layers=num_gru_layers,
        gru_dropout=gru_dropout,
        conv_dropout=conv_dropout,
        classifier_dropout=classifier_dropout,
        num_attn_heads=num_attn_heads,
        num_classes=model_cfg.get("num_classes", 2),
        init_weights=model_cfg.get("init_weights", True),
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("CRNN (%s) initialized with %d trainable parameters.", feature_type.upper(), param_count)

    checkpoint_filename = ckpt_name or f"crnn_{feature_type}_best.pt"

    with Timer() as timer:
        trained_model, calibrator, summary = train_crnn(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            epochs=epochs,
            lr=training_cfg.get("learning_rate", 1e-3),
            weight_decay=training_cfg.get("weight_decay", 1e-4),
            optimizer_type=training_cfg.get("optimizer", "adamw"),
            scheduler_type=training_cfg.get("lr_scheduler", "plateau"),
            seed=seed,
            device=device,
            early_stopping_patience=training_cfg.get("early_stopping", {}).get("patience", 5),
            grad_clip_norm=training_cfg.get("grad_clip_norm", 5.0),
            use_amp=(training_cfg.get("amp", True) and device.type == "cuda"),
            use_spec_augment=training_cfg.get("augmentation", {}).get("spec_augment", True),
            checkpoint_dir="checkpoints",
            results_dir="results",
            model_config=model_cfg,
            feature_config={"feature_type": feature_type, "dim": in_dim},
            norm_stats=load_json(norm_stats_file) if norm_stats_file.exists() else None,
            checkpoint_name=checkpoint_filename,
        )

    training_time_sec = timer.get_elapsed_sec()

    logger.info("Evaluating best checkpoint on frozen eval split using Dev threshold...")
    dev_threshold = summary["best_dev_threshold"]
    all_eval_scores = []
    all_eval_labels = []

    trained_model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            feats = batch["feature"].to(device)
            labels = batch["label"].to(device)
            scores = trained_model.predict_score(feats).cpu().numpy()
            all_eval_scores.extend(scores)
            all_eval_labels.extend(labels.cpu().numpy())

    eval_scores_arr = np.array(all_eval_scores, dtype=np.float64)
    eval_labels_arr = np.array(all_eval_labels, dtype=np.int64)

    asv_eval_score_file = base_dir / "ASVspoof2019_LA_asv_scores" / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    asv_path_str = str(asv_eval_score_file) if asv_eval_score_file.exists() else None

    eval_report = full_evaluation(
        eval_labels_arr,
        eval_scores_arr,
        dev_threshold=dev_threshold,
        asv_score_file=asv_path_str,
        compute_ci=True,
    )

    run_result = {
        "model": f"CRNN-{feature_type.upper()}",
        "features": feature_type,
        "seed": seed,
        "parameters": param_count,
        "training_time_sec": training_time_sec,
        "checkpoint_path": str(Path("checkpoints") / checkpoint_filename),
        "dev": {
            "best_epoch": summary["best_epoch"],
            "eer": summary["best_dev_eer"],
            "threshold": dev_threshold,
        },
        "eval": eval_report,
    }

    results_out = Path("results/metrics") / f"crnn_{feature_type}_results.json"
    results_out.parent.mkdir(parents=True, exist_ok=True)
    save_json(run_result, results_out)

    return run_result


def main():
    parser = argparse.ArgumentParser(description="CRNN Training and Evaluation CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--features", type=str, choices=["lfcc", "logmel"], default=None, help="Feature type")
    parser.add_argument("--smoke", action="store_true", help="Run 2-epoch smoke test on mock data")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Primary RNG seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of seeds for multi-seed run")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cuda', 'mps', 'cpu', 'auto')")
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Checkpoint filename")
    args = parser.parse_args()

    setup_logging()

    if args.config is not None:
        cfg_path = Path(args.config)
    elif args.features == "logmel":
        cfg_path = Path("configs/crnn_logmel.yaml")
    else:
        cfg_path = Path("configs/crnn_lfcc.yaml")

    if not cfg_path.exists():
        cfg_path = Path("configs/crnn.yaml")

    cfg = load_yaml_config(cfg_path)
    feature_type = args.features or cfg.get("features", {}).get("feature_type", "lfcc")

    seeds = args.seeds if args.seeds is not None else [args.seed]
    all_runs = []

    print("\n" + "=" * 80)
    print(f"STARTING CRNN TRAINING: {feature_type.upper()} FRONT-END | SEEDS: {seeds}")
    print("=" * 80 + "\n")

    for s in seeds:
        logger.info(">>> Commencing Run with Seed %d ...", s)
        ckpt_name = args.checkpoint_name or (
            f"crnn_{feature_type}_seed{s}.pt" if len(seeds) > 1 else f"crnn_{feature_type}_best.pt"
        )
        res = run_single_crnn_training(
            cfg=cfg,
            feature_type=feature_type,
            seed=s,
            smoke=args.smoke,
            override_epochs=args.epochs,
            override_device=args.device,
            ckpt_name=ckpt_name,
        )
        all_runs.append(res)

        print("-" * 60)
        print(f"Seed {s} Results: Dev EER = {res['dev']['eer']:.2f}% | Eval EER = {res['eval']['eer']:.2f}% | Eval min-tDCF = {res['eval']['min_tdcf']:.4f}")
        print("-" * 60)

    if len(all_runs) > 1:
        dev_eers = [r["dev"]["eer"] for r in all_runs]
        eval_eers = [r["eval"]["eer"] for r in all_runs]
        eval_aucs = [r["eval"]["roc_auc"] for r in all_runs]
        eval_tdcfs = [r["eval"]["min_tdcf"] for r in all_runs]

        print("\n" + "=" * 80)
        print(f"MULTI-SEED AGGREGATE SUMMARY ({len(all_runs)} Seeds for CRNN-{feature_type.upper()}):")
        print("=" * 80)
        print(f" Dev EER:       {np.mean(dev_eers):.2f}% +/- {np.std(dev_eers):.2f}%")
        print(f" Eval EER:      {np.mean(eval_eers):.2f}% +/- {np.std(eval_eers):.2f}%")
        print(f" Eval ROC-AUC:  {np.mean(eval_aucs):.4f} +/- {np.std(eval_aucs):.4f}")
        print(f" Eval min-tDCF: {np.mean(eval_tdcfs):.4f} +/- {np.std(eval_tdcfs):.4f}")
        print("=" * 80 + "\n")

        multi_seed_summary = {
            "model": f"CRNN-{feature_type.upper()}",
            "feature_type": feature_type,
            "seeds": seeds,
            "metrics": {
                "dev_eer_mean": float(np.mean(dev_eers)),
                "dev_eer_std": float(np.std(dev_eers)),
                "eval_eer_mean": float(np.mean(eval_eers)),
                "eval_eer_std": float(np.std(eval_eers)),
                "eval_auc_mean": float(np.mean(eval_aucs)),
                "eval_auc_std": float(np.std(eval_aucs)),
                "eval_tdcf_mean": float(np.mean(eval_tdcfs)),
                "eval_tdcf_std": float(np.std(eval_tdcfs)),
            },
            "individual_runs": all_runs,
        }
        multi_out = Path("results/metrics") / f"crnn_{feature_type}_multiseed_summary.json"
        save_json(multi_seed_summary, multi_out)
        logger.info("Saved multi-seed summary to %s", multi_out)


if __name__ == "__main__":
    main()
