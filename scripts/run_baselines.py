"""Classical ML Baselines Training, Tuning, and Evaluation CLI.

Author: Jashan Goyal (IIT2024223)
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from acforensics.config import load_config
from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_RESULTS_DIR,
)
from acforensics.data.protocols import load_asvspoof_protocol
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.statistics import extract_utterance_statistics
from acforensics.models.classical import (
    train_logistic_regression,
    train_rbf_svm,
    train_random_forest,
    evaluate_classical_model,
)
from acforensics.evaluation.plots import (
    plot_roc_curve,
    plot_det_curve,
    plot_score_distributions,
    plot_confusion_matrix,
)
from acforensics.utils.io import save_json
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def extract_features_for_split(
    df: pd.DataFrame,
    lfcc_ext: LFCCExtractor,
    logmel_ext: LogMelExtractor,
    target_samples: int = 64000,
    desc: str = "Extracting Features",
) -> dict[str, np.ndarray]:
    """Extracts utterance-level statistical vectors for LFCC, Log-Mel, and Combined."""
    lfcc_list = []
    logmel_list = []
    labels_list = []
    file_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        audio_path = row["audio_path"]
        res = load_and_validate_audio(audio_path, target_sr=16000)
        if not res.is_valid or res.waveform is None:
            continue

        wave = fixed_length_crop_or_pad(res.waveform, target_samples=target_samples, mode="center")

        with torch.no_grad():
            feat_lfcc = lfcc_ext(wave)
            feat_logmel = logmel_ext(wave)

        stat_lfcc = extract_utterance_statistics(feat_lfcc)
        stat_logmel = extract_utterance_statistics(feat_logmel)

        lfcc_list.append(stat_lfcc)
        logmel_list.append(stat_logmel)
        labels_list.append(row["label"])
        file_ids.append(row["file_id"])

    X_lfcc = np.array(lfcc_list, dtype=np.float32)
    X_logmel = np.array(logmel_list, dtype=np.float32)
    X_comb = np.hstack([X_lfcc, X_logmel])
    y = np.array(labels_list, dtype=np.int64)

    return {
        "lfcc": X_lfcc,
        "logmel": X_logmel,
        "combined": X_comb,
        "labels": y,
        "file_ids": file_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Classical Baselines Training and Benchmarking CLI")
    parser.add_argument("--config", type=str, default="configs/baselines.yaml", help="Path to baselines config")
    parser.add_argument("--smoke", action="store_true", help="Run on smoke test dataset")
    parser.add_argument("--max-train", type=int, default=None, help="Limit training samples for debug")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)

    # Determine directories
    base_dir = Path("data/smoke_test" if args.smoke else "data/raw/LA")
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

    if args.max_train:
        train_df = train_df.head(args.max_train)
        dev_df = dev_df.head(args.max_train // 2)
        eval_df = eval_df.head(args.max_train // 2)

    # Feature extractors
    lfcc_ext = LFCCExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    logmel_ext = LogMelExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80)

    logger.info("Extracting classical features for train, dev, and eval...")
    train_feats = extract_features_for_split(train_df, lfcc_ext, logmel_ext, desc="Train Features")
    dev_feats = extract_features_for_split(dev_df, lfcc_ext, logmel_ext, desc="Dev Features")
    eval_feats = extract_features_for_split(eval_df, lfcc_ext, logmel_ext, desc="Eval Features")

    feature_types = ["lfcc", "logmel", "combined"]
    models_to_run = {
        "Logistic Regression": train_logistic_regression,
        "RBF-SVM": train_rbf_svm,
        "Random Forest": train_random_forest,
    }

    results = {}
    summary_rows = []
    roc_curves_dict = {}

    checkpoints_dir = Path("checkpoints/classical")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for m_name, train_fn in models_to_run.items():
        results[m_name] = {}
        for f_type in feature_types:
            logger.info(">>> Training %s on %s features...", m_name, f_type.upper())
            X_tr, y_tr = train_feats[f_type], train_feats["labels"]
            X_dv, y_dv = dev_feats[f_type], dev_feats["labels"]
            X_ev, y_ev = eval_feats[f_type], eval_feats["labels"]

            model = train_fn(X_tr, y_tr, X_dv, y_dv, seed=42)

            # Save model
            model_slug = m_name.lower().replace(" ", "_").replace("-", "_")
            joblib_path = checkpoints_dir / f"{model_slug}_{f_type}.joblib"
            joblib.dump(model, str(joblib_path))

            # Dev evaluation to calibrate threshold
            dev_scores, dev_metrics = evaluate_classical_model(model, X_dv, y_dv)
            dev_eer_thresh = dev_metrics["eer_threshold"]

            # Eval evaluation using dev threshold
            eval_scores, eval_metrics = evaluate_classical_model(model, X_ev, y_ev, dev_threshold=dev_eer_thresh)

            results[m_name][f_type] = {
                "dev": dev_metrics,
                "eval": eval_metrics,
                "best_params": model.named_steps["classifier"].get_params(),
            }

            key_name = f"{m_name} ({f_type.upper()})"
            roc_curves_dict[key_name] = (y_ev, eval_scores)

            summary_rows.append({
                "Model": m_name,
                "Features": f_type.upper(),
                "Feature Dim": X_tr.shape[1],
                "Dev EER (%)": f"{dev_metrics['eer']:.2f}",
                "Eval EER (%)": f"{eval_metrics['eer']:.2f}",
                "Eval EER 95% CI": f"[{eval_metrics['eer_ci_95'][0]:.2f}, {eval_metrics['eer_ci_95'][1]:.2f}]",
                "Eval ROC-AUC": f"{eval_metrics['roc_auc']:.4f}",
                "Eval min-tDCF": f"{eval_metrics['min_tdcf']:.4f}",
                "Applied Threshold": f"{dev_eer_thresh:.4f} (from Dev)",
                "Accuracy (%)": f"{eval_metrics['at_applied_threshold']['accuracy'] * 100:.2f}",
            })

    # Save metrics JSON & Summary CSV
    metrics_path = results_dir / "metrics" / "baselines_results.json"
    save_json(results, metrics_path)
    logger.info("Saved baseline metrics to %s", metrics_path)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = tables_dir / "baselines_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info("Saved baselines summary table to %s", csv_path)

    # Print summary table
    print("\n" + "=" * 90)
    print("CLASSICAL BASELINE BENCHMARK RESULTS (ASVspoof 2019 LA)")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90 + "\n")

    # Plot multi-model ROC & DET curves
    plot_roc_curve(roc_curves_dict, figures_dir / "baselines_roc.png", title="Classical Baselines ROC on Eval Split")
    plot_det_curve(roc_curves_dict, figures_dir / "baselines_det.png", title="Classical Baselines DET on Eval Split")


if __name__ == "__main__":
    main()
