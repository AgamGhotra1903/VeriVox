"""Classical ML Baselines Training CLI: GMM, Logistic Regression, RBF-SVM, Random Forest.

Features:
- Official ASVspoof 2019 GMM baseline (64 diagonal components, EM on LFCC frames).
- StandardScaler + PCA(95% variance) fitted strictly on TRAIN data only.
- Mean + Std utterance statistical pooling across spectral frames.
- Consistent continuous score sign convention (higher score = more bona fide) across all models.
- Hyperparameter tuning strictly on DEV split; EVAL split is evaluated frozen.
- Supports --smoke mode for fast end-to-end verification.

"""

import argparse
from pathlib import Path
import sys
from typing import Any, Optional
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_RESULTS_DIR,
)
from acforensics.data.protocols import load_asvspoof_protocol
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.logmel import LogMelExtractor
from acforensics.models.gmm import GMMForensics
from acforensics.models.classical import (
    extract_mean_std_features,
    predict_classical_score,
    train_logistic_regression,
    train_rbf_svm,
    train_random_forest,
    evaluate_classical_model,
)
from acforensics.evaluation.metrics import full_evaluation
from acforensics.evaluation.plots import plot_roc_curve, plot_det_curve
from acforensics.utils.io import save_json
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def extract_classical_data(
    df: pd.DataFrame,
    lfcc_ext: LFCCExtractor,
    logmel_ext: LogMelExtractor,
    target_samples: int = TARGET_SAMPLES,
    desc: str = "Extracting Features",
) -> dict[str, Any]:
    """Extracts frame features and pooled mean+std vectors for all utterances."""
    lfcc_frames_list = []
    logmel_frames_list = []
    lfcc_vecs = []
    logmel_vecs = []
    labels = []
    file_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        audio_path = row["audio_path"]
        res = load_and_validate_audio(audio_path, target_sr=SAMPLE_RATE)
        if not res.is_valid or res.waveform is None:
            logger.warning("Skipping or zeroing corrupt audio: %s", audio_path)
            wave = torch.zeros(target_samples, dtype=torch.float32)
        else:
            wave = fixed_length_crop_or_pad(res.waveform, target_samples=target_samples, mode="center")

        with torch.no_grad():
            f_lfcc = lfcc_ext(wave).cpu().numpy().astype(np.float64)
            f_logmel = logmel_ext(wave).cpu().numpy().astype(np.float64)

        lfcc_frames_list.append(f_lfcc.T)
        logmel_frames_list.append(f_logmel.T)

        lfcc_vecs.append(extract_mean_std_features(f_lfcc))
        logmel_vecs.append(extract_mean_std_features(f_logmel))

        labels.append(int(row["label"]))
        file_ids.append(str(row["file_id"]))

    return {
        "lfcc_frames": lfcc_frames_list,
        "logmel_frames": logmel_frames_list,
        "lfcc_vecs": np.array(lfcc_vecs, dtype=np.float64),
        "logmel_vecs": np.array(logmel_vecs, dtype=np.float64),
        "labels": np.array(labels, dtype=np.int64),
        "file_ids": file_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Classical Baselines Training CLI")
    parser.add_argument("--config", type=str, default="configs/baselines.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run in quick smoke test mode")
    parser.add_argument("--pca-variance", type=float, default=0.95, help="Variance retained by PCA")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed, deterministic=True)

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

    train_ids = set(train_df["file_id"])
    dev_ids = set(dev_df["file_id"])
    eval_ids = set(eval_df["file_id"])
    assert len(train_ids.intersection(dev_ids)) == 0, "DATA LEAKAGE: Overlapping IDs between train and dev!"
    assert len(train_ids.intersection(eval_ids)) == 0, "DATA LEAKAGE: Overlapping IDs between train and eval!"
    assert len(dev_ids.intersection(eval_ids)) == 0, "DATA LEAKAGE: Overlapping IDs between dev and eval!"

    lfcc_ext = LFCCExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    logmel_ext = LogMelExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_mels=80)

    logger.info("Extracting classical features for train, dev, and eval...")
    train_data = extract_classical_data(train_df, lfcc_ext, logmel_ext, desc="Train Features")
    dev_data = extract_classical_data(dev_df, lfcc_ext, logmel_ext, desc="Dev Features")
    eval_data = extract_classical_data(eval_df, lfcc_ext, logmel_ext, desc="Eval Features")

    checkpoints_dir = Path("checkpoints/classical")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(DEFAULT_RESULTS_DIR)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    summary_rows = []
    roc_curves = {}

    asv_score_file = base_dir / "ASVspoof2019_LA_asv_scores" / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    asv_path_str = str(asv_score_file) if asv_score_file.exists() else None

    logger.info(">>> Training GMM Baseline on LFCC frames...")
    n_components = 16 if args.smoke else 64
    gmm_model = GMMForensics(n_components=n_components, covariance_type="diag", max_iter=100, random_state=args.seed)
    gmm_model.fit(train_data["lfcc_frames"], train_data["labels"])

    gmm_path = checkpoints_dir / "gmm_lfcc.joblib"
    gmm_model.save(gmm_path)

    dev_llr_scores = gmm_model.predict_score(dev_data["lfcc_frames"])
    dev_gmm_metrics = full_evaluation(dev_data["labels"], dev_llr_scores)
    dev_gmm_thresh = dev_gmm_metrics["eer_threshold"]

    eval_llr_scores = gmm_model.predict_score(eval_data["lfcc_frames"])
    eval_gmm_metrics = full_evaluation(
        eval_data["labels"],
        eval_llr_scores,
        dev_threshold=dev_gmm_thresh,
        asv_score_file=asv_path_str,
    )

    results["GMM"] = {
        "lfcc": {
            "dev": dev_gmm_metrics,
            "eval": eval_gmm_metrics,
        }
    }
    roc_curves["GMM (LFCC)"] = (eval_data["labels"], eval_llr_scores)

    summary_rows.append({
        "Model": "GMM (Baseline)",
        "Features": "LFCC",
        "Feature Dim": "60 (Frames)",
        "Dev EER (%)": f"{dev_gmm_metrics['eer']:.2f}",
        "Eval EER (%)": f"{eval_gmm_metrics['eer']:.2f}",
        "Eval ROC-AUC": f"{eval_gmm_metrics['roc_auc']:.4f}",
        "Eval min-tDCF": f"{eval_gmm_metrics['min_tdcf']:.4f}",
        "Applied Threshold": f"{dev_gmm_thresh:.4f} (from Dev)",
    })

    classical_models = {
        "Logistic Regression": train_logistic_regression,
        "RBF-SVM": train_rbf_svm,
        "Random Forest": train_random_forest,
    }
    feature_sets = {
        "lfcc": (train_data["lfcc_vecs"], dev_data["lfcc_vecs"], eval_data["lfcc_vecs"]),
        "logmel": (train_data["logmel_vecs"], dev_data["logmel_vecs"], eval_data["logmel_vecs"]),
    }

    for m_name, train_fn in classical_models.items():
        results[m_name] = {}
        for f_name, (X_tr, X_dv, X_ev) in feature_sets.items():
            logger.info(">>> Training %s on %s features with PCA...", m_name, f_name.upper())
            model_pipe = train_fn(
                X_tr,
                train_data["labels"],
                X_dv,
                dev_data["labels"],
                pca_variance=args.pca_variance,
                seed=args.seed,
            )

            model_slug = m_name.lower().replace(" ", "_").replace("-", "_")
            joblib_path = checkpoints_dir / f"{model_slug}_{f_name}.joblib"
            joblib.dump(model_pipe, str(joblib_path))

            dev_scores, dev_metrics = evaluate_classical_model(model_pipe, X_dv, dev_data["labels"])
            dev_thresh = dev_metrics["eer_threshold"]

            eval_scores, eval_metrics = evaluate_classical_model(
                model_pipe,
                X_ev,
                eval_data["labels"],
                dev_threshold=dev_thresh,
                asv_score_file=asv_path_str,
            )

            pca_step = model_pipe.named_steps["pca"]
            n_components_kept = int(pca_step.n_components_)

            results[m_name][f_name] = {
                "dev": dev_metrics,
                "eval": eval_metrics,
                "pca_components_kept": n_components_kept,
                "pca_explained_variance": float(np.sum(pca_step.explained_variance_ratio_)),
            }

            key_name = f"{m_name} ({f_name.upper()})"
            roc_curves[key_name] = (eval_data["labels"], eval_scores)

            summary_rows.append({
                "Model": m_name,
                "Features": f_name.upper(),
                "Feature Dim": f"{X_tr.shape[1]} -> PCA({n_components_kept})",
                "Dev EER (%)": f"{dev_metrics['eer']:.2f}",
                "Eval EER (%)": f"{eval_metrics['eer']:.2f}",
                "Eval ROC-AUC": f"{eval_metrics['roc_auc']:.4f}",
                "Eval min-tDCF": f"{eval_metrics['min_tdcf']:.4f}",
                "Applied Threshold": f"{dev_thresh:.4f} (from Dev)",
            })

    metrics_path = results_dir / "metrics" / "baselines_results.json"
    save_json(results, metrics_path)
    logger.info("Saved baseline metrics to %s", metrics_path)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = tables_dir / "baselines_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info("Saved baselines summary table to %s", csv_path)

    print("\n" + "=" * 95)
    print("CLASSICAL BASELINES & GMM BENCHMARK RESULTS (ASVspoof 2019 LA)")
    print("=" * 95)
    print(summary_df.to_string(index=False))
    print("=" * 95 + "\n")

    plot_roc_curve(roc_curves, figures_dir / "baselines_roc.png", title="Classical Baselines ROC on Eval Split")
    plot_det_curve(roc_curves, figures_dir / "baselines_det.png", title="Classical Baselines DET on Eval Split")


if __name__ == "__main__":
    main()
