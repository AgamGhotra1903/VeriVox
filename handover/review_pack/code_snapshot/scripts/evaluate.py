"""Master Evaluation and Ablation Suite CLI.

Generates the complete set of forensic evaluation deliverables into results/:
1. Table of EER, AUC, and min-tDCF on dev and eval for:
   - Each single model (CRNN-LFCC, CRNN-LogMel, GMM, SVM, Random Forest, Logistic Regression)
   - Baseline mean fusion (unweighted mean of dev z-scores)
   - Stacked ensemble (L2 Logistic Regression meta-learner)
2. Feature Ablation: LFCC vs. Log-Mel vs. Both (Score/Combined Fusion)
3. Architecture Ablation: CNN vs. CNN+BiGRU vs. CNN+BiGRU+Attention
4. Regularization Ablation: Regularized vs. Unregularized (Dropout & Weight Decay)
5. Meta-Learner Coefficient Table (Base model contributions)
6. Multi-Model ROC & DET curves, and Stacked Ensemble Confusion Matrix (PNG & SVG)
7. Per-Attack-Type EER on eval (A07 - A19)
8. Codec Robustness Stress-Testing on Frozen Ensemble (MP3, AAC, Opus @ 16-64 kbps)
9. Cross-Domain Zero-Shot Evaluation on In-The-Wild Deepfakes
10. Multi-Seed Variance Analysis (Mean +/- Std)
11. Silence Shortcut Bias Audit

Author: Agam Ghotra (IIT2024226)
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Optional
import warnings

# Suppress macOS Accelerate BLAS false positive FP exception flags
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    SAMPLE_RATE,
    LFCC_DEFAULT_DIM,
    LOGMEL_DEFAULT_DIM,
    DEFAULT_RESULTS_DIR,
    DEFAULT_CHECKPOINTS_DIR,
)
from acforensics.data.protocols import load_asvspoof_protocol, load_in_the_wild_protocol
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.data.datasets import ASVspoofDataset, collate_fn
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.normalization import FeatureNormalizer
from acforensics.models.crnn import CRNNForensics
from acforensics.ensemble.predictor import EnsemblePredictor
from acforensics.ensemble.stacking import StackedEnsemble, assert_zero_data_leakage
from acforensics.evaluation.metrics import full_evaluation
from acforensics.evaluation.plots import (
    plot_roc_curve,
    plot_det_curve,
    plot_confusion_matrix,
)
from acforensics.evaluation.ablations import (
    run_feature_ablation,
    run_architecture_ablation,
    run_regularization_ablation,
    run_ensemble_codec_robustness,
    run_ensemble_crossdomain,
    run_per_attack_evaluation,
    run_multi_seed_evaluation,
)
from acforensics.utils.io import save_json
from acforensics.utils.logging import setup_logging, get_logger
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def energy_based_vad_trim(waveform: torch.Tensor, energy_threshold_ratio: float = 0.01) -> torch.Tensor:
    """Trims leading and trailing silence using an energy-based Voice Activity Detection heuristic."""
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


def run_silence_shortcut_check(
    eval_df: pd.DataFrame,
    device: torch.device,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> dict[str, Any]:
    """Evaluates whether the primary CRNN exploits boundary silence artifacts."""
    logger.info("=== Running Silence Shortcut Bias Audit ===")
    ckpt_path = Path("checkpoints/crnn_lfcc_best.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/crnn_best.pt")

    if not ckpt_path.exists():
        logger.warning("No CRNN checkpoint found for silence check.")
        return {}

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = CRNNForensics(in_channels=1, in_freq_dim=LFCC_DEFAULT_DIM)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    dev_thresh = ckpt.get("dev_threshold", 0.0)
    extractor = LFCCExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    norm_file = Path("checkpoints/normalization_stats.json")
    normalizer = FeatureNormalizer().load(norm_file) if norm_file.exists() else FeatureNormalizer()

    untrimmed_scores = []
    trimmed_scores = []
    labels = []

    for _, row in eval_df.iterrows():
        res = load_and_validate_audio(row["audio_path"], target_sr=16000)
        if not res.is_valid or res.waveform is None:
            continue

        labels.append(row["label"])

        # 1. Untrimmed
        wave_untrimmed = fixed_length_crop_or_pad(res.waveform, target_samples=64000, mode="center")
        with torch.no_grad():
            feat1 = extractor(wave_untrimmed)
            if normalizer.is_fitted:
                feat1 = normalizer(feat1)
            score1 = model.predict_score(feat1.to(device)).item()
            untrimmed_scores.append(score1)

        # 2. Trimmed
        trimmed_wave = energy_based_vad_trim(res.waveform)
        wave_trimmed = fixed_length_crop_or_pad(trimmed_wave, target_samples=64000, mode="center")
        with torch.no_grad():
            feat2 = extractor(wave_trimmed)
            if normalizer.is_fitted:
                feat2 = normalizer(feat2)
            score2 = model.predict_score(feat2.to(device)).item()
            trimmed_scores.append(score2)

    labels_arr = np.array(labels, dtype=np.int64)
    s_untrimmed = np.array(untrimmed_scores, dtype=np.float64)
    s_trimmed = np.array(trimmed_scores, dtype=np.float64)

    eval_untrimmed = full_evaluation(labels_arr, s_untrimmed, dev_threshold=dev_thresh, compute_ci=False)
    eval_trimmed = full_evaluation(labels_arr, s_trimmed, dev_threshold=dev_thresh, compute_ci=False)

    silence_delta = eval_trimmed["eer"] - eval_untrimmed["eer"]
    report = {
        "model": "CRNN-LFCC",
        "eval_trials": len(labels_arr),
        "standard_evaluation": eval_untrimmed,
        "silence_trimmed_evaluation": eval_trimmed,
        "silence_shortcut_delta_eer": silence_delta,
        "silence_analysis": (
            f"Standard Eval EER: {eval_untrimmed['eer']:.2f}% vs Silence-Trimmed EER: {eval_trimmed['eer']:.2f}%. "
            f"Delta EER: {silence_delta:+.2f}%. "
            "A small change confirms genuine acoustic feature extraction rather than boundary silence artifact exploitation."
        ),
    }

    metrics_out = Path(results_dir) / "metrics" / "silence_shortcut_analysis.json"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, metrics_out)
    return report


def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Suite & Forensic Ablations CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to default config")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke-test mode with quick verification data")
    parser.add_argument("--all", action="store_true", default=True, help="Run all evaluations and ablations")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('cuda', 'mps', 'cpu', 'auto')")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for output deliverables")
    args = parser.parse_args()

    setup_logging()
    set_seed(42, deterministic=True)

    # 1. Device Resolution
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Master Evaluation running on device: %s", device)

    # 2. Locate Data & Protocols
    is_smoke = bool(args.smoke)
    base_dir = Path("data/smoke_test" if is_smoke else "data/raw/LA")
    proto_dir = base_dir / "ASVspoof2019_LA_cm_protocols"
    if not proto_dir.exists():
        logger.warning("Dataset not found at %s. Falling back to data/smoke_test.", base_dir)
        base_dir = Path("data/smoke_test")
        proto_dir = base_dir / "ASVspoof2019_LA_cm_protocols"
        is_smoke = True

    train_proto = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_proto = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_proto = proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"

    train_df = load_asvspoof_protocol(train_proto, audio_dir=base_dir / "ASVspoof2019_LA_train" / "flac", split_name="train")
    dev_df = load_asvspoof_protocol(dev_proto, audio_dir=base_dir / "ASVspoof2019_LA_dev" / "flac", split_name="dev")
    eval_df = load_asvspoof_protocol(eval_proto, audio_dir=base_dir / "ASVspoof2019_LA_eval" / "flac", split_name="eval")

    # In-The-Wild protocol
    itw_meta = Path("data/smoke_test/in_the_wild/meta.csv" if is_smoke else "data/raw/in_the_wild/meta.csv")
    itw_audio = Path("data/smoke_test/in_the_wild/audio" if is_smoke else "data/raw/in_the_wild/audio")

    # ASV official score file
    asv_score_file = base_dir / "ASVspoof2019_LA_asv_scores" / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    asv_path_str = str(asv_score_file) if asv_score_file.exists() else None

    # Zero-Leakage Assertion
    assert_zero_data_leakage(train_df["file_id"], dev_df["file_id"], eval_df["file_id"])

    # Output paths
    res_dir = Path(args.results_dir)
    tables_dir = res_dir / "tables"
    metrics_dir = res_dir / "metrics"
    figures_dir = res_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PART 1: MASTER MODEL & ENSEMBLE BENCHMARK TABLE
    # =========================================================================
    logger.info("=== Generating Master Model & Ensemble Benchmark Table ===")
    dev_scores_path = Path("results/scores/dev_scores.npy")
    eval_scores_path = Path("results/scores/eval_scores.npy")
    ensemble_ckpt = Path("checkpoints/ensemble/stacked_ensemble.joblib")

    assert dev_scores_path.exists() and eval_scores_path.exists(), (
        "Score matrices not found. Run scripts/build_scores.py first."
    )
    assert ensemble_ckpt.exists(), "Stacked ensemble checkpoint not found. Run scripts/fit_stacker.py first."

    dev_scores = np.load(dev_scores_path)
    dev_labels = np.load("results/scores/dev_labels.npy")
    eval_scores = np.load(eval_scores_path)
    eval_labels = np.load("results/scores/eval_labels.npy")

    stacker = StackedEnsemble.load(ensemble_ckpt)
    model_names = stacker.model_names

    # Dev evaluation (no redundant bootstrap for threshold discovery)
    dev_mean_scores = stacker.predict_baseline_mean(dev_scores)
    dev_mean_rep = full_evaluation(dev_labels, dev_mean_scores, asv_score_file=asv_path_str, compute_ci=False)
    dev_mean_th = dev_mean_rep["eer_threshold"]

    dev_fused_scores = stacker.predict_score(dev_scores)
    dev_stacker_rep = full_evaluation(dev_labels, dev_fused_scores, asv_score_file=asv_path_str, compute_ci=False)
    dev_stacker_th = dev_stacker_rep["eer_threshold"]

    master_rows = []
    roc_dict = {}

    for i, m_name in enumerate(model_names):
        m_dev_s = dev_scores[:, i]
        m_eval_s = eval_scores[:, i]
        m_dev_rep = full_evaluation(dev_labels, m_dev_s, asv_score_file=asv_path_str, compute_ci=False)
        m_dev_th = m_dev_rep["eer_threshold"]
        m_eval_rep = full_evaluation(eval_labels, m_eval_s, dev_threshold=m_dev_th, asv_score_file=asv_path_str, compute_ci=not is_smoke)

        roc_dict[m_name] = (eval_labels, m_eval_s)

        master_rows.append({
            "System": f"Single: {m_name}",
            "Type": "Base Model",
            "Dev EER (%)": f"{m_dev_rep['eer']:.2f}",
            "Eval EER (%)": f"{m_eval_rep['eer']:.2f}",
            "Eval ROC-AUC": f"{m_eval_rep['roc_auc']:.4f}",
            "Eval min-tDCF": f"{m_eval_rep['min_tdcf']:.4f}",
            "Applied Threshold": f"{m_dev_th:.4f} (Dev)",
        })

    # Baseline Mean Fusion
    eval_mean_s = stacker.predict_baseline_mean(eval_scores)
    eval_mean_rep = full_evaluation(eval_labels, eval_mean_s, dev_threshold=dev_mean_th, asv_score_file=asv_path_str, compute_ci=not is_smoke)
    roc_dict["Mean Fusion (Z-Score)"] = (eval_labels, eval_mean_s)
    master_rows.append({
        "System": "Baseline Fusion (Mean Z-Scores)",
        "Type": "Unweighted Average",
        "Dev EER (%)": f"{dev_mean_rep['eer']:.2f}",
        "Eval EER (%)": f"{eval_mean_rep['eer']:.2f}",
        "Eval ROC-AUC": f"{eval_mean_rep['roc_auc']:.4f}",
        "Eval min-tDCF": f"{eval_mean_rep['min_tdcf']:.4f}",
        "Applied Threshold": f"{dev_mean_th:.4f} (Dev)",
    })

    # Stacked Ensemble
    eval_fused_s = stacker.predict_score(eval_scores)
    eval_stacker_rep = full_evaluation(eval_labels, eval_fused_s, dev_threshold=dev_stacker_th, asv_score_file=asv_path_str, compute_ci=not is_smoke)
    roc_dict["Stacked Ensemble (Proposed)"] = (eval_labels, eval_fused_s)
    master_rows.append({
        "System": "Stacked Ensemble (Proposed)",
        "Type": "L2 Logistic Regression",
        "Dev EER (%)": f"{dev_stacker_rep['eer']:.2f}",
        "Eval EER (%)": f"{eval_stacker_rep['eer']:.2f}",
        "Eval ROC-AUC": f"{eval_stacker_rep['roc_auc']:.4f}",
        "Eval min-tDCF": f"{eval_stacker_rep['min_tdcf']:.4f}",
        "Applied Threshold": f"{dev_stacker_th:.4f} (Dev)",
    })

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(tables_dir / "models_and_ensembles_eval.csv", index=False)
    master_df.to_markdown(tables_dir / "models_and_ensembles_eval.md", index=False)
    try:
        master_df.to_latex(tables_dir / "models_and_ensembles_eval.tex", index=False)
    except Exception:
        pass

    # Plot Multi-Model ROC and DET Curves
    plot_roc_curve(roc_dict, figures_dir / "all_models_roc.png", title="Multi-Model ROC Curves (ASVspoof 2019 Eval)")
    plot_det_curve(roc_dict, figures_dir / "all_models_det.png", title="Multi-Model DET Curves (ASVspoof 2019 Eval)")

    # Confusion matrix of Stacked Ensemble at Dev EER threshold
    cm = np.array([
        [eval_stacker_rep["at_applied_threshold"]["tn"], eval_stacker_rep["at_applied_threshold"]["fp"]],
        [eval_stacker_rep["at_applied_threshold"]["fn"], eval_stacker_rep["at_applied_threshold"]["tp"]],
    ])
    plot_confusion_matrix(cm, figures_dir / "ensemble_confusion_matrix.png", title="Stacked Ensemble Confusion Matrix (Eval)")

    # Meta-Learner Weights Table
    weights_df = stacker.get_coefficients_table()
    weights_df.to_csv(tables_dir / "meta_learner_weights.csv", index=False)
    weights_df.to_markdown(tables_dir / "meta_learner_weights.md", index=False)

    # =========================================================================
    # PART 2: PER-ATTACK-TYPE FORENSIC AUDIT (A07 - A19)
    # =========================================================================
    run_per_attack_evaluation(eval_df, eval_fused_s, dev_threshold=dev_stacker_th, results_dir=res_dir)

    # =========================================================================
    # PART 3: FROZEN ENSEMBLE PREDICTOR FOR ROBUSTNESS & CROSS-DOMAIN
    # =========================================================================
    predictor = EnsemblePredictor(device=device)

    # Codec Robustness (MP3, AAC, Opus)
    run_ensemble_codec_robustness(predictor, eval_df, max_files=20 if is_smoke else 100, smoke=is_smoke, results_dir=res_dir)

    # Cross-Domain In-The-Wild
    if itw_meta.exists() and itw_audio.exists():
        run_ensemble_crossdomain(predictor, itw_meta, itw_audio, dev_threshold=dev_stacker_th, max_clips=50 if is_smoke else 200, results_dir=res_dir)

    # =========================================================================
    # PART 4: ABLATION STUDIES (DataLoaders for Feature & Architecture Ablations)
    # =========================================================================
    # LFCC DataLoaders
    lfcc_ext = LFCCExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    norm_file = Path("checkpoints/normalization_stats.json")
    normalizer = FeatureNormalizer().load(norm_file) if norm_file.exists() else FeatureNormalizer()

    def transform_lfcc(wave: torch.Tensor) -> torch.Tensor:
        feat = lfcc_ext(wave)
        return normalizer(feat) if normalizer.is_fitted else feat

    ds_tr_lfcc = ASVspoofDataset(train_df, split="train", feature_transform=transform_lfcc)
    ds_dv_lfcc = ASVspoofDataset(dev_df, split="dev", feature_transform=transform_lfcc)
    ds_ev_lfcc = ASVspoofDataset(eval_df, split="eval", feature_transform=transform_lfcc)

    dl_tr_lfcc = DataLoader(ds_tr_lfcc, batch_size=16, shuffle=True, collate_fn=collate_fn)
    dl_dv_lfcc = DataLoader(ds_dv_lfcc, batch_size=16, shuffle=False, collate_fn=collate_fn)
    dl_ev_lfcc = DataLoader(ds_ev_lfcc, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Log-Mel DataLoaders
    logmel_ext = LogMelExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80)
    norm_logmel_file = Path("checkpoints/normalization_stats_logmel.json")
    normalizer_logmel = FeatureNormalizer().load(norm_logmel_file) if norm_logmel_file.exists() else FeatureNormalizer()

    def transform_logmel(wave: torch.Tensor) -> torch.Tensor:
        feat = logmel_ext(wave)
        return normalizer_logmel(feat) if normalizer_logmel.is_fitted else feat

    ds_tr_logmel = ASVspoofDataset(train_df, split="train", feature_transform=transform_logmel)
    ds_dv_logmel = ASVspoofDataset(dev_df, split="dev", feature_transform=transform_logmel)
    ds_ev_logmel = ASVspoofDataset(eval_df, split="eval", feature_transform=transform_logmel)

    dl_tr_logmel = DataLoader(ds_tr_logmel, batch_size=16, shuffle=True, collate_fn=collate_fn)
    dl_dv_logmel = DataLoader(ds_dv_logmel, batch_size=16, shuffle=False, collate_fn=collate_fn)
    dl_ev_logmel = DataLoader(ds_ev_logmel, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 1. Feature Ablation
    feat_df = run_feature_ablation(
        dl_tr_lfcc, dl_dv_lfcc, dl_ev_lfcc,
        dl_tr_logmel, dl_dv_logmel, dl_ev_logmel,
        device=device,
        asv_score_file=asv_path_str,
        smoke=is_smoke,
        results_dir=res_dir,
    )

    # 2. Architecture Ablation
    arch_df = run_architecture_ablation(
        dl_tr_lfcc, dl_dv_lfcc, dl_ev_lfcc,
        device=device,
        asv_score_file=asv_path_str,
        smoke=is_smoke,
        results_dir=res_dir,
    )

    # 3. Regularization Ablation
    reg_df = run_regularization_ablation(
        dl_tr_lfcc, dl_dv_lfcc, dl_ev_lfcc,
        device=device,
        asv_score_file=asv_path_str,
        smoke=is_smoke,
        results_dir=res_dir,
    )

    # 4. Multi-seed Variance Check
    run_multi_seed_evaluation("lfcc", results_dir=res_dir)
    run_multi_seed_evaluation("logmel", results_dir=res_dir)

    # 5. Silence Shortcut Check
    silence_report = run_silence_shortcut_check(eval_df, device=device, results_dir=res_dir)

    # =========================================================================
    # CONSOLE REPORT PRESENTATION
    # =========================================================================
    print("\n" + "=" * 96)
    print("TABLE 1: MASTER PERFORMANCE BENCHMARK (BASE MODELS VS. FUSIONS)")
    print("=" * 96)
    print(master_df.to_string(index=False))
    print("=" * 96)

    print("\n" + "=" * 96)
    print("TABLE 2: META-LEARNER BASE MODEL CONTRIBUTION WEIGHTS")
    print("=" * 96)
    print(weights_df.to_string(index=False))
    print("=" * 96)

    print("\n" + "=" * 96)
    print("TABLE 3: FEATURE ABLATION (LFCC VS. LOG-MEL VS. BOTH)")
    print("=" * 96)
    print(feat_df.to_string(index=False))
    print("=" * 96)

    print("\n" + "=" * 96)
    print("TABLE 4: ARCHITECTURE ABLATION (CNN VS. CNN+BIGRU VS. CNN+BIGRU+ATTENTION)")
    print("=" * 96)
    print(arch_df.to_string(index=False))
    print("=" * 96)

    print("\n" + "=" * 96)
    print("TABLE 5: REGULARIZATION ABLATION (UNREGULARIZED VS. REGULARIZED)")
    print("=" * 96)
    print(reg_df.to_string(index=False))
    print("=" * 96 + "\n")

    logger.info("All evaluation benchmarks, ablations, tables, and figures successfully written to %s", res_dir)


if __name__ == "__main__":
    main()
