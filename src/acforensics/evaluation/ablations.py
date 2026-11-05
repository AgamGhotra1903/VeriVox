"""Ablation Studies and Robustness Evaluation Suite.

Implements all required experimental ablations:
1. Feature Ablation: LFCC vs. Log-Mel vs. Both (Score/Combined Fusion).
2. Architecture Ablation: CNN vs. CNN+BiGRU vs. CNN+BiGRU+Attention.
3. Regularization Ablation: Regularized vs. Unregularized (Dropout & Weight Decay).
4. Codec Robustness Stress-Testing on the Frozen Ensemble (MP3, AAC, Opus @ 16-64 kbps).
5. Cross-Domain In-The-Wild Evaluation on the Frozen Ensemble.
6. Per-Attack-Type Forensic Breakdown on Eval (A07 - A19).

All tables are saved to results/tables/ in CSV and Markdown formats.
All figures are saved to results/figures/ in PNG and SVG formats.

Author: Agam Ghotra (IIT2024226)
"""

from pathlib import Path
from typing import Any, Optional
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    LFCC_DEFAULT_DIM,
    LOGMEL_DEFAULT_DIM,
    SAMPLE_RATE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_CHECKPOINTS_DIR,
)
from acforensics.data.protocols import load_asvspoof_protocol, load_in_the_wild_protocol
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.data.datasets import ASVspoofDataset, collate_fn
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.normalization import FeatureNormalizer, per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics
from acforensics.training.train_crnn import train_crnn
from acforensics.ensemble.predictor import EnsemblePredictor
from acforensics.ensemble.stacking import StackedEnsemble
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_min_tdcf,
    full_evaluation,
)
from acforensics.evaluation.plots import (
    plot_roc_curve,
    plot_det_curve,
    plot_confusion_matrix,
)
from acforensics.evaluation.robustness import (
    check_ffmpeg_available,
    compress_and_decompress_audio,
    plot_robustness_curves,
)
from acforensics.evaluation.error_analysis import (
    run_error_audit,
    plot_per_attack_eer,
)
from acforensics.utils.io import save_json
from acforensics.utils.logging import get_logger
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def evaluate_model_on_split(
    model: CRNNForensics,
    data_loader: DataLoader,
    device: torch.device,
    dev_threshold: Optional[float] = None,
    asv_score_file: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluates a PyTorch CRNN model on a DataLoader split."""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch["feature"].to(device)
            labels = batch["label"].numpy()
            scores = model.predict_score(x).cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels)

    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int64)

    metrics = full_evaluation(
        labels_arr,
        scores_arr,
        dev_threshold=dev_threshold,
        asv_score_file=asv_score_file,
        compute_ci=False,
    )
    return scores_arr, labels_arr, metrics


def run_feature_ablation(
    train_loader_lfcc: DataLoader,
    dev_loader_lfcc: DataLoader,
    eval_loader_lfcc: DataLoader,
    train_loader_logmel: DataLoader,
    dev_loader_logmel: DataLoader,
    eval_loader_logmel: DataLoader,
    device: torch.device,
    asv_score_file: Optional[str] = None,
    smoke: bool = False,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """Ablation 1: Compares representations across LFCC (60-dim), Log-Mel (80-dim), and Combined."""
    logger.info("=== Running Feature Ablation Study (LFCC vs Log-Mel vs Both) ===")
    out_dir = Path(results_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # 1. LFCC Model
    lfcc_ckpt = Path("checkpoints/crnn_lfcc_best.pt")
    if not lfcc_ckpt.exists():
        lfcc_ckpt = Path("checkpoints/crnn_best.pt")
    m_lfcc = CRNNForensics(in_channels=1, in_freq_dim=LFCC_DEFAULT_DIM, features="lfcc")
    if lfcc_ckpt.exists():
        ckpt = torch.load(str(lfcc_ckpt), map_location=device, weights_only=False)
        m_lfcc.load_state_dict(ckpt["model_state_dict"])
    else:
        m_lfcc, _, _ = train_crnn(m_lfcc, train_loader_lfcc, dev_loader_lfcc, epochs=2 if smoke else 10, device=device)
    m_lfcc.to(device)

    dev_s, dev_l, dev_rep = evaluate_model_on_split(m_lfcc, dev_loader_lfcc, device, asv_score_file=asv_score_file)
    dev_th = dev_rep["eer_threshold"]
    eval_s, eval_l, eval_rep = evaluate_model_on_split(m_lfcc, eval_loader_lfcc, device, dev_threshold=dev_th, asv_score_file=asv_score_file)

    rows.append({
        "Feature Representation": "LFCC (60-dim)",
        "Input Dimension": "60 (20 stat + 20 delta + 20 delta-delta)",
        "Dev EER (%)": f"{dev_rep['eer']:.2f}",
        "Eval EER (%)": f"{eval_rep['eer']:.2f}",
        "Eval ROC-AUC": f"{eval_rep['roc_auc']:.4f}",
        "Eval min-tDCF": f"{eval_rep['min_tdcf']:.4f}",
    })

    # 2. Log-Mel Model
    logmel_ckpt = Path("checkpoints/crnn_logmel_best.pt")
    m_logmel = CRNNForensics(in_channels=1, in_freq_dim=LOGMEL_DEFAULT_DIM, features="logmel")
    if logmel_ckpt.exists():
        ckpt = torch.load(str(logmel_ckpt), map_location=device, weights_only=False)
        m_logmel.load_state_dict(ckpt["model_state_dict"])
    else:
        m_logmel, _, _ = train_crnn(m_logmel, train_loader_logmel, dev_loader_logmel, epochs=2 if smoke else 10, device=device)
    m_logmel.to(device)

    dev_s2, dev_l2, dev_rep2 = evaluate_model_on_split(m_logmel, dev_loader_logmel, device, asv_score_file=asv_score_file)
    dev_th2 = dev_rep2["eer_threshold"]
    eval_s2, eval_l2, eval_rep2 = evaluate_model_on_split(m_logmel, eval_loader_logmel, device, dev_threshold=dev_th2, asv_score_file=asv_score_file)

    rows.append({
        "Feature Representation": "Log-Mel (80-dim)",
        "Input Dimension": "80 Mel filterbank energy bins",
        "Dev EER (%)": f"{dev_rep2['eer']:.2f}",
        "Eval EER (%)": f"{eval_rep2['eer']:.2f}",
        "Eval ROC-AUC": f"{eval_rep2['roc_auc']:.4f}",
        "Eval min-tDCF": f"{eval_rep2['min_tdcf']:.4f}",
    })

    # 3. Both (Fused LFCC + LogMel Score Fusion)
    # Mean of z-normalized scores of LFCC and Log-Mel
    z_dev_comb = (
        (dev_s - np.mean(dev_s)) / (np.std(dev_s) + 1e-8)
        + (dev_s2 - np.mean(dev_s2)) / (np.std(dev_s2) + 1e-8)
    ) / 2.0
    comb_dev_rep = full_evaluation(dev_l, z_dev_comb, asv_score_file=asv_score_file)
    comb_dev_th = comb_dev_rep["eer_threshold"]

    z_eval_comb = (
        (eval_s - np.mean(dev_s)) / (np.std(dev_s) + 1e-8)
        + (eval_s2 - np.mean(dev_s2)) / (np.std(dev_s2) + 1e-8)
    ) / 2.0
    comb_eval_rep = full_evaluation(eval_l, z_eval_comb, dev_threshold=comb_dev_th, asv_score_file=asv_score_file)

    rows.append({
        "Feature Representation": "Both (LFCC + Log-Mel Fusion)",
        "Input Dimension": "140 combined spectral channels",
        "Dev EER (%)": f"{comb_dev_rep['eer']:.2f}",
        "Eval EER (%)": f"{comb_eval_rep['eer']:.2f}",
        "Eval ROC-AUC": f"{comb_eval_rep['roc_auc']:.4f}",
        "Eval min-tDCF": f"{comb_eval_rep['min_tdcf']:.4f}",
    })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "feature_ablation.csv", index=False)
    df.to_markdown(out_dir / "feature_ablation.md", index=False)
    logger.info("Saved Feature Ablation results to %s", out_dir / "feature_ablation.csv")
    return df


def run_architecture_ablation(
    train_loader: DataLoader,
    dev_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    asv_score_file: Optional[str] = None,
    smoke: bool = False,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """Ablation 2: Compares CNN vs CNN+BiGRU vs CNN+BiGRU+Attention."""
    logger.info("=== Running Architecture Ablation Study ===")
    out_dir = Path(results_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "CNN Only",
            "desc": "4 ConvBlocks -> Linear Head",
            "kwargs": {"use_gru": False, "use_attention": False, "pooling": "mean"},
            "ckpt": "checkpoints/crnn_ablation_cnn_only.pt",
        },
        {
            "name": "CNN + BiGRU",
            "desc": "4 ConvBlocks -> 2-layer BiGRU -> Mean Pooling",
            "kwargs": {"use_gru": True, "use_attention": False, "pooling": "mean"},
            "ckpt": "checkpoints/crnn_ablation_cnn_gru.pt",
        },
        {
            "name": "CNN + BiGRU + Attention (Full)",
            "desc": "4 ConvBlocks -> 2-layer BiGRU -> Multi-Head Self-Attention",
            "kwargs": {"use_gru": True, "use_attention": True, "pooling": "attention"},
            "ckpt": "checkpoints/crnn_lfcc_best.pt",
        },
    ]

    rows = []

    for cfg in configs:
        model = CRNNForensics(
            in_channels=1,
            in_freq_dim=LFCC_DEFAULT_DIM,
            features="lfcc",
            **cfg["kwargs"],
        )
        ckpt_path = Path(cfg["ckpt"])
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model, _, _ = train_crnn(
                model,
                train_loader,
                dev_loader,
                epochs=2 if smoke else 10,
                checkpoint_name=ckpt_path.name,
                device=device,
            )
        model.to(device)

        dev_s, dev_l, dev_rep = evaluate_model_on_split(model, dev_loader, device, asv_score_file=asv_score_file)
        dev_th = dev_rep["eer_threshold"]
        eval_s, eval_l, eval_rep = evaluate_model_on_split(model, eval_loader, device, dev_threshold=dev_th, asv_score_file=asv_score_file)

        rows.append({
            "Architecture Variant": cfg["name"],
            "Description": cfg["desc"],
            "Dev EER (%)": f"{dev_rep['eer']:.2f}",
            "Eval EER (%)": f"{eval_rep['eer']:.2f}",
            "Eval ROC-AUC": f"{eval_rep['roc_auc']:.4f}",
            "Eval min-tDCF": f"{eval_rep['min_tdcf']:.4f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "architecture_ablation.csv", index=False)
    df.to_markdown(out_dir / "architecture_ablation.md", index=False)
    logger.info("Saved Architecture Ablation results to %s", out_dir / "architecture_ablation.csv")
    return df


def run_regularization_ablation(
    train_loader: DataLoader,
    dev_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    asv_score_file: Optional[str] = None,
    smoke: bool = False,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """Ablation 3: Evaluates effect of Dropout and Weight Decay regularization."""
    logger.info("=== Running Regularization Ablation Study ===")
    out_dir = Path(results_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    reg_variants = [
        {
            "name": "Unregularized CRNN",
            "desc": "No Dropout (0.0), No Weight Decay (0.0)",
            "dropout": 0.0,
            "weight_decay": 0.0,
            "ckpt": "checkpoints/crnn_ablation_unregularized.pt",
        },
        {
            "name": "Regularized CRNN (Proposed)",
            "desc": "Conv/GRU Dropout (0.3), Head Dropout (0.4), AdamW Weight Decay (1e-4)",
            "dropout": 0.3,
            "weight_decay": 1e-4,
            "ckpt": "checkpoints/crnn_lfcc_best.pt",
        },
    ]

    rows = []

    for v in reg_variants:
        model = CRNNForensics(
            in_channels=1,
            in_freq_dim=LFCC_DEFAULT_DIM,
            features="lfcc",
            conv_dropout=v["dropout"],
            gru_dropout=v["dropout"],
            classifier_dropout=0.4 if v["dropout"] > 0 else 0.0,
        )
        ckpt_path = Path(v["ckpt"])
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model, _, _ = train_crnn(
                model,
                train_loader,
                dev_loader,
                epochs=2 if smoke else 10,
                weight_decay=v["weight_decay"],
                checkpoint_name=ckpt_path.name,
                device=device,
            )
        model.to(device)

        dev_s, dev_l, dev_rep = evaluate_model_on_split(model, dev_loader, device, asv_score_file=asv_score_file)
        dev_th = dev_rep["eer_threshold"]
        eval_s, eval_l, eval_rep = evaluate_model_on_split(model, eval_loader, device, dev_threshold=dev_th, asv_score_file=asv_score_file)

        rows.append({
            "Configuration": v["name"],
            "Regularization Details": v["desc"],
            "Dev EER (%)": f"{dev_rep['eer']:.2f}",
            "Eval EER (%)": f"{eval_rep['eer']:.2f}",
            "Eval ROC-AUC": f"{eval_rep['roc_auc']:.4f}",
            "Eval min-tDCF": f"{eval_rep['min_tdcf']:.4f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "regularization_ablation.csv", index=False)
    df.to_markdown(out_dir / "regularization_ablation.md", index=False)
    logger.info("Saved Regularization Ablation results to %s", out_dir / "regularization_ablation.csv")
    return df


def run_ensemble_codec_robustness(
    predictor: EnsemblePredictor,
    eval_df: pd.DataFrame,
    max_files: int = 50,
    smoke: bool = False,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> dict[str, Any]:
    """Ablation 4: Evaluates the frozen stacked ensemble on MP3, AAC, and Opus lossy audio."""
    logger.info("=== Running Codec Compression Stress-Testing on Frozen Ensemble ===")
    out_dir = Path(results_dir)
    tables_dir = out_dir / "tables"
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not check_ffmpeg_available():
        logger.warning("ffmpeg is not installed or not in PATH. Skipping codec robustness evaluation.")
        return {}

    test_subset = eval_df.head(max_files)
    clean_scores = []
    labels = []

    for _, row in test_subset.iterrows():
        try:
            pred = predictor.predict_file(row["audio_path"])
            clean_scores.append(pred["fused_score"])
            labels.append(row["label"])
        except Exception as e:
            logger.warning("Skipping file %s: %s", row["audio_path"], e)

    clean_scores_arr = np.array(clean_scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int64)

    clean_eer, _ = compute_eer(labels_arr, clean_scores_arr)

    codecs = ["mp3", "aac", "opus"]
    bitrates = [16, 32, 64] if smoke else [16, 24, 32, 48, 64]

    cache_root = Path("data/compressed")
    results_by_codec = {c: {} for c in codecs}
    table_rows = []

    for codec in codecs:
        for kbps in bitrates:
            comp_dir = cache_root / f"{codec}_{kbps}k"
            comp_scores = []
            comp_labels = []

            for _, row in test_subset.iterrows():
                in_path = Path(row["audio_path"])
                out_path = comp_dir / f"{in_path.stem}.wav"

                if not out_path.exists():
                    compress_and_decompress_audio(in_path, out_path, codec=codec, bitrate_kbps=kbps)

                if out_path.exists():
                    try:
                        pred = predictor.predict_file(out_path)
                        comp_scores.append(pred["fused_score"])
                        comp_labels.append(row["label"])
                    except Exception as e:
                        logger.warning("Failed predicting compressed file %s: %s", out_path, e)

            if len(comp_scores) > 0:
                eer_comp, _ = compute_eer(np.array(comp_labels), np.array(comp_scores))
                results_by_codec[codec][kbps] = float(eer_comp)
                table_rows.append({
                    "Codec": codec.upper(),
                    "Bitrate (kbps)": kbps,
                    "Clean Ensemble EER (%)": f"{clean_eer:.2f}",
                    "Compressed Ensemble EER (%)": f"{eer_comp:.2f}",
                    "Degradation (pts)": f"{eer_comp - clean_eer:+.2f}",
                })

    plot_robustness_curves(results_by_codec, clean_eer, figures_dir / "robustness_curves.png")

    df_rob = pd.DataFrame(table_rows)
    df_rob.to_csv(tables_dir / "codec_robustness_ensemble.csv", index=False)
    df_rob.to_markdown(tables_dir / "codec_robustness_ensemble.md", index=False)

    summary = {
        "model": "Stacked Forensic Ensemble (Frozen)",
        "clean_baseline_eer": float(clean_eer),
        "results_by_codec": results_by_codec,
        "table": table_rows,
    }
    save_json(summary, metrics_dir / "codec_robustness_ensemble.json")
    logger.info("Saved Codec Robustness results to %s", tables_dir / "codec_robustness_ensemble.csv")
    return summary


def run_ensemble_crossdomain(
    predictor: EnsemblePredictor,
    meta_csv: Path | str,
    audio_dir: Path | str,
    dev_threshold: float,
    max_clips: int = 50,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> dict[str, Any]:
    """Ablation 5: Zero-shot cross-domain evaluation on In-The-Wild deepfake speech."""
    logger.info("=== Running In-The-Wild Cross-Domain Evaluation on Frozen Ensemble ===")
    out_dir = Path(results_dir)
    tables_dir = out_dir / "tables"
    metrics_dir = out_dir / "metrics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(meta_csv)
    if not meta_path.exists():
        logger.warning("In-the-Wild metadata not found at %s. Skipping ITW evaluation.", meta_path)
        return {}

    df = load_in_the_wild_protocol(meta_path, audio_dir=audio_dir)
    if max_clips is not None:
        df = df.head(max_clips)

    scores = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring ITW Ensemble"):
        try:
            pred = predictor.predict_file(row["audio_path"])
            scores.append(pred["fused_score"])
            labels.append(row["label"])
        except Exception as e:
            logger.warning("Skipping ITW clip %s: %s", row["audio_path"], e)

    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int64)

    itw_eer, itw_opt_thresh = compute_eer(labels_arr, scores_arr)
    itw_auc = compute_roc_auc(labels_arr, scores_arr)

    # Accuracy @ ASVspoof dev threshold
    preds_dev_th = (scores_arr >= dev_threshold).astype(int)
    acc_dev_th = float(np.mean(preds_dev_th == labels_arr))

    summary = {
        "model": "Stacked Forensic Ensemble (Frozen)",
        "dataset": "In-the-Wild Deepfake Audio (Müller et al., 2022)",
        "num_evaluated_clips": len(labels_arr),
        "bonafide_count": int(np.sum(labels_arr == BONAFIDE)),
        "spoof_count": int(np.sum(labels_arr == SPOOF)),
        "itw_optimal_eer": float(itw_eer),
        "itw_optimal_threshold": float(itw_opt_thresh),
        "itw_roc_auc": float(itw_auc),
        "asvspoof_dev_threshold": float(dev_threshold),
        "accuracy_at_dev_threshold": float(acc_dev_th),
    }

    rows = [{
        "Model System": "Stacked Ensemble (Proposed)",
        "Domain": "In-the-Wild (Unseen Zero-Shot)",
        "Evaluated Clips": len(labels_arr),
        "Optimal In-Domain EER (%)": f"{itw_eer:.2f}",
        "In-Domain ROC-AUC": f"{itw_auc:.4f}",
        "Operating Threshold": f"{dev_threshold:.4f} (from Dev)",
        "Accuracy @ Dev Threshold (%)": f"{acc_dev_th * 100:.2f}",
    }]
    df_itw = pd.DataFrame(rows)
    df_itw.to_csv(tables_dir / "crossdomain_in_the_wild.csv", index=False)
    df_itw.to_markdown(tables_dir / "crossdomain_in_the_wild.md", index=False)
    save_json(summary, metrics_dir / "crossdomain_ensemble.json")
    logger.info("Saved Cross-Domain results to %s", tables_dir / "crossdomain_in_the_wild.csv")
    return summary


def run_per_attack_evaluation(
    eval_df: pd.DataFrame,
    eval_scores: np.ndarray,
    dev_threshold: float,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """Ablation 6: Computes and plots per-attack-type EER across unseen eval attacks (A07-A19)."""
    logger.info("=== Running Per-Attack-Type EER Analysis (A07 - A19) ===")
    out_dir = Path(results_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = eval_df.copy().reset_index(drop=True)
    assert len(df) == len(eval_scores), "Mismatch between eval protocol rows and scores"
    df["score"] = eval_scores

    bf_scores = df[df["label"] == BONAFIDE]["score"].values
    attack_data = []
    attack_eers = {}

    for attack_id in sorted(df[df["label"] == SPOOF]["attack_id"].unique()):
        sp_scores = df[df["attack_id"] == attack_id]["score"].values
        if len(sp_scores) > 0 and len(bf_scores) > 0:
            labels_comb = np.concatenate([np.ones_like(bf_scores), np.zeros_like(sp_scores)])
            scores_comb = np.concatenate([bf_scores, sp_scores])
            eer_val, opt_th = compute_eer(labels_comb, scores_comb)
            auc_val = compute_roc_auc(labels_comb, scores_comb)
            attack_eers[attack_id] = float(eer_val)

            nature = "Known Variant" if attack_id in ("A16", "A19") else "Unseen Technology"
            attack_data.append({
                "Attack ID": attack_id,
                "Taxonomy": nature,
                "Spoof Trials": len(sp_scores),
                "Attack EER (%)": f"{eer_val:.2f}",
                "ROC-AUC": f"{auc_val:.4f}",
            })

    plot_per_attack_eer(attack_eers, figures_dir / "per_attack_eer.png")

    df_attack = pd.DataFrame(attack_data)
    df_attack.to_csv(tables_dir / "per_attack_eer.csv", index=False)
    df_attack.to_markdown(tables_dir / "per_attack_eer.md", index=False)
    logger.info("Saved Per-Attack EER results to %s", tables_dir / "per_attack_eer.csv")
    return df_attack


def run_multi_seed_evaluation(
    feature_type: str = "lfcc",
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> Optional[pd.DataFrame]:
    """Ablation 7: Gathers multi-seed training runs and computes mean +/- std."""
    metrics_dir = Path(results_dir) / "metrics"
    tables_dir = Path(results_dir) / "tables"
    multi_file = metrics_dir / f"crnn_{feature_type}_multiseed_summary.json"

    if not multi_file.exists():
        logger.info("No multi-seed summary found at %s. Single seed reporting active.", multi_file)
        return None

    data = json.loads(multi_file.read_text())
    m = data["metrics"]
    rows = [{
        "Model": f"CRNN-{feature_type.upper()}",
        "Evaluated Seeds": ", ".join(map(str, data.get("seeds", []))),
        "Dev EER (%)": f"{m['dev_eer_mean']:.2f} +/- {m['dev_eer_std']:.2f}",
        "Eval EER (%)": f"{m['eval_eer_mean']:.2f} +/- {m['eval_eer_std']:.2f}",
        "Eval ROC-AUC": f"{m['eval_auc_mean']:.4f} +/- {m['eval_auc_std']:.4f}",
        "Eval min-tDCF": f"{m['eval_tdcf_mean']:.4f} +/- {m['eval_tdcf_std']:.4f}",
    }]
    df_ms = pd.DataFrame(rows)
    df_ms.to_csv(tables_dir / f"crnn_{feature_type}_multi_seed_metrics.csv", index=False)
    df_ms.to_markdown(tables_dir / f"crnn_{feature_type}_multi_seed_metrics.md", index=False)
    return df_ms

