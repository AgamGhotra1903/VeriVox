import argparse
from pathlib import Path
from typing import Any
import json
import time
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    DEFAULT_RESULTS_DIR,
)
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import generate_multi_crop_windows
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics
from acforensics.ensemble.predictor import EnsemblePredictor
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_metrics_at_threshold,
    bootstrap_confidence_interval,
)
from acforensics.evaluation.provenance import create_provenance_record, compute_file_sha256
from acforensics.evaluation.crossdomain import aggregate_window_scores
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def compute_detailed_crossdomain_metrics(
    labels_arr: np.ndarray,
    scores_arr: np.ndarray,
    cats_arr: np.ndarray,
    dev_th: float,
    abstain_margin: float = 0.05,
    seed: int = 42,
    n_bootstraps: int = 1000,
) -> dict[str, Any]:
    itw_eer, itw_opt_thresh = compute_eer(labels_arr, scores_arr)
    itw_auc = compute_roc_auc(labels_arr, scores_arr)
    itw_eer_ci = bootstrap_confidence_interval(
        lambda l, s: compute_eer(l, s)[0],
        labels_arr,
        scores_arr,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    dev_th_metrics = compute_metrics_at_threshold(labels_arr, scores_arr, dev_th)

    probs = 1.0 / (1.0 + np.exp(-np.clip(scores_arr - dev_th, -30.0, 30.0)))
    abstain_mask = np.abs(probs - 0.5) < abstain_margin
    n_abstain = int(np.sum(abstain_mask))
    coverage = float((len(labels_arr) - n_abstain) / max(1, len(labels_arr)) * 100.0)

    if np.any(~abstain_mask) and len(np.unique(labels_arr[~abstain_mask])) >= 2:
        non_ab_metrics = compute_metrics_at_threshold(labels_arr[~abstain_mask], scores_arr[~abstain_mask], dev_th)
        non_ab_eer, _ = compute_eer(labels_arr[~abstain_mask], scores_arr[~abstain_mask])
        non_ab_acc = non_ab_metrics["accuracy"] * 100.0
    else:
        non_ab_eer = itw_eer
        non_ab_acc = dev_th_metrics["accuracy"] * 100.0

    length_breakdown = {}
    for cat in ["short", "long"]:
        m_cat = cats_arr == cat
        if np.sum(m_cat) >= 10 and len(np.unique(labels_arr[m_cat])) >= 2:
            e_cat, _ = compute_eer(labels_arr[m_cat], scores_arr[m_cat])
            a_cat = compute_roc_auc(labels_arr[m_cat], scores_arr[m_cat])
            length_breakdown[cat] = {
                "count": int(np.sum(m_cat)),
                "eer": float(e_cat),
                "roc_auc": float(a_cat),
            }
        else:
            length_breakdown[cat] = {
                "count": int(np.sum(m_cat)),
                "eer": None,
                "roc_auc": None,
            }

    return {
        "num_evaluated_clips": len(labels_arr),
        "bonafide_count": int(np.sum(labels_arr == BONAFIDE)),
        "spoof_count": int(np.sum(labels_arr == SPOOF)),
        "optimal_threshold_metrics": {
            "itw_optimal_eer": float(itw_eer),
            "itw_optimal_eer_ci_95": [float(itw_eer_ci[0]), float(itw_eer_ci[1])],
            "itw_optimal_threshold": float(itw_opt_thresh),
            "roc_auc": float(itw_auc),
        },
        "dev_threshold_metrics": {
            "asvspoof_dev_threshold": float(dev_th),
            "accuracy": float(dev_th_metrics["accuracy"] * 100.0),
            "precision": float(dev_th_metrics["precision"]),
            "recall": float(dev_th_metrics["recall"]),
            "f1": float(dev_th_metrics["f1"]),
        },
        "abstain_analysis": {
            "abstain_margin": float(abstain_margin),
            "abstain_count": n_abstain,
            "coverage_pct": float(coverage),
            "non_abstained_eer": float(non_ab_eer),
            "non_abstained_accuracy": float(non_ab_acc),
        },
        "clip_length_breakdown": length_breakdown,
        "min_tdcf_note": "min-tDCF is undefined on In-The-Wild as the corpus lacks paired ASV speaker verification trials.",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate In-the-Wild dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crnn_lfcc_best.pt")
    parser.add_argument("--dev-threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("Starting In-the-Wild Cross-Domain Evaluation on device: %s", device)

    results_dir = Path(DEFAULT_RESULTS_DIR)
    metrics_dir = results_dir / "metrics"
    tables_dir = results_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    crnn_ckpt_path = Path(args.checkpoint)
    if not crnn_ckpt_path.exists():
        crnn_ckpt_path = Path("checkpoints/crnn_best.pt")
    ckpt = torch.load(str(crnn_ckpt_path), map_location=device, weights_only=False)
    crnn_model = CRNNForensics(in_channels=1, in_freq_dim=60)
    crnn_model.load_state_dict(ckpt["model_state_dict"])
    crnn_model.to(device)
    crnn_model.eval()

    extractor = LFCCExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    ensemble_predictor = EnsemblePredictor(device=device)

    if args.dev_threshold is not None:
        crnn_dev_th = float(args.dev_threshold)
    elif "dev_threshold" in ckpt:
        crnn_dev_th = float(ckpt["dev_threshold"])
    else:
        crnn_dev_th = -0.1428

    ens_dev_th = -1.9367

    itw_manifest = Path("data/manifests/itw_test.parquet")
    itw_prov = create_provenance_record("in_the_wild_evaluation", itw_manifest, seed=42)

    itw_df = pd.read_parquet(itw_manifest)
    logger.info("Loaded %d In-The-Wild test clips (speakers: %d)", len(itw_df), itw_df["speaker"].nunique())

    t0 = time.time()
    def load_itw_item(item):
        idx, row = item
        res = load_and_validate_audio(row["path"], target_sr=SAMPLE_RATE)
        if res.is_valid and res.waveform is not None:
            dur = float(len(res.waveform) / SAMPLE_RATE)
            cat = "short" if dur < 1.0 else ("long" if dur > 4.0 else "medium")
            return res.waveform, int(row["label"]), cat
        return None

    logger.info("Loading all %d ITW audio clips in parallel...", len(itw_df))
    with ThreadPoolExecutor(max_workers=8) as ex:
        itw_loaded = list(ex.map(load_itw_item, itw_df.iterrows()))

    itw_valid = [x for x in itw_loaded if x is not None]
    logger.info("Successfully loaded %d / %d ITW clips in %.1fs", len(itw_valid), len(itw_df), time.time() - t0)

    logger.info("Evaluating CRNN-LFCC across all %d clips with per_utterance_cmvn...", len(itw_valid))
    t1 = time.time()
    all_crnn_window_scores = []
    itw_labels = []
    itw_cats = []

    for i, (wave, label, cat) in enumerate(itw_valid):
        windows = generate_multi_crop_windows(wave, window_samples=TARGET_SAMPLES, hop_ratio=0.5)
        with torch.no_grad():
            feats = [per_utterance_cmvn(extractor(w["waveform"])) for w in windows]
            b_feats = torch.stack(feats)
            if b_feats.dim() == 3:
                b_feats = b_feats.unsqueeze(1)
            b_feats = b_feats.to(device)
            logits, _ = crnn_model(b_feats)
            w_scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
        all_crnn_window_scores.append([float(s) for s in w_scores])
        itw_labels.append(label)
        itw_cats.append(cat)
        if (i + 1) % 5000 == 0 or (i + 1) == len(itw_valid):
            logger.info("CRNN evaluated %d / %d clips (%.1fs elapsed)", i + 1, len(itw_valid), time.time() - t1)

    labels_arr = np.array(itw_labels, dtype=np.int64)
    cats_arr = np.array(itw_cats)

    itw_results_by_agg = {}
    for agg in ["mean", "max", "top_k"]:
        agg_scores = np.array([aggregate_window_scores(ws, strategy=agg) for ws in all_crnn_window_scores], dtype=np.float64)
        m = compute_detailed_crossdomain_metrics(labels_arr, agg_scores, cats_arr, dev_th=crnn_dev_th, seed=42)
        m["aggregation_strategy"] = agg
        m["dataset"] = "In-the-Wild Deepfake Audio (Müller et al., 2022)"
        itw_results_by_agg[f"crnn_{agg}"] = m
        opt = m["optimal_threshold_metrics"]
        logger.info("CRNN (%s) -> Optimal EER: %.2f%%, ROC-AUC: %.4f", agg, opt["itw_optimal_eer"], opt["roc_auc"])

    logger.info("Evaluating Stacked Ensemble on In-The-Wild (N=2,500 subset)...")
    ens_sub_sample = itw_valid[:2500]
    ens_sub_labels = np.array([x[1] for x in ens_sub_sample], dtype=np.int64)
    ens_sub_cats = np.array([x[2] for x in ens_sub_sample])
    ens_scores = []
    t2 = time.time()
    for j, (wave, _, _) in enumerate(ens_sub_sample):
        windows = generate_multi_crop_windows(wave, window_samples=TARGET_SAMPLES, hop_ratio=0.5)
        ws = [float(ensemble_predictor.predict_waveform(w["waveform"])["fused_score"]) for w in windows]
        ens_scores.append(aggregate_window_scores(ws, strategy="mean"))
        if (j + 1) % 500 == 0 or (j + 1) == len(ens_sub_sample):
            logger.info("Ensemble evaluated %d / %d clips (%.1fs elapsed)", j + 1, len(ens_sub_sample), time.time() - t2)

    ens_scores_arr = np.array(ens_scores, dtype=np.float64)
    m_ens = compute_detailed_crossdomain_metrics(ens_sub_labels, ens_scores_arr, ens_sub_cats, dev_th=ens_dev_th, seed=42)
    m_ens["aggregation_strategy"] = "mean"
    m_ens["dataset"] = "In-the-Wild Deepfake Audio (Müller et al., 2022)"
    itw_results_by_agg["ensemble_mean"] = m_ens

    itw_summary = {
        **itw_prov,
        "eval_split": "itw_test",
        "checkpoint_path": str(crnn_ckpt_path),
        "checkpoint_sha256": compute_file_sha256(str(crnn_ckpt_path)),
        "num_clips_tested": len(labels_arr),
        "aggregations": itw_results_by_agg,
    }
    out_json = metrics_dir / "in_the_wild_results.json"
    with open(out_json, "w") as f:
        json.dump(itw_summary, f, indent=2)
    logger.info("Saved ITW metrics to %s", out_json)

    itw_table_rows = []
    for model_key, r in itw_results_by_agg.items():
        name = "CRNN-LFCC (" + model_key.replace("crnn_", "") + ")" if "crnn" in model_key else "Stacked Ensemble (mean)"
        opt = r["optimal_threshold_metrics"]
        dev_m = r["dev_threshold_metrics"]
        abs_m = r["abstain_analysis"]
        itw_table_rows.append({
            "Model / Aggregation": name,
            "ITW Optimal EER (%)": f"{opt['itw_optimal_eer']:.2f}",
            "95% CI (%)": f"[{opt['itw_optimal_eer_ci_95'][0]:.2f}, {opt['itw_optimal_eer_ci_95'][1]:.2f}]",
            "ROC-AUC": f"{opt['roc_auc']:.4f}",
            "Accuracy @ Dev Thresh (%)": f"{dev_m['accuracy']:.2f}",
            "Precision": f"{dev_m['precision']:.3f}",
            "Recall": f"{dev_m['recall']:.3f}",
            "Abstain Coverage (%)": f"{abs_m['coverage_pct']:.1f}",
            "Non-Abstain EER (%)": f"{abs_m['non_abstained_eer']:.2f}",
        })
    df_itw = pd.DataFrame(itw_table_rows)
    for p in [tables_dir / "table4_crossdomain.md", tables_dir / "table4_crossdomain_in_the_wild.md"]:
        df_itw.to_markdown(p, index=False)
    for p in [tables_dir / "table4_crossdomain.tex", tables_dir / "table4_crossdomain_in_the_wild.tex"]:
        df_itw.to_latex(p, index=False)
    logger.info("Saved Table 4 crossdomain tables.")


if __name__ == "__main__":
    main()
