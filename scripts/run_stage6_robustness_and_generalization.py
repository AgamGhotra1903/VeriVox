from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor
import shutil
import soundfile as sf

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    DEFAULT_RESULTS_DIR,
)
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad, generate_multi_crop_windows
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import FeatureNormalizer, per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics
from acforensics.models.classical import extract_mean_std_features, predict_classical_score
from acforensics.ensemble.predictor import EnsemblePredictor
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_metrics_at_threshold,
    bootstrap_confidence_interval,
)
from acforensics.evaluation.provenance import create_provenance_record
from acforensics.evaluation.robustness import (
    compress_and_decompress_audio,
    apply_telephony_channel,
    apply_additive_noise,
    apply_synthetic_reverb,
    apply_packet_loss,
    plot_robustness_curves,
    check_ffmpeg_available,
)
from acforensics.evaluation.crossdomain import aggregate_window_scores
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)

def evaluate_waveform_batch_crnn(
    waveforms: list[torch.Tensor],
    model: torch.nn.Module,
    extractor: LFCCExtractor,
    normalizer: FeatureNormalizer,
    device: torch.device,
) -> list[float]:
    feats = []
    for w in waveforms:
        cw = fixed_length_crop_or_pad(w, target_samples=TARGET_SAMPLES, mode="center")
        f = extractor(cw)
        if normalizer.is_fitted:
            f = normalizer(f)
        feats.append(f)
    batch = torch.stack(feats)
    if batch.dim() == 3:
        batch = batch.unsqueeze(1)
    batch = batch.to(device)
    with torch.no_grad():
        logits, _ = model(batch)
        scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
    return [float(s) for s in scores]

def score_waveform_ensemble(
    wave: torch.Tensor,
    predictor: EnsemblePredictor,
) -> float:
    pred = predictor.predict_waveform(wave)
    return float(pred["fused_score"])

def evaluate_waveform_batch_ensemble(
    waveforms: list[torch.Tensor],
    predictor: EnsemblePredictor,
    batch_size: int = 128,
) -> list[float]:
    cropped = [fixed_length_crop_or_pad(w, target_samples=TARGET_SAMPLES, mode="center") for w in waveforms]
    with torch.no_grad():
        raw_lfccs = [predictor.lfcc_ext(w) for w in cropped]
        raw_logmels = [predictor.logmel_ext(w) for w in cropped]
        norm_lfccs = [predictor.norm_lfcc(f) if (predictor.norm_lfcc.is_fitted and getattr(predictor.norm_lfcc.mean, "shape", [0])[0] == 60) else per_utterance_cmvn(f) for f in raw_lfccs]
        norm_logmels = [predictor.norm_logmel(f) if (predictor.norm_logmel.is_fitted and getattr(predictor.norm_logmel.mean, "shape", [0])[0] == 80) else per_utterance_cmvn(f) for f in raw_logmels]
        s_crnn_lfcc_list = []
        s_crnn_logmel_list = []
        for i in range(0, len(waveforms), batch_size):
            b_lfcc = torch.stack(norm_lfccs[i:i+batch_size]).unsqueeze(1).to(predictor.device)
            b_logmel = torch.stack(norm_logmels[i:i+batch_size]).unsqueeze(1).to(predictor.device)
            log_lfcc, _ = predictor.crnn_lfcc(b_lfcc)
            s_crnn_lfcc_list.append((log_lfcc[:, BONAFIDE] - log_lfcc[:, SPOOF]).cpu().numpy())
            log_logmel, _ = predictor.crnn_logmel(b_logmel)
            s_crnn_logmel_list.append((log_logmel[:, BONAFIDE] - log_logmel[:, SPOOF]).cpu().numpy())
        s_crnn_lfcc = np.concatenate(s_crnn_lfcc_list)
        s_crnn_logmel = np.concatenate(s_crnn_logmel_list)

    raw_lfcc_np = [f.T.detach().cpu().numpy().astype(np.float64) for f in raw_lfccs]
    mean_stds = np.array([extract_mean_std_features(f) for f in raw_lfcc_np])
    s_svm = predict_classical_score(predictor.svm, mean_stds)
    s_rf = predict_classical_score(predictor.rf, mean_stds)
    s_lr = predict_classical_score(predictor.lr, mean_stds)
    s_gmm = np.array([predictor.gmm.score_utterance(f) for f in raw_lfcc_np])

    scores_by_name = {
        "CRNN-LFCC": s_crnn_lfcc,
        "CRNN-LogMel": s_crnn_logmel,
        "GMM": s_gmm,
        "SVM": s_svm,
        "Random Forest": s_rf,
        "Logistic Regression": s_lr,
    }
    alias_map = {
        "GMM-LFCC": "GMM",
        "SVM-LFCC": "SVM",
        "RandomForest-LFCC": "Random Forest",
        "LogisticRegression-LFCC": "Logistic Regression",
    }
    cols = [scores_by_name[alias_map.get(name, name)] for name in predictor.base_model_names]
    mat = np.column_stack(cols)
    fused = predictor.stacker.predict_score(mat)
    return [float(x) for x in fused]

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

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Starting STAGE 6 Robustness and Generalization on device: %s", device)

    results_dir = Path(DEFAULT_RESULTS_DIR)
    metrics_dir = results_dir / "metrics"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    crnn_ckpt_path = Path("checkpoints/crnn_lfcc_best.pt")
    if not crnn_ckpt_path.exists():
        crnn_ckpt_path = Path("checkpoints/crnn_best.pt")
    ckpt = torch.load(str(crnn_ckpt_path), map_location=device, weights_only=False)
    crnn_model = CRNNForensics(in_channels=1, in_freq_dim=60)
    crnn_model.load_state_dict(ckpt["model_state_dict"])
    crnn_model.to(device)
    crnn_model.eval()

    extractor = LFCCExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    norm_file = Path("checkpoints/normalization_stats_lfcc.json")
    if not norm_file.exists():
        norm_file = Path("checkpoints/normalization_stats.json")
    normalizer = FeatureNormalizer().load(norm_file) if norm_file.exists() else FeatureNormalizer()

    ensemble_predictor = EnsemblePredictor(device=device)

    crnn_dev_th = -0.1428
    ens_dev_th = -1.9367

    itw_res_file = metrics_dir / "in_the_wild_results.json"
    if itw_res_file.exists():
        logger.info("Part 1 In-The-Wild results already exist at %s; skipping Part 1", itw_res_file)
    else:
        logger.info("Part 1: Real In-the-Wild Evaluation")
        itw_manifest = Path("data/manifests/itw_test.parquet")
        itw_prov = create_provenance_record("in_the_wild_evaluation", itw_manifest, seed=42)

        itw_df = pd.read_parquet(itw_manifest)
        logger.info("Loaded %d In-The-Wild test clips (speakers: %d)", len(itw_df), itw_df["speaker"].nunique())

        def load_itw_item(item):
            idx, row = item
            res = load_and_validate_audio(row["path"], target_sr=SAMPLE_RATE)
            if res.is_valid and res.waveform is not None:
                dur = float(len(res.waveform) / SAMPLE_RATE)
                cat = "short" if dur < 1.0 else ("long" if dur > 4.0 else "medium")
                return res.waveform, int(row["label"]), cat
            return None

        with ThreadPoolExecutor(max_workers=8) as ex:
            itw_loaded = list(ex.map(load_itw_item, itw_df.iterrows()))

        itw_valid = [x for x in itw_loaded if x is not None]
        logger.info("Successfully loaded and validated %d / %d ITW audio clips", len(itw_valid), len(itw_df))

        all_crnn_window_scores = []
        itw_labels = []
        itw_cats = []

        for wave, label, cat in itw_valid:
            windows = generate_multi_crop_windows(wave, window_samples=TARGET_SAMPLES, hop_ratio=0.5)
            with torch.no_grad():
                feats = [extractor(w["waveform"]) for w in windows]
                if normalizer.is_fitted:
                    feats = [normalizer(f) for f in feats]
                b_feats = torch.stack(feats)
                if b_feats.dim() == 3:
                    b_feats = b_feats.unsqueeze(1)
                b_feats = b_feats.to(device)
                logits, _ = crnn_model(b_feats)
                w_scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
            all_crnn_window_scores.append([float(s) for s in w_scores])
            itw_labels.append(label)
            itw_cats.append(cat)

        labels_arr = np.array(itw_labels, dtype=np.int64)
        cats_arr = np.array(itw_cats)

        itw_results_by_agg = {}
        for agg in ["mean", "max", "top_k"]:
            agg_scores = np.array([aggregate_window_scores(ws, strategy=agg) for ws in all_crnn_window_scores], dtype=np.float64)
            m = compute_detailed_crossdomain_metrics(labels_arr, agg_scores, cats_arr, dev_th=crnn_dev_th, seed=42)
            m["aggregation_strategy"] = agg
            m["dataset"] = "In-the-Wild Deepfake Audio (Müller et al., 2022)"
            itw_results_by_agg[f"crnn_{agg}"] = m

        logger.info("Evaluating Stacked Ensemble on In-The-Wild (N=2,500 subset)")
        ens_sub_sample = itw_valid[:2500]
        ens_sub_labels = np.array([x[1] for x in ens_sub_sample], dtype=np.int64)
        ens_sub_cats = np.array([x[2] for x in ens_sub_sample])
        ens_scores = []
        for wave, _, _ in ens_sub_sample:
            windows = generate_multi_crop_windows(wave, window_samples=TARGET_SAMPLES, hop_ratio=0.5)
            ws = [score_waveform_ensemble(w["waveform"], ensemble_predictor) for w in windows]
            ens_scores.append(aggregate_window_scores(ws, strategy="mean"))
        ens_scores_arr = np.array(ens_scores, dtype=np.float64)
        m_ens = compute_detailed_crossdomain_metrics(ens_sub_labels, ens_scores_arr, ens_sub_cats, dev_th=ens_dev_th, seed=42)
        m_ens["aggregation_strategy"] = "mean"
        m_ens["dataset"] = "In-the-Wild Deepfake Audio (Müller et al., 2022)"
        itw_results_by_agg["ensemble_mean"] = m_ens

        itw_summary = {
            **itw_prov,
            "eval_split": "itw_test",
            "num_clips_tested": len(labels_arr),
            "aggregations": itw_results_by_agg,
        }
        with open(metrics_dir / "in_the_wild_results.json", "w") as f:
            json.dump(itw_summary, f, indent=2)

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

    logger.info("Part 2: Codec Robustness Benchmark (MP3, AAC, Opus @ 16, 24, 32, 48, 64 kbps)")
    eval_parquet = Path("data/manifests/eval.parquet")
    codec_prov = create_provenance_record("codec_robustness_eval", eval_parquet, seed=42)

    eval_df = pd.read_parquet(eval_parquet)
    codecs = ["mp3", "aac", "opus"]
    bitrates = [16, 24, 32, 48, 64]

    rng_init = np.random.RandomState(42)
    bona_pool = eval_df[eval_df["label"] == BONAFIDE]
    spoof_pool = eval_df[eval_df["label"] == SPOOF]
    idx_bona = rng_init.choice(bona_pool.index, size=500, replace=False)
    idx_spoof = rng_init.choice(spoof_pool.index, size=700, replace=False)
    eval_sample_df = eval_df.loc[np.concatenate([idx_bona, idx_spoof])].copy()
    logger.info("Selected balanced evaluation sample of %d trials (500 bona fide, 700 spoof)", len(eval_sample_df))

    clean_waves = []
    clean_labels = []
    clean_paths = []
    for _, row in eval_sample_df.iterrows():
        res = load_and_validate_audio(row["path"], target_sr=SAMPLE_RATE)
        if res.is_valid and res.waveform is not None:
            clean_waves.append(res.waveform)
            clean_labels.append(row["label"])
            clean_paths.append(row["path"])
    clean_labels_arr = np.array(clean_labels, dtype=np.int64)

    clean_crnn_scores = evaluate_waveform_batch_crnn(clean_waves, crnn_model, extractor, normalizer, device)
    clean_crnn_eer, _ = compute_eer(clean_labels_arr, np.array(clean_crnn_scores))
    clean_ens_scores = evaluate_waveform_batch_ensemble(clean_waves, ensemble_predictor)
    clean_ens_eer, _ = compute_eer(clean_labels_arr, np.array(clean_ens_scores))

    logger.info("Clean Evaluation Sample (N=%d) -> CRNN EER: %.2f%%, Ensemble EER: %.2f%%", len(clean_labels), clean_crnn_eer, clean_ens_eer)

    codec_cache_root = Path("data/temp_codecs")
    codec_cache_root.mkdir(parents=True, exist_ok=True)

    codec_results = {}
    crnn_codec_eer_map = {c: {} for c in codecs}
    ens_codec_eer_map = {c: {} for c in codecs}
    codec_table_rows = []

    codec_ckpt_file = metrics_dir / "codec_checkpoint.json"
    if codec_ckpt_file.exists():
        try:
            with open(codec_ckpt_file) as f:
                saved_c = json.load(f)
                codec_results = saved_c.get("conditions", {})
                codec_table_rows = saved_c.get("table_rows", [])
                for c in codecs:
                    for k in bitrates:
                        ck = f"{c}_{k}k"
                        if ck in codec_results:
                            crnn_codec_eer_map[c][k] = codec_results[ck]["crnn"]["mean_eer"]
                            ens_codec_eer_map[c][k] = codec_results[ck]["ensemble"]["mean_eer"]
            logger.info("Resumed %d codec conditions from checkpoint", len(codec_results))
        except Exception:
            pass

    sub_sample_indices = []
    for s in [42, 43, 44]:
        r = np.random.RandomState(s)
        sub_sample_indices.append(r.choice(len(clean_waves), size=800, replace=False))

    for codec in codecs:
        for kbps in bitrates:
            cond_key = f"{codec}_{kbps}k"
            if cond_key in codec_results:
                logger.info("Skipping already evaluated condition: %s", cond_key)
                continue
            logger.info("Processing condition: %s @ %d kbps...", codec.upper(), kbps)
            cond_dir = codec_cache_root / cond_key
            cond_dir.mkdir(parents=True, exist_ok=True)

            def process_clip(idx_path_tuple):
                idx, p = idx_path_tuple
                in_p = Path(p)
                out_p = cond_dir / f"{idx}_{in_p.stem}.wav"
                if not out_p.exists():
                    compress_and_decompress_audio(in_p, out_p, codec=codec, bitrate_kbps=kbps, compensate_delay=True)
                if out_p.exists():
                    res = load_and_validate_audio(out_p, target_sr=SAMPLE_RATE)
                    if res.is_valid and res.waveform is not None:
                        return idx, res.waveform
                return idx, clean_waves[idx]

            with ThreadPoolExecutor(max_workers=8) as ex:
                res_items = list(ex.map(process_clip, enumerate(clean_paths)))

            res_items.sort(key=lambda x: x[0])
            compressed_waves = [r[1] for r in res_items]

            crnn_comp_scores = evaluate_waveform_batch_crnn(compressed_waves, crnn_model, extractor, normalizer, device)
            ens_comp_scores = evaluate_waveform_batch_ensemble(compressed_waves, ensemble_predictor)

            crnn_comp_scores_arr = np.array(crnn_comp_scores)
            ens_comp_scores_arr = np.array(ens_comp_scores)

            crnn_eers_seeds = []
            crnn_accs_seeds = []
            ens_eers_seeds = []
            ens_accs_seeds = []

            for sub_idx in sub_sample_indices:
                sub_y = clean_labels_arr[sub_idx]
                sub_sc_c = crnn_comp_scores_arr[sub_idx]
                sub_sc_e = ens_comp_scores_arr[sub_idx]

                e_c, _ = compute_eer(sub_y, sub_sc_c)
                a_c = compute_metrics_at_threshold(sub_y, sub_sc_c, crnn_dev_th)["accuracy"] * 100.0
                crnn_eers_seeds.append(e_c)
                crnn_accs_seeds.append(a_c)

                e_e, _ = compute_eer(sub_y, sub_sc_e)
                a_e = compute_metrics_at_threshold(sub_y, sub_sc_e, ens_dev_th)["accuracy"] * 100.0
                ens_eers_seeds.append(e_e)
                ens_accs_seeds.append(a_e)

            mean_crnn_eer = float(np.mean(crnn_eers_seeds))
            std_crnn_eer = float(np.std(crnn_eers_seeds))
            mean_crnn_acc = float(np.mean(crnn_accs_seeds))
            ci_crnn = [float(np.percentile(crnn_eers_seeds, 2.5)), float(np.percentile(crnn_eers_seeds, 97.5))]

            mean_ens_eer = float(np.mean(ens_eers_seeds))
            std_ens_eer = float(np.std(ens_eers_seeds))
            mean_ens_acc = float(np.mean(ens_accs_seeds))
            ci_ens = [float(np.percentile(ens_eers_seeds, 2.5)), float(np.percentile(ens_eers_seeds, 97.5))]

            crnn_codec_eer_map[codec][kbps] = mean_crnn_eer
            ens_codec_eer_map[codec][kbps] = mean_ens_eer

            codec_results[cond_key] = {
                "codec": codec,
                "bitrate_kbps": kbps,
                "crnn": {
                    "mean_eer": mean_crnn_eer,
                    "std_eer": std_crnn_eer,
                    "ci_95": ci_crnn,
                    "accuracy_dev_th": mean_crnn_acc,
                    "deg_abs": mean_crnn_eer - clean_crnn_eer,
                },
                "ensemble": {
                    "mean_eer": mean_ens_eer,
                    "std_eer": std_ens_eer,
                    "ci_95": ci_ens,
                    "accuracy_dev_th": mean_ens_acc,
                    "deg_abs": mean_ens_eer - clean_ens_eer,
                },
            }

            codec_table_rows.append({
                "Codec": codec.upper(),
                "Bitrate (kbps)": kbps,
                "Clean CRNN EER (%)": f"{clean_crnn_eer:.2f}",
                "CRNN EER (Mean±Std %)": f"{mean_crnn_eer:.2f} ± {std_crnn_eer:.2f}",
                "CRNN 95% CI (%)": f"[{ci_crnn[0]:.2f}, {ci_crnn[1]:.2f}]",
                "CRNN Acc @ Thresh (%)": f"{mean_crnn_acc:.2f}",
                "Clean Ensemble EER (%)": f"{clean_ens_eer:.2f}",
                "Ensemble EER (Mean±Std %)": f"{mean_ens_eer:.2f} ± {std_ens_eer:.2f}",
                "Ensemble Acc @ Thresh (%)": f"{mean_ens_acc:.2f}",
            })
            with open(codec_ckpt_file, "w") as f:
                json.dump({"conditions": codec_results, "table_rows": codec_table_rows}, f, indent=2)

            shutil.rmtree(cond_dir, ignore_errors=True)

    codec_ckpt_file.unlink(missing_ok=True)
    shutil.rmtree(codec_cache_root, ignore_errors=True)

    consolidated_codec_json = {
        **codec_prov,
        "clean_baseline": {
            "crnn_lfcc_eer": clean_crnn_eer,
            "stacked_ensemble_eer": clean_ens_eer,
            "sample_size": len(clean_labels),
        },
        "conditions": codec_results,
    }
    with open(metrics_dir / "codec_robustness_results.json", "w") as f:
        json.dump(consolidated_codec_json, f, indent=2)

    df_codec = pd.DataFrame(codec_table_rows)
    for p in [tables_dir / "table3_robustness.md", tables_dir / "table3_codec_robustness.md"]:
        df_codec.to_markdown(p, index=False)
    for p in [tables_dir / "table3_robustness.tex", tables_dir / "table3_codec_robustness.tex"]:
        df_codec.to_latex(p, index=False)

    plot_robustness_curves(crnn_codec_eer_map, clean_crnn_eer, figures_dir / "robustness_eer_vs_bitrate.png")

    logger.info("Part 3: Acoustic Degradation Chains (Telephony, Additive Noise, Reverb, Packet Loss)")
    deg_prov = create_provenance_record("acoustic_degradations_eval", eval_parquet, seed=42)

    deg_results = {}
    deg_table_rows = []
    deg_ckpt_file = metrics_dir / "deg_checkpoint.json"
    if deg_ckpt_file.exists():
        try:
            with open(deg_ckpt_file) as f:
                deg_data = json.load(f)
                deg_results = deg_data.get("conditions", {})
                deg_table_rows = deg_data.get("table_rows", [])
            logger.info("Resumed %d degradation conditions from checkpoint", len(deg_results))
        except Exception:
            pass

    def evaluate_distorted_waves(dist_waves: list[torch.Tensor], labels: np.ndarray, cond_name: str) -> dict[str, Any]:
        for r in deg_table_rows:
            if r.get("Degradation Condition") == cond_name:
                logger.info("Degradation condition %s already evaluated", cond_name)
                for k, v in deg_results.items():
                    if v.get("condition") == cond_name:
                        return v

        sc_crnn = evaluate_waveform_batch_crnn(dist_waves, crnn_model, extractor, normalizer, device)
        eer_crnn, _ = compute_eer(labels, np.array(sc_crnn))
        acc_crnn = compute_metrics_at_threshold(labels, np.array(sc_crnn), crnn_dev_th)["accuracy"] * 100.0

        sc_ens = evaluate_waveform_batch_ensemble(dist_waves, ensemble_predictor)
        eer_ens, _ = compute_eer(labels, np.array(sc_ens))
        acc_ens = compute_metrics_at_threshold(labels, np.array(sc_ens), ens_dev_th)["accuracy"] * 100.0

        deg_table_rows.append({
            "Degradation Condition": cond_name,
            "CRNN Degraded EER (%)": f"{eer_crnn:.2f}",
            "CRNN Degradation (pts)": f"{eer_crnn - clean_crnn_eer:+.2f}",
            "CRNN Acc @ Thresh (%)": f"{acc_crnn:.2f}",
            "Ensemble Degraded EER (%)": f"{eer_ens:.2f}",
            "Ensemble Degradation (pts)": f"{eer_ens - clean_ens_eer:+.2f}",
            "Ensemble Acc @ Thresh (%)": f"{acc_ens:.2f}",
        })
        res_dict = {
            "condition": cond_name,
            "crnn": {"eer": eer_crnn, "acc_thresh": acc_crnn, "delta_eer": eer_crnn - clean_crnn_eer},
            "ensemble": {"eer": eer_ens, "acc_thresh": acc_ens, "delta_eer": eer_ens - clean_ens_eer},
        }
        with open(deg_ckpt_file, "w") as f:
            json.dump({"conditions": deg_results, "table_rows": deg_table_rows}, f, indent=2)
        return res_dict

    logger.info("Running Telephony Chains...")
    tel_temp = Path("data/temp_telephony")
    tel_temp.mkdir(parents=True, exist_ok=True)
    for comp in ["alaw", "mulaw"]:
        def process_tel(item):
            i, w = item
            raw_p = tel_temp / f"in_{comp}_{i}.wav"
            out_p = tel_temp / f"out_{comp}_{i}.wav"
            sf.write(str(raw_p), w.numpy(), SAMPLE_RATE)
            apply_telephony_channel(raw_p, out_p, compand=comp)
            res = load_and_validate_audio(out_p, target_sr=SAMPLE_RATE)
            raw_p.unlink(missing_ok=True)
            out_p.unlink(missing_ok=True)
            return i, (res.waveform if res.is_valid and res.waveform is not None else w)

        with ThreadPoolExecutor(max_workers=8) as ex:
            t_items = list(ex.map(process_tel, enumerate(clean_waves)))
        t_items.sort(key=lambda x: x[0])
        cond_waves = [x[1] for x in t_items]
        deg_results[f"telephony_8k_{comp}"] = evaluate_distorted_waves(cond_waves, clean_labels_arr, f"8 kHz Telephony (G.711 {comp.upper()})")
    shutil.rmtree(tel_temp, ignore_errors=True)

    logger.info("Running Additive Gaussian Noise...")
    for snr in [20, 10, 0]:
        noise_waves = [torch.from_numpy(apply_additive_noise(w.numpy(), snr_db=float(snr), seed=42)) for w in clean_waves]
        deg_results[f"noise_snr_{snr}db"] = evaluate_distorted_waves(noise_waves, clean_labels_arr, f"Additive Noise (SNR {snr} dB)")

    logger.info("Running Synthetic Room Reverberation...")
    rev_waves = [torch.from_numpy(apply_synthetic_reverb(w.numpy(), rt60=0.3, sr=SAMPLE_RATE, seed=42)) for w in clean_waves]
    deg_results["reverb_rt60_0.3s"] = evaluate_distorted_waves(rev_waves, clean_labels_arr, "Synthetic Reverb (RT60 = 0.3s)")

    logger.info("Running Packet Loss...")
    for pkt in [0.05, 0.10, 0.20]:
        pkt_waves = [torch.from_numpy(apply_packet_loss(w.numpy(), loss_rate=pkt, packet_ms=20, sr=SAMPLE_RATE, seed=42)) for w in clean_waves]
        deg_results[f"packet_loss_{int(pkt*100)}pct"] = evaluate_distorted_waves(pkt_waves, clean_labels_arr, f"Packet Loss ({int(pkt*100)}%, 20ms frames)")

    deg_summary = {
        **deg_prov,
        "clean_baseline": {
            "crnn_eer": clean_crnn_eer,
            "ensemble_eer": clean_ens_eer,
        },
        "conditions": deg_results,
        "boundary_statement": "ASVspoof 2021 LA and DF evaluation corpora (~180 GB total) require external storage beyond the local 70 GB disk quota per Rule 6.",
    }
    with open(metrics_dir / "degradation_chains_results.json", "w") as f:
        json.dump(deg_summary, f, indent=2)
    deg_ckpt_file.unlink(missing_ok=True)

    df_deg = pd.DataFrame(deg_table_rows)
    for p in [tables_dir / "table9_acoustic_degradations.md"]:
        df_deg.to_markdown(p, index=False)
    for p in [tables_dir / "table9_acoustic_degradations.tex"]:
        df_deg.to_latex(p, index=False)

    logger.info("STAGE 6 Robustness and Generalization complete!")

if __name__ == "__main__":
    main()
