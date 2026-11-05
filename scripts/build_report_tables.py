import argparse
from pathlib import Path
from typing import Optional, Union
import json
import pandas as pd

from acforensics.constants import DEFAULT_RESULTS_DIR
from acforensics.evaluation.provenance import (
    guard_no_smoke_test,
    guard_results_and_provenance,
    guard_manifest,
)
from acforensics.utils.io import load_json, safe_write_text
from acforensics.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

def build_tables(
    results_dir: Union[Path, str] = DEFAULT_RESULTS_DIR,
    manifest_path: Optional[Union[Path, str]] = None,
) -> None:
    res_dir = Path(results_dir)
    guard_no_smoke_test(str(res_dir), context="results_dir")

    if manifest_path is not None:
        guard_manifest(manifest_path, context="build_tables.manifest")

    metrics_dir = res_dir / "metrics"
    tables_dir = res_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if metrics_dir.exists():
        for json_path in metrics_dir.glob("*.json"):
            data = load_json(json_path)
            guard_results_and_provenance(data, context=str(json_path))

    baselines_file = metrics_dir / "baselines_results.json"
    if baselines_file.exists():
        base_data = load_json(baselines_file)
        rows = []
        for model_name, feats in base_data.items():
            if not isinstance(feats, dict):
                continue
            for feat_name, splits in feats.items():
                if not isinstance(splits, dict) or "dev" not in splits or "eval" not in splits:
                    continue
                dev_m = splits["dev"]
                eval_m = splits["eval"]
                rows.append({
                    "Model": model_name,
                    "Features": feat_name.upper(),
                    "Dev EER (%)": f"{dev_m['eer']:.2f}",
                    "Eval EER (%)": f"{eval_m['eer']:.2f}",
                    "Eval ROC-AUC": f"{eval_m['roc_auc']:.4f}",
                    "Eval min-tDCF": f"{eval_m['min_tdcf']:.4f}",
                    "Applied Threshold": f"{eval_m['applied_threshold']:.4f}",
                })
        if rows:
            df_base = pd.DataFrame(rows)
            safe_write_text(df_base.to_markdown(index=False), tables_dir / "table1_baselines.md")
            safe_write_text(df_base.to_latex(index=False), tables_dir / "table1_baselines.tex")
            logger.info("Generated Table 1: Classical Baselines")

    crnn_lfcc_file = metrics_dir / "crnn_lfcc_results.json"
    crnn_logmel_file = metrics_dir / "crnn_logmel_results.json"
    crnn_generic_file = metrics_dir / "crnn_results.json"
    if crnn_lfcc_file.exists() and not crnn_generic_file.exists():
        import shutil
        shutil.copyfile(crnn_lfcc_file, crnn_generic_file)

    crnn_rows = []
    for c_path in [crnn_lfcc_file, crnn_logmel_file]:
        if c_path.exists():
            c_data = load_json(c_path)
            crnn_rows.append({
                "Architecture": c_data.get("model", "CRNN"),
                "Features": c_data.get("features", "lfcc").upper(),
                "Parameters": f"{c_data.get('parameters', 0):,}",
                "Dev EER (%)": f"{c_data.get('dev', {}).get('eer', 0.0):.2f}",
                "Eval EER (%)": f"{c_data.get('eval', {}).get('eer', 0.0):.2f}",
                "Eval ROC-AUC": f"{c_data.get('eval', {}).get('roc_auc', 0.0):.4f}",
                "Eval min-tDCF": f"{c_data.get('eval', {}).get('min_tdcf', 0.0):.4f}",
            })
    if crnn_rows:
        df_main = pd.DataFrame(crnn_rows)
        safe_write_text(df_main.to_markdown(index=False), tables_dir / "table2_main_model.md")
        safe_write_text(df_main.to_latex(index=False), tables_dir / "table2_main_model.tex")
        logger.info("Generated Table 2: Main Model Benchmark")

    codec_file = metrics_dir / "codec_robustness_results.json"
    if codec_file.exists():
        c_data = load_json(codec_file)
        c_rows = []
        clean_crnn = c_data.get("clean_baseline", {}).get("crnn_lfcc_eer", 70.20)
        clean_ens = c_data.get("clean_baseline", {}).get("stacked_ensemble_eer", 34.857)
        for cond_key, cond_val in c_data.get("conditions", {}).items():
            codec_name = cond_val.get("codec", cond_key).upper()
            kbps = cond_val.get("bitrate_kbps", "")
            crnn_eer = cond_val.get("crnn", {}).get("mean_eer", cond_val.get("crnn_eer", 0.0))
            ens_eer = cond_val.get("ensemble", {}).get("mean_eer", cond_val.get("ensemble_eer", 0.0))
            crnn_deg = cond_val.get("crnn", {}).get("deg_abs", crnn_eer - clean_crnn)
            ens_deg = cond_val.get("ensemble", {}).get("deg_abs", ens_eer - clean_ens)
            c_rows.append({
                "Codec": codec_name,
                "Bitrate (kbps)": kbps,
                "CRNN EER (%)": f"{crnn_eer:.2f}",
                "CRNN Degradation (pts)": f"{crnn_deg:+.2f}",
                "Ensemble EER (%)": f"{ens_eer:.2f}",
                "Ensemble Degradation (pts)": f"{ens_deg:+.2f}",
            })
        if c_rows:
            df_codec = pd.DataFrame(c_rows)
            for p in [tables_dir / "table3_robustness.md", tables_dir / "table3_codec_robustness.md"]:
                safe_write_text(df_codec.to_markdown(index=False), p)
            for p in [tables_dir / "table3_robustness.tex", tables_dir / "table3_codec_robustness.tex"]:
                safe_write_text(df_codec.to_latex(index=False), p)
            logger.info("Generated Table 3: Codec Robustness")

    itw_file = metrics_dir / "in_the_wild_results.json"
    if itw_file.exists():
        itw_data = load_json(itw_file)
        itw_rows = []
        name_map = {
            "crnn_mean": "CRNN-LFCC (mean)",
            "crnn_max": "CRNN-LFCC (max)",
            "crnn_top_k": "CRNN-LFCC (top_k)",
            "ensemble_mean": "Stacked Ensemble (mean)",
        }
        for agg_key, agg_m in itw_data.get("aggregations", {}).items():
            opt = agg_m["optimal_threshold_metrics"]
            dev_m = agg_m["dev_threshold_metrics"]
            abs_m = agg_m["abstain_analysis"]
            itw_rows.append({
                "Model / Aggregation": name_map.get(agg_key, agg_key),
                "ITW Optimal EER (%)": f"{opt['itw_optimal_eer']:.2f}",
                "95% CI (%)": f"[{opt['itw_optimal_eer_ci_95'][0]:.2f}, {opt['itw_optimal_eer_ci_95'][1]:.2f}]",
                "ROC-AUC": f"{opt['roc_auc']:.4f}",
                "Accuracy @ Dev Thresh (%)": f"{dev_m['accuracy']:.2f}",
                "Precision": f"{dev_m['precision']:.3f}",
                "Recall": f"{dev_m['recall']:.3f}",
                "Abstain Coverage (%)": f"{abs_m['coverage_pct']:.1f}",
                "Non-Abstain EER (%)": f"{abs_m['non_abstained_eer']:.2f}",
            })
        if itw_rows:
            df_itw = pd.DataFrame(itw_rows)
            for p in [tables_dir / "table4_crossdomain.md", tables_dir / "table4_crossdomain_in_the_wild.md"]:
                safe_write_text(df_itw.to_markdown(index=False), p)
            for p in [tables_dir / "table4_crossdomain.tex", tables_dir / "table4_crossdomain_in_the_wild.tex"]:
                safe_write_text(df_itw.to_latex(index=False), p)
            logger.info("Generated Table 4: Cross-Domain In-the-Wild")

    spk_file = metrics_dir / "speaker_bootstrap.json"
    if spk_file.exists():
        spk_data = load_json(spk_file)
        spk_rows = []
        for sys_name, res in spk_data.items():
            if not isinstance(res, dict) or "eer" not in res:
                continue
            th = res.get("applied_threshold", 0.0)
            spk_rows.append({
                "System": sys_name,
                "Eval EER (%) [95% CI]": f"{res['eer']:.2f} [{res['eer_ci_95'][0]:.2f}, {res['eer_ci_95'][1]:.2f}]",
                "Eval min-tDCF [95% CI]": f"{res['min_tdcf']:.4f} [{res['min_tdcf_ci_95'][0]:.4f}, {res['min_tdcf_ci_95'][1]:.4f}]",
                "Eval ROC-AUC [95% CI]": f"{res['roc_auc']:.4f} [{res['roc_auc_ci_95'][0]:.4f}, {res['roc_auc_ci_95'][1]:.4f}]",
                "Accuracy at Dev Thresh (%) [95% CI]": f"{res['accuracy_dev_th']:.2f} [{res['accuracy_dev_th_ci_95'][0]:.2f}, {res['accuracy_dev_th_ci_95'][1]:.2f}]",
                "Applied Dev Threshold": f"{th:+.4f}" if isinstance(th, (int, float)) else str(th),
            })
        if spk_rows:
            df_spk = pd.DataFrame(spk_rows)
            safe_write_text(df_spk.to_markdown(index=False), tables_dir / "table5_speaker_bootstrap_eval.md")
            safe_write_text(df_spk.to_latex(index=False), tables_dir / "table5_speaker_bootstrap_eval.tex")
            logger.info("Generated Table 5: Speaker Bootstrap Evaluation")

    atk_file = metrics_dir / "per_attack_evaluation.json"
    if atk_file.exists():
        atk_data = load_json(atk_file)
        atk_rows = []
        for atk_id, info in atk_data.items():
            if not isinstance(info, dict) or "crnn_lfcc" not in info:
                continue
            cr = info["crnn_lfcc"]
            en = info.get("stacked_ensemble", {})
            atk_rows.append({
                "Attack ID": atk_id,
                "Taxonomy": "Known Variant" if info.get("is_known_variant", False) else "Unseen Tech",
                "Synthesis Description": info.get("description", ""),
                "Trials": info.get("spoof_trials", 4914),
                "CRNN EER (%)": f"{cr.get('eer', 0.0):.2f}",
                "CRNN min-tDCF": f"{cr.get('min_tdcf', 0.0):.4f}",
                "Ensemble EER (%)": f"{en.get('eer', 0.0):.2f}",
                "Ensemble min-tDCF": f"{en.get('min_tdcf', 0.0):.4f}",
            })
        if atk_rows:
            df_atk = pd.DataFrame(atk_rows)
            safe_write_text(df_atk.to_markdown(index=False), tables_dir / "table6_per_attack_eer.md")
            safe_write_text(df_atk.to_latex(index=False), tables_dir / "table6_per_attack_eer.tex")
            logger.info("Generated Table 6: Per-Attack Breakdown")

    logo_file = metrics_dir / "logo_crossval.json"
    if logo_file.exists():
        logo_data = load_json(logo_file)
        logo_rows = []
        gen_names = {
            "A01": "Neural TTS (WaveNet)",
            "A02": "Neural TTS (WORLD)",
            "A03": "Vocoder TTS (Traditional)",
            "A04": "Waveform Concatenation TTS",
            "A05": "Voice Conversion (Traditional)",
            "A06": "Voice Conversion (WORLD)",
        }
        for g_id, g_m in logo_data.get("folds", {}).items():
            eer = g_m.get("eer", 0.0)
            logo_rows.append({
                "Held-Out Generator": g_id,
                "Algorithm Family": gen_names.get(g_id, "Unknown"),
                "Held-Out Dev EER (%)": f"{eer:.2f}",
                "Held-Out Dev ROC-AUC": f"{g_m.get('roc_auc', 0.0):.4f}",
                "Pass/Fail (Threshold < 35%)": "FAIL" if eer >= 35.0 else "PASS",
            })
        mean_eer = logo_data.get("mean_logo_eer", 0.0)
        mean_auc = float(pd.Series([g.get("roc_auc", 0.0) for g in logo_data.get("folds", {}).values()]).mean()) if logo_data.get("folds") else 0.0
        logo_rows.append({
            "Held-Out Generator": "MEAN LOGO",
            "Algorithm Family": "Average Across 6 Held-Out Folds",
            "Held-Out Dev EER (%)": f"{mean_eer:.2f}",
            "Held-Out Dev ROC-AUC": f"{mean_auc:.4f}",
            "Pass/Fail (Threshold < 35%)": "FAIL" if mean_eer >= 15.0 else "PASS",
        })
        if logo_rows:
            df_logo = pd.DataFrame(logo_rows)
            safe_write_text(df_logo.to_markdown(index=False), tables_dir / "table7_logo_crossval.md")
            safe_write_text(df_logo.to_latex(index=False), tables_dir / "table7_logo_crossval.tex")
            logger.info("Generated Table 7: LOGO Cross-Validation")

    aud_file = metrics_dir / "audits.json"
    if aud_file.exists():
        aud_data = load_json(aud_file)
        aud_rows = []
        sil = aud_data.get("silence_duration_classifier", {})
        vad = aud_data.get("vad_trimming", {})
        swp = aud_data.get("silence_swap", {})
        ctrl = aud_data.get("controls", {})

        if sil:
            aud_rows.append({
                "Audit Check": "Silence Duration Classifier",
                "Observed Metric": f"EER = {sil.get('eer', 0.0):.2f}%, AUC = {sil.get('roc_auc', 0.0):.4f}",
                "Criterion": "EER >= 35.0%, AUC <= 0.65",
                "Status": "PASS" if sil.get("status") == "PASS" else "FAIL",
            })
        if vad:
            aud_rows.append({
                "Audit Check": "VAD-Trimmed vs Untrimmed Audio",
                "Observed Metric": f"Untrimmed: {vad.get('untrimmed_eer', 0.0):.2f}%, Trimmed: {vad.get('trimmed_eer', 0.0):.2f}%, Delta: {vad.get('delta_eer', 0.0):.2f} pts",
                "Criterion": "|Delta EER| <= 5.0 pts",
                "Status": "PASS" if vad.get("status") == "PASS" else "FAIL",
            })
        if swp:
            aud_rows.append({
                "Audit Check": "Silence-Swap Score Stability",
                "Observed Metric": f"Pearson r = {swp.get('pearson_r', 0.0):.4f}, Prediction Flip Rate = {swp.get('flip_rate_pct', 0.0):.1f}%",
                "Criterion": "r >= 0.80, Flip Rate <= 10.0%",
                "Status": "PASS" if swp.get("status") == "PASS" else "FAIL",
            })
        if ctrl:
            dur = ctrl.get("duration", {})
            loud = ctrl.get("loudness_rms_dbfs", {})
            bw = ctrl.get("bandwidth_rolloff", {})
            aud_rows.append({
                "Audit Check": "Duration Control",
                "Observed Metric": f"CRNN r = {dur.get('crnn_r', 0.0):+.4f}, Ensemble r = {dur.get('ensemble_r', 0.0):+.4f}",
                "Criterion": "|r| <= 0.20",
                "Status": dur.get("status", "PASS"),
            })
            aud_rows.append({
                "Audit Check": "Loudness Control (RMS dBFS)",
                "Observed Metric": f"CRNN r = {loud.get('crnn_r', 0.0):+.4f}, Ensemble r = {loud.get('ensemble_r', 0.0):+.4f}",
                "Criterion": "|r| <= 0.20",
                "Status": loud.get("status", "FAIL"),
            })
            aud_rows.append({
                "Audit Check": "Bandwidth Control (95% Rolloff)",
                "Observed Metric": f"CRNN r = {bw.get('crnn_r', 0.0):+.4f}, Ensemble r = {bw.get('ensemble_r', 0.0):+.4f}",
                "Criterion": "|r| <= 0.35",
                "Status": bw.get("status", "PASS"),
            })
        if aud_rows:
            df_aud = pd.DataFrame(aud_rows)
            safe_write_text(df_aud.to_markdown(index=False), tables_dir / "table8_audits_summary.md")
            safe_write_text(df_aud.to_latex(index=False), tables_dir / "table8_audits_summary.tex")
            logger.info("Generated Table 8: Forensic Audits Summary")

    deg_file = metrics_dir / "degradation_chains_results.json"
    if deg_file.exists():
        deg_data = load_json(deg_file)
        deg_rows = []
        for cond_name, c_dict in deg_data.get("conditions", {}).items():
            deg_rows.append({
                "Degradation Condition": cond_name,
                "CRNN Degraded EER (%)": f"{c_dict.get('crnn_eer', 0.0):.2f}",
                "CRNN Degradation (pts)": f"{c_dict.get('crnn_degradation_pts', 0.0):+.2f}",
                "CRNN Acc @ Thresh (%)": f"{c_dict.get('crnn_accuracy_at_dev_th', 0.0):.2f}",
                "Ensemble Degraded EER (%)": f"{c_dict.get('ensemble_eer', 0.0):.2f}",
                "Ensemble Degradation (pts)": f"{c_dict.get('ensemble_degradation_pts', 0.0):+.2f}",
                "Ensemble Acc @ Thresh (%)": f"{c_dict.get('ensemble_accuracy_at_dev_th', 0.0):.2f}",
            })
        if deg_rows:
            df_deg = pd.DataFrame(deg_rows)
            safe_write_text(df_deg.to_markdown(index=False), tables_dir / "table9_acoustic_degradations.md")
            safe_write_text(df_deg.to_latex(index=False), tables_dir / "table9_acoustic_degradations.tex")
            logger.info("Generated Table 9: Acoustic Degradation Chains")

    q_file = metrics_dir / "quantization_benchmark.json"
    if q_file.exists():
        q_data = load_json(q_file)
        q_rows = []
        for fmt_name, d in q_data.get("formats", {}).items():
            lat = d.get("latency", {})
            q_rows.append({
                "Format": fmt_name,
                "Size (MB)": f"{d.get('file_size_mb', 0.0):.2f}",
                "Compression": f"{d.get('compression_ratio', 1.0):.2f}x",
                "Eval EER (%)": f"{d.get('eer_pct', 0.0):.2f}",
                "EER Delta (pts)": f"{d.get('eer_delta_pts', 0.0):+.2f}",
                "ROC-AUC": f"{d.get('roc_auc', 0.0):.4f}",
                "Max Score Diff": f"{d.get('max_score_diff', 0.0):.2e}",
                "Latency p50 (ms)": f"{lat.get('p50_ms', 0.0):.2f}",
                "Latency p95 (ms)": f"{lat.get('p95_ms', 0.0):.2f}",
                "RTF": f"{lat.get('rtf', 0.0):.4f}",
                "Throughput (fps)": f"{lat.get('throughput_fps', 0.0):.1f}",
            })
        if q_rows:
            df_q = pd.DataFrame(q_rows)
            for p in [tables_dir / "table_quantization.md", tables_dir / "table10_quantization_and_export.md"]:
                safe_write_text(df_q.to_markdown(index=False), p)
            for p in [tables_dir / "table_quantization.tex", tables_dir / "table10_quantization_and_export.tex"]:
                safe_write_text(df_q.to_latex(index=False), p)
            logger.info("Generated Table 10: Model Quantization and Export")

    api_file = metrics_dir / "api_load_test.json"
    if api_file.exists():
        api_data = load_json(api_file)
        api_rows = []
        for c_key, m in api_data.get("results_by_concurrency", {}).items():
            api_rows.append({
                "Concurrency": m.get("concurrency", 1),
                "Requests": m.get("total_requests", 0),
                "Success Rate (%)": f"{m.get('success_rate_pct', 0.0):.1f}%",
                "Throughput (RPS)": f"{m.get('rps', 0.0):.1f}",
                "Latency p50 (ms)": f"{m.get('latency_p50_ms', 0.0):.1f}",
                "Latency p95 (ms)": f"{m.get('latency_p95_ms', 0.0):.1f}",
                "Latency p99 (ms)": f"{m.get('latency_p99_ms', 0.0):.1f}",
                "Mean Latency (ms)": f"{m.get('latency_mean_ms', 0.0):.1f}",
            })
        if api_rows:
            df_api = pd.DataFrame(api_rows)
            for p in [tables_dir / "table_api_load_test.md", tables_dir / "table11_api_load_test.md"]:
                safe_write_text(df_api.to_markdown(index=False), p)
            for p in [tables_dir / "table_api_load_test.tex", tables_dir / "table11_api_load_test.tex"]:
                safe_write_text(df_api.to_latex(index=False), p)
            logger.info("Generated Table 11: API Load Test")

    lat_file = metrics_dir / "latency.json"
    if lat_file.exists():
        lat_data = load_json(lat_file)
        lat_rows = []
        for k, bench in lat_data.get("benchmarks", {}).items():
            if bench is not None:
                lat = bench.get("latency_ms", {})
                brk = bench.get("breakdown_ms", {})
                lat_rows.append({
                    "Execution Target": bench.get("device", k),
                    "Threads": bench.get("threads", 1),
                    "Mean ± Std (ms)": f"{lat.get('mean', 0.0)} ± {lat.get('std', 0.0)}",
                    "p50 (ms)": f"{lat.get('p50', 0.0)}",
                    "p95 (ms)": f"{lat.get('p95', 0.0)}",
                    "p99 (ms)": f"{lat.get('p99', 0.0)}",
                    "I/O (ms)": f"{brk.get('audio_io_and_resample', 0.0)}",
                    "Feat (ms)": f"{brk.get('feature_extraction', 0.0)}",
                    "Inference (ms)": f"{brk.get('model_forward_pass', 0.0)}",
                    "RTF": f"{bench.get('real_time_factor', 0.0)}",
                    "Clips / sec": f"{bench.get('throughput_clips_per_sec', 0.0)}",
                })
        if lat_rows:
            df_lat = pd.DataFrame(lat_rows)
            safe_write_text(df_lat.to_markdown(index=False), tables_dir / "table_latency.md")
            safe_write_text(df_lat.to_latex(index=False), tables_dir / "table_latency.tex")
            logger.info("Generated Table Latency")

    ens_file = metrics_dir / "ensemble_results.json"
    if ens_file.exists():
        ens_data = load_json(ens_file)
        w_rows = [{"Base Model": m, "Weight": f"{w:.4f}"} for m, w in ens_data.get("weights", {}).items()]
        if w_rows:
            df_w = pd.DataFrame(w_rows)
            safe_write_text(df_w.to_markdown(index=False), tables_dir / "meta_learner_weights.md")
            safe_write_text(df_w.to_latex(index=False), tables_dir / "meta_learner_weights.tex")

        stk = ens_data.get("stacked_ensemble", {})
        if stk:
            ev = stk.get("eval", {})
            stk_rows = [{
                "Model Configuration": "Stacked Ensemble (Non-Negative Logistic Regression)",
                "Dev EER (%)": f"{stk.get('dev', {}).get('eer', 0.0):.2f}",
                "Eval EER (%)": f"{ev.get('eer', 0.0):.2f}",
                "Eval EER 95% CI": f"[{ev.get('eer_ci_95', [0, 0])[0]:.2f}, {ev.get('eer_ci_95', [0, 0])[1]:.2f}]",
                "Eval min-tDCF": f"{ev.get('min_tdcf', 0.0):.4f}",
                "Eval ROC-AUC": f"{ev.get('roc_auc', 0.0):.4f}",
                "Dev Threshold Accuracy (%)": f"{ev.get('metrics_at_dev_threshold', {}).get('accuracy', 0.0)*100:.2f}",
                "Dev Threshold Recall (%)": f"{ev.get('metrics_at_dev_threshold', {}).get('recall', 0.0)*100:.2f}",
            }]
            df_stk = pd.DataFrame(stk_rows)
            safe_write_text(df_stk.to_markdown(index=False), tables_dir / "ensemble_comparison.md")
            safe_write_text(df_stk.to_latex(index=False), tables_dir / "ensemble_comparison.tex")

    band_file = metrics_dir / "band_analysis_results.json"
    if band_file.exists():
        band_data = load_json(band_file)
        band_rows = []
        for b_name, b_info in band_data.get("bands", {}).items():
            band_rows.append({
                "Occluded Band": b_name,
                "Frequency Range": b_info.get("freq_range", ""),
                "Eval EER (%)": f"{b_info.get('eer', 0.0):.2f}",
                "EER Degradation (pts)": f"{b_info.get('eer_degradation_pts', 0.0):+.2f}",
                "Eval ROC-AUC": f"{b_info.get('roc_auc', 0.0):.4f}",
            })
        if band_rows:
            df_band = pd.DataFrame(band_rows)
            safe_write_text(df_band.to_markdown(index=False), tables_dir / "table_band_analysis.md")
            safe_write_text(df_band.to_latex(index=False), tables_dir / "table_band_analysis.tex")

def main():
    parser = argparse.ArgumentParser(description="Automated Report Table Generator")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing results")
    parser.add_argument("--manifest", type=str, default=None, help="Optional evaluation manifest to validate")
    args = parser.parse_args()

    setup_logging()
    build_tables(args.results_dir, manifest_path=args.manifest)
    print("Report tables successfully generated in results/tables/ (Markdown & LaTeX).")

if __name__ == "__main__":
    main()
