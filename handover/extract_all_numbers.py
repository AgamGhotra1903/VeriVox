import json
from pathlib import Path

def dump_file(p):
    p = Path(p)
    if p.exists():
        return json.loads(p.read_text())
    return {}

print("=== BASELINES RESULTS ===")
b = dump_file("results/metrics/baselines_results.json")
for m, feats in b.items():
    if isinstance(feats, dict):
        for f, s in feats.items():
            if isinstance(s, dict) and "dev" in s and "eval" in s:
                dev_eer = s["dev"]["eer"]
                eval_eer = s["eval"]["eer"]
                auc = s["eval"]["roc_auc"]
                min_tdcf = s["eval"]["min_tdcf"]
                th = s["eval"]["applied_threshold"]
                print(f"{m} {f}: Dev EER={dev_eer:.2f} Eval EER={eval_eer:.2f} AUC={auc:.4f} min_tdcf={min_tdcf:.4f} th={th:.4f}")

print("\n=== OLD VS NEW ENSEMBLE ===")
ens_new = dump_file("results/metrics/ensemble_results.json")
ens_old = dump_file("handover/evidence/metrics/ensemble_results.json")
print("OLD Stacked Ensemble Eval:", ens_old.get("stacked_ensemble", {}).get("eval", {}).get("eer"), ens_old.get("stacked_ensemble", {}).get("eval", {}).get("min_tdcf"))
print("NEW Stacked Ensemble Eval:", ens_new.get("stacked_ensemble", {}).get("eval", {}).get("eer"), ens_new.get("stacked_ensemble", {}).get("eval", {}).get("min_tdcf"))
print("OLD CRNN Single Eval:", ens_old.get("best_single_model", {}).get("eval", {}).get("eer"), ens_old.get("best_single_model", {}).get("eval", {}).get("min_tdcf"))
print("NEW CRNN Single Eval:", ens_new.get("best_single_model", {}).get("eval", {}).get("eer"), ens_new.get("best_single_model", {}).get("eval", {}).get("min_tdcf"))

print("\n=== OLD VS NEW SPEAKER BOOTSTRAP ===")
spk_new = dump_file("results/metrics/speaker_bootstrap.json")
spk_old = dump_file("handover/evidence/metrics/speaker_bootstrap.json")
for k in ["Stacked Ensemble", "Best Single Model (CRNN-LFCC)", "Baseline Mean Fusion", "LFCC-GMM Baseline"]:
    old_item = spk_old.get(k, {})
    new_item = spk_new.get(k, {})
    print(f"OLD {k}: EER={old_item.get('eer')}, CI={old_item.get('eer_ci_95')}, tdcf={old_item.get('min_tdcf')}, auc={old_item.get('roc_auc')}, acc={old_item.get('accuracy_dev_th')}")
    print(f"NEW {k}: EER={new_item.get('eer')}, CI={new_item.get('eer_ci_95')}, tdcf={new_item.get('min_tdcf')}, auc={new_item.get('roc_auc')}, acc={new_item.get('accuracy_dev_th')}")

print("\n=== OLD VS NEW IN THE WILD ===")
itw_new = dump_file("results/metrics/in_the_wild_results.json")
itw_old = dump_file("handover/evidence/metrics/in_the_wild_results.json")
print("OLD ITW crnn_mean:", itw_old.get("aggregations", {}).get("crnn_mean", {}).get("optimal_threshold_metrics"))
print("NEW ITW crnn_mean:", itw_new.get("aggregations", {}).get("crnn_mean", {}).get("optimal_threshold_metrics"))
print("OLD ITW ensemble_mean:", itw_old.get("aggregations", {}).get("ensemble_mean", {}).get("optimal_threshold_metrics"))
print("NEW ITW ensemble_mean:", itw_new.get("aggregations", {}).get("ensemble_mean", {}).get("optimal_threshold_metrics"))

print("\n=== OLD VS NEW CODEC ===")
cod_new = dump_file("results/metrics/codec_robustness_results.json")
cod_old = dump_file("handover/evidence/metrics/codec_robustness_results.json")
print("OLD Clean EER:", cod_old.get("clean_baseline_eer"), cod_old.get("clean_baseline"))
print("NEW Clean EER:", cod_new.get("clean_baseline"))

print("\n=== OLD VS NEW DEGRADATION ===")
deg_new = dump_file("results/metrics/degradation_chains_results.json")
deg_old = dump_file("handover/evidence/metrics/degradation_chains_results.json")
print("OLD degradation noise 0db:", deg_old.get("conditions", {}).get("noise_snr_0db"))
print("NEW degradation noise 0db:", deg_new.get("conditions", {}).get("noise_snr_0db"))

print("\n=== OLD VS NEW AUDITS ===")
aud_new = dump_file("results/metrics/crnn_silence_audit.json")
aud_old = dump_file("handover/evidence/metrics/audits.json")
print("OLD Audits vad_trimming:", aud_old.get("vad_trimming"))
print("NEW Audits vad_trimming:", aud_new.get("vad_trimming"))

print("\n=== QUANTIZATION AND LATENCY ===")
lat = dump_file("results/metrics/latency_benchmark.json")
q = dump_file("results/metrics/quantization_benchmark.json")
print("Latency CPU single thread:", lat.get("benchmarks", {}).get("cpu_single_thread", {}).get("latency_ms"))
print("Quantization formats:", list(q.get("quantization_results", {}).keys()) if "quantization_results" in q else list(q.keys()))

