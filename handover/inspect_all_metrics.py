import json
from pathlib import Path

def get_json(p):
    path = Path(p)
    if path.exists():
        return json.loads(path.read_text())
    return {}

ms = get_json("results/metrics/crnn_multiseed_results.json")
print("=== MULTISEED ===")
print("Metrics:", ms.get("metrics"))
for r in ms.get("individual_runs", []):
    print(f"Seed {r['seed']}: Dev EER={r['dev']['eer']:.2f}% Eval EER={r['eval']['eer']:.2f}% Eval min-tDCF={r['eval']['min_tdcf']:.4f} Eval AUC={r['eval']['roc_auc']:.4f}")

print("\n=== LOGMEL RETRAINING ===")
logmel = get_json("results/metrics/crnn_logmel_results.json")
print("LogMel best epoch:", logmel.get("best_epoch"))
print("LogMel Dev:", logmel.get("dev"))
print("LogMel Eval:", logmel.get("eval"))

print("\n=== IN THE WILD ===")
itw = get_json("results/metrics/in_the_wild_results.json")
for agg, data in itw.get("aggregations", {}).items():
    opt = data.get("optimal_threshold_metrics", {})
    dev_m = data.get("dev_threshold_metrics", {})
    print(f"{agg}: Opt EER={opt.get('itw_optimal_eer'):.2f}%, Opt CI={opt.get('itw_optimal_eer_ci_95')}, AUC={opt.get('roc_auc'):.4f}, Dev Acc={dev_m.get('accuracy'):.4f}, Dev Prec={dev_m.get('precision'):.4f}, Dev Rec={dev_m.get('recall'):.4f}")
print("ITW length breakdown:", itw.get("clip_length_breakdown"))
print("ITW class breakdown:", itw.get("class_breakdown"))

print("\n=== AUDITS AND ABLATIONS ===")
aud = get_json("results/metrics/crnn_silence_audit.json")
print("Audits silence duration:", aud.get("silence_duration_classifier"))
print("Audits VAD:", aud.get("vad_trimming"))
print("Audits silence swap:", aud.get("silence_swap_test"))
print("Audits correlation:", aud.get("metadata_correlation_audit"))

band = get_json("results/metrics/band_analysis_results.json")
print("Band analysis:", band.get("bands"))

logo = get_json("results/metrics/logo_crossval.json")
print("LOGO crossval:", logo)

atk = get_json("results/metrics/per_attack_evaluation.json")
print("Per-attack attacks:", list(atk.get("attacks", {}).keys()))
for a, val in atk.get("attacks", {}).items():
    print(f"Attack {a} ({val.get('type')}, {val.get('family')}): CRNN EER={val.get('crnn_eer')}, Ens EER={val.get('ensemble_eer')}")

print("\n=== DEPLOYMENT AND LATENCY ===")
lat = get_json("results/metrics/latency_benchmark.json")
print("Latency:", lat.get("benchmarks"))
load = get_json("results/metrics/api_load_test.json")
print("Load test:", load)
q = get_json("results/metrics/quantization_benchmark.json")
print("Quantization formats:", q.get("formats"))
