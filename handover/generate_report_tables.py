import json
from pathlib import Path

def load_json(p):
    return json.loads(Path(p).read_text())

ens = load_json("results/metrics/ensemble_results.json")
spk = load_json("results/metrics/speaker_bootstrap.json")
ms = load_json("results/metrics/crnn_multiseed_results.json")
crnn_lfcc = load_json("results/metrics/crnn_lfcc_results.json")
crnn_logmel = load_json("results/metrics/crnn_logmel_results.json")
itw = load_json("results/metrics/in_the_wild_results.json")
itw_old = load_json("handover/evidence/metrics/in_the_wild_results.json")
codec = load_json("results/metrics/codec_robustness_results.json")
codec_old = load_json("handover/evidence/metrics/codec_robustness_results.json")
deg = load_json("results/metrics/degradation_chains_results.json")
deg_old = load_json("handover/evidence/metrics/degradation_chains_results.json")
audits = load_json("results/metrics/crnn_silence_audit.json")
audits_old = load_json("handover/evidence/metrics/audits.json")
base = load_json("results/metrics/baselines_results.json")
logo = load_json("results/metrics/logo_crossval.json")
atk = load_json("results/metrics/per_attack_evaluation.json")
lat = load_json("results/metrics/latency_benchmark.json")
q = load_json("results/metrics/quantization_benchmark.json")
api = load_json("results/metrics/api_load_test.json")

print("DUMP COMPLETE")

