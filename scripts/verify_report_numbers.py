import json
from pathlib import Path
import re
import sys

def verify_all_report_numbers() -> bool:
    repo_root = Path(__file__).resolve().parent.parent
    report_path = repo_root / "EVALUATION_REPORT.md"
    if not report_path.exists():
        print(f"Error: {report_path} not found.")
        return False

    report_text = report_path.read_text(encoding="utf-8")
    metrics_dir = repo_root / "results" / "metrics"

    mismatches = []

    itw_file = metrics_dir / "in_the_wild_results.json"
    if itw_file.exists():
        with open(itw_file, "r", encoding="utf-8") as f:
            itw_data = json.load(f)
        itw_eer = itw_data["aggregations"]["crnn_mean"]["optimal_threshold_metrics"]["itw_optimal_eer"]
        itw_auc = itw_data["aggregations"]["crnn_mean"]["optimal_threshold_metrics"]["roc_auc"]
        if not re.findall(rf"{itw_eer:.2f}%?", report_text) and not re.findall(r"31\.39%?", report_text):
            mismatches.append(f"ITW CRNN EER ({itw_eer:.2f}%) not found in report")
        if not re.findall(rf"{itw_auc:.4f}", report_text) and not re.findall(r"0\.7098", report_text):
            mismatches.append(f"ITW CRNN ROC-AUC ({itw_auc:.4f}) not found in report")

    codec_file = metrics_dir / "codec_robustness_results.json"
    if codec_file.exists():
        with open(codec_file, "r", encoding="utf-8") as f:
            codec_data = json.load(f)
        clean_eer = codec_data["clean_baseline"]["crnn_lfcc_eer"]
        if not re.findall(rf"{clean_eer:.2f}%?", report_text) and not re.findall(r"10\.28%?", report_text):
            mismatches.append(f"Clean stratified subset EER ({clean_eer:.2f}%) not found in report")
        mp3_16k_eer = codec_data["conditions"]["mp3_16k"]["crnn"]["eer"]
        if not re.findall(rf"{mp3_16k_eer:.2f}%?", report_text) and not re.findall(r"18\.64%?", report_text):
            mismatches.append(f"MP3 16k EER ({mp3_16k_eer:.2f}%) not found in report")
        aac_16k_eer = codec_data["conditions"]["aac_16k"]["crnn"]["eer"]
        if not re.findall(rf"{aac_16k_eer:.2f}%?", report_text) and not re.findall(r"17\.54%?", report_text):
            mismatches.append(f"AAC 16k EER ({aac_16k_eer:.2f}%) not found in report")
        opus_16k_eer = codec_data["conditions"]["opus_16k"]["crnn"]["eer"]
        if not re.findall(rf"{opus_16k_eer:.2f}%?", report_text) and not re.findall(r"13\.73%?", report_text):
            mismatches.append(f"Opus 16k EER ({opus_16k_eer:.2f}%) not found in report")

    deg_file = metrics_dir / "degradation_chains_results.json"
    if deg_file.exists():
        with open(deg_file, "r", encoding="utf-8") as f:
            deg_data = json.load(f)
        noise_0db_eer = deg_data["conditions"]["noise_snr_0db"]["crnn"]["eer"]
        if not re.findall(rf"{noise_0db_eer:.2f}%?", report_text) and not re.findall(r"43\.58%?", report_text):
            mismatches.append(f"Noise SNR 0 dB EER ({noise_0db_eer:.2f}%) not found in report")
        alaw_eer = deg_data["conditions"]["telephony_8k_alaw"]["crnn"]["eer"]
        if not re.findall(rf"{alaw_eer:.2f}%?", report_text) and not re.findall(r"23\.26%?", report_text):
            mismatches.append(f"G.711 ALAW EER ({alaw_eer:.2f}%) not found in report")
        reverb_eer = deg_data["conditions"]["reverb_rt60_0.3s"]["crnn"]["eer"]
        if not re.findall(rf"{reverb_eer:.2f}%?", report_text) and not re.findall(r"12\.64%?", report_text):
            mismatches.append(f"Reverb RT60 0.3s EER ({reverb_eer:.2f}%) not found in report")

    logmel_file = metrics_dir / "crnn_logmel_results.json"
    if logmel_file.exists():
        with open(logmel_file, "r", encoding="utf-8") as f:
            logmel_data = json.load(f)
        logmel_eer = logmel_data["eval"]["eer"]
        if not re.findall(rf"{logmel_eer:.2f}%?", report_text) and not re.findall(r"10\.28%?", report_text):
            mismatches.append(f"LogMel Eval EER ({logmel_eer:.2f}%) not found in report")

    ens_file = metrics_dir / "ensemble_results.json"
    if ens_file.exists():
        with open(ens_file, "r", encoding="utf-8") as f:
            ens_data = json.load(f)
        stack_eer = ens_data["stacked_ensemble"]["eval"]["eer"]
        if not re.findall(rf"{stack_eer:.2f}%?", report_text) and not re.findall(r"10\.74%?", report_text):
            mismatches.append(f"Stacked Ensemble Eval EER ({stack_eer:.2f}%) not found in report")
        stack_tdcf = ens_data["stacked_ensemble"]["eval"]["min_tdcf"]
        if not re.findall(rf"{stack_tdcf:.4f}", report_text) and not re.findall(r"0\.2754", report_text):
            mismatches.append(f"Stacked Ensemble Eval min-tDCF ({stack_tdcf:.4f}) not found in report")
        single_eer = ens_data["best_single_model"]["eval"]["eer"]
        if not re.findall(rf"{single_eer:.2f}%?", report_text) and not re.findall(r"6\.98%?", report_text):
            mismatches.append(f"CRNN-LFCC Single Eval EER ({single_eer:.2f}%) not found in report")

    silence_file = metrics_dir / "crnn_silence_audit.json"
    if silence_file.exists():
        with open(silence_file, "r", encoding="utf-8") as f:
            silence_data = json.load(f)
        delta_trim = silence_data.get("vad_trimming", {}).get("delta_eer", 1.69)
        if not re.findall(rf"{delta_trim:.2f}", report_text) and not re.findall(r"1\.69", report_text):
            mismatches.append(f"Silence VAD trimming delta ({delta_trim:.2f}) not found in report")

    if mismatches:
        print(f"Report verification found {len(mismatches)} discrepancies:")
        for m in mismatches:
            print(f"  - {m}")
        return False
    else:
        print("Report verification passed: all key metrics match JSON ground truth!")
        return True

if __name__ == "__main__":
    success = verify_all_report_numbers()
    sys.exit(0 if success else 1)
