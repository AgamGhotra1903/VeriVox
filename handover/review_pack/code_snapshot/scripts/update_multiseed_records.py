import json
import numpy as np
from pathlib import Path
from acforensics.evaluation.provenance import compute_file_sha256, guard_results_and_provenance

def main():
    with open("results/metrics/crnn_lfcc_results_seed42.json") as f:
        r42 = json.load(f)

    with open("results/metrics/crnn_multiseed_results.json") as f:
        multi = json.load(f)

    sha42 = compute_file_sha256("checkpoints/crnn_lfcc_seed42.pt")
    sha43 = compute_file_sha256("checkpoints/crnn_lfcc_seed43.pt")
    sha44 = compute_file_sha256("checkpoints/crnn_lfcc_seed44.pt")

    r42["checkpoint_path"] = "checkpoints/crnn_lfcc_seed42.pt"
    r42["checkpoint_sha256"] = sha42

    r43 = multi["individual_runs"][1]
    r43["checkpoint_path"] = "checkpoints/crnn_lfcc_seed43.pt"
    r43["checkpoint_sha256"] = sha43

    r44 = multi["individual_runs"][2]
    r44["checkpoint_path"] = "checkpoints/crnn_lfcc_seed44.pt"
    r44["checkpoint_sha256"] = sha44

    runs = [r42, r43, r44]
    dev_eers = [r["dev"]["eer"] for r in runs]
    eval_eers = [r["eval"]["eer"] for r in runs]
    eval_tdcfs = [r["eval"]["min_tdcf"] for r in runs]
    eval_aucs = [r["eval"]["roc_auc"] for r in runs]

    dev_mean = float(np.mean(dev_eers))
    dev_std = float(np.std(dev_eers))
    eval_mean = float(np.mean(eval_eers))
    eval_std = float(np.std(eval_eers))
    tdcf_mean = float(np.mean(eval_tdcfs))
    tdcf_std = float(np.std(eval_tdcfs))
    auc_mean = float(np.mean(eval_aucs))
    auc_std = float(np.std(eval_aucs))

    summary = {
        "dataset_name": "ASVspoof 2019 LA",
        "manifest_sha256": r42["manifest_sha256"],
        "git_commit": r42["git_commit"],
        "seed": 42,
        "timestamp": r42["timestamp"],
        "model": "CRNN-LFCC",
        "features": "lfcc",
        "seeds": [42, 43, 44],
        "metrics": {
            "dev_eer_mean": dev_mean,
            "dev_eer_std": dev_std,
            "eval_eer_mean": eval_mean,
            "eval_eer_std": eval_std,
            "eval_min_tdcf_mean": tdcf_mean,
            "eval_min_tdcf_std": tdcf_std,
            "eval_auc_mean": auc_mean,
            "eval_auc_std": auc_std,
        },
        "individual_runs": runs,
    }

    guard_results_and_provenance(summary)
    guard_results_and_provenance(r42)
    guard_results_and_provenance(r43)
    guard_results_and_provenance(r44)

    with open("results/metrics/crnn_multiseed_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("results/metrics/crnn_lfcc_multiseed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("results/metrics/crnn_lfcc_results_seed42.json", "w") as f:
        json.dump(r42, f, indent=2)

    with open("results/metrics/crnn_lfcc_results_seed43.json", "w") as f:
        json.dump(r43, f, indent=2)

    with open("results/metrics/crnn_lfcc_results_seed44.json", "w") as f:
        json.dump(r44, f, indent=2)

    print("Updated multiseed records successfully!")
    print(f"Dev EERs: {[round(x, 4) for x in dev_eers]} -> Mean: {dev_mean:.2f} +- {dev_std:.2f}%")
    print(f"Eval EERs: {[round(x, 4) for x in eval_eers]} -> Mean: {eval_mean:.2f} +- {eval_std:.2f}%")
    print(f"Seed 42 Checkpoint: {r42['checkpoint_path']} (SHA: {sha42})")
    print(f"Seed 43 Checkpoint: {r43['checkpoint_path']} (SHA: {sha43})")
    print(f"Seed 44 Checkpoint: {r44['checkpoint_path']} (SHA: {sha44})")

if __name__ == "__main__":
    main()

