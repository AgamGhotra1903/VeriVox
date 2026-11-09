import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acforensics.evaluation.provenance import (
    compute_file_sha256,
    create_provenance_record,
    guard_results_and_provenance,
)


def export_crnn_training_histories(
    multiseed_json_path: str = "results/metrics/crnn_multiseed_results.json",
    manifest_path: str = "data/manifests/train.parquet",
    output_dir: str = "results/metrics",
    figures_dir: str = "results/figures",
) -> None:
    p_multi = Path(multiseed_json_path)
    with open(p_multi, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    fig_p = Path(figures_dir)
    fig_p.mkdir(parents=True, exist_ok=True)

    ckpt_map = {
        42: "checkpoints/crnn_lfcc_seed42.pt",
        43: "checkpoints/crnn_lfcc_seed43.pt",
        44: "checkpoints/crnn_lfcc_seed44.pt",
    }

    histories_by_seed = {}

    for run in data.get("individual_runs", []):
        s = int(run["seed"])
        ckpt_path = ckpt_map.get(s, f"checkpoints/crnn_lfcc_seed{s}.pt")
        prov = create_provenance_record(
            dataset_name="asvspoof2019_la",
            manifest_path=manifest_path,
            seed=s,
            checkpoint_path=ckpt_path if Path(ckpt_path).exists() else None,
        )

        hist = run["history"]
        histories_by_seed[s] = hist

        payload = {
            **prov,
            "model": "CRNNForensics",
            "features": "lfcc",
            "seed": s,
            "parameters": run.get("parameters", 1081922),
            "total_epochs": len(hist),
            "best_dev_eer": float(min(e["dev_eer"] for e in hist)),
            "final_dev_eer": float(hist[-1]["dev_eer"]),
            "eval_eer": float(run.get("eval", {}).get("eer", 0.0)),
            "eval_min_tdcf": float(run.get("eval", {}).get("min_tdcf", 0.0)),
            "training_time_sec": float(run.get("training_time_sec", 0.0)),
            "history": hist,
        }

        guard_results_and_provenance(payload)

        target_file = out_p / f"crnn_training_run_lfcc_seed{s}.json"
        with open(target_file, "w", encoding="utf-8") as f_out:
            json.dump(payload, f_out, indent=2)
        print(f"Exported training history to {target_file} (epochs: {len(hist)})")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax_loss, ax_eer) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    palette = {
        42: ("#2563EB", "#60A5FA"),
        43: ("#DC2626", "#F87171"),
        44: ("#059669", "#34D399"),
    }

    for s, hist in histories_by_seed.items():
        epochs = [e["epoch"] for e in hist]
        tr_loss = [e["train_loss"] for e in hist]
        dv_loss = [e["dev_loss"] for e in hist]
        dv_eer = [e["dev_eer"] for e in hist]

        c_train, c_dev = palette.get(s, ("#000000", "#888888"))
        ax_loss.plot(epochs, tr_loss, color=c_train, lw=1.8, label=f"Train Loss (Seed {s})")
        ax_loss.plot(epochs, dv_loss, color=c_dev, lw=1.8, linestyle="--", label=f"Dev Loss (Seed {s})")

        ax_eer.plot(epochs, dv_eer, color=c_train, lw=1.8, marker="o", markersize=4, label=f"Dev EER (Seed {s})")

    ax_loss.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_loss.set_ylabel("Binary Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax_loss.set_title("CRNN-LFCC Convergence: Loss vs. Epoch", fontsize=12, fontweight="bold")
    ax_loss.legend(frameon=True, fontsize=9)
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    ax_eer.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_eer.set_ylabel("Dev Equal Error Rate (EER %)", fontsize=11, fontweight="bold")
    ax_eer.set_title("CRNN-LFCC Validation: Dev EER vs. Epoch", fontsize=12, fontweight="bold")
    ax_eer.legend(frameon=True, fontsize=9)
    ax_eer.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    pdf_path = fig_p / "crnn_training_curves.pdf"
    png_path = fig_p / "crnn_training_curves.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated vector PDF figure: {pdf_path}")
    print(f"Generated PNG figure: {png_path}")


if __name__ == "__main__":
    export_crnn_training_histories()
