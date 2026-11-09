import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    metrics_path = Path("results/metrics/crnn_multiseed_results.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("individual_runs", [])
    if not runs:
        raise ValueError("No individual_runs found in crnn_multiseed_results.json")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)

    palette = {
        42: "#1f77b4",
        43: "#2ca02c",
        44: "#ff7f0e",
    }

    all_train_losses = []
    all_dev_losses = []
    all_dev_eers = []

    for run in runs:
        seed = run["seed"]
        history = run.get("history", [])
        if not history:
            continue

        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        dev_loss = [h["dev_loss"] for h in history]
        dev_eer = [h["dev_eer"] for h in history]
        lr = [h["lr"] for h in history]

        all_train_losses.append(train_loss)
        all_dev_losses.append(dev_loss)
        all_dev_eers.append(dev_eer)

        color = palette.get(seed, "#333333")
        axes[0, 0].plot(epochs, train_loss, color=color, alpha=0.75, linewidth=1.6, label=f"Seed {seed}")
        axes[0, 1].plot(epochs, dev_loss, color=color, alpha=0.75, linewidth=1.6, label=f"Seed {seed}")
        axes[1, 0].plot(epochs, dev_eer, color=color, alpha=0.75, linewidth=1.6, label=f"Seed {seed}")
        axes[1, 1].plot(epochs, lr, color=color, alpha=0.75, linewidth=1.6, label=f"Seed {seed}")

        best_dev_eer = min(dev_eer)
        best_epoch = epochs[dev_eer.index(best_dev_eer)]
        axes[1, 0].scatter([best_epoch], [best_dev_eer], color=color, s=40, zorder=5)

    if all_train_losses:
        max_len = max(len(x) for x in all_train_losses)
        epochs_arr = list(range(1, max_len + 1))
        mean_train = [float(np.mean([x[i] for x in all_train_losses if i < len(x)])) for i in range(max_len)]
        axes[0, 0].plot(epochs_arr, mean_train, color="black", linestyle="--", linewidth=2.0, label="3-Seed Mean")

    if all_dev_losses:
        max_len = max(len(x) for x in all_dev_losses)
        epochs_arr = list(range(1, max_len + 1))
        mean_dev_loss = [float(np.mean([x[i] for x in all_dev_losses if i < len(x)])) for i in range(max_len)]
        axes[0, 1].plot(epochs_arr, mean_dev_loss, color="black", linestyle="--", linewidth=2.0, label="3-Seed Mean")

    if all_dev_eers:
        max_len = max(len(x) for x in all_dev_eers)
        epochs_arr = list(range(1, max_len + 1))
        mean_dev_eer = [float(np.mean([x[i] for x in all_dev_eers if i < len(x)])) for i in range(max_len)]
        axes[1, 0].plot(epochs_arr, mean_dev_eer, color="black", linestyle="--", linewidth=2.0, label="3-Seed Mean")

    axes[0, 0].set_title("Training Loss vs Epoch", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontsize=10)
    axes[0, 0].set_ylabel("Cross-Entropy Loss", fontsize=10)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="upper right", framealpha=0.9)

    axes[0, 1].set_title("Validation (Dev) Loss vs Epoch", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch", fontsize=10)
    axes[0, 1].set_ylabel("Cross-Entropy Loss", fontsize=10)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(loc="upper right", framealpha=0.9)

    axes[1, 0].set_title("Validation (Dev) EER (%) vs Epoch", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch", fontsize=10)
    axes[1, 0].set_ylabel("Equal Error Rate (%)", fontsize=10)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="upper right", framealpha=0.9)

    axes[1, 1].set_title("Learning Rate Schedule (Cosine Annealing)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch", fontsize=10)
    axes[1, 1].set_ylabel("Learning Rate", fontsize=10)
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="upper right", framealpha=0.9)

    plt.suptitle("CRNN-LFCC Multiseed Optimization & Learning Dynamics (Seeds 42, 43, 44)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_fig = Path("results/figures/crnn_multiseed_learning_dynamics.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved multiseed learning dynamics figure to {out_fig}")

if __name__ == "__main__":
    main()
