"""Forensic visualization suite: ROC curves, DET curves, score distributions, and confusion matrices.

Author: Chunit Bansal (IIT2024223)
"""

from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, det_curve

from acforensics.constants import BONAFIDE, SPOOF, DEFAULT_RESULTS_DIR

# Forensic dark lab aesthetic colors
PALETTE = {
    "genuine": "#22C55E",      # Vibrant green
    "synthetic": "#EF4444",    # Forensic red
    "accent": "#2DD4BF",       # Lab teal
    "neutral": "#94A3B8",      # Slate grey
    "bg_dark": "#0E1117",
    "surface": "#161B22",
}


def plot_roc_curve(
    models_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path | str,
    title: str = "Receiver Operating Characteristic (ROC)",
) -> None:
    """Plots and saves multi-model ROC curves with equal error diagonal."""
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    for name, (labels, scores) in models_dict.items():
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=BONAFIDE)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (Synthetic called Genuine)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Genuine correctly identified)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out))
    # Also save SVG vector format
    plt.savefig(str(out.with_suffix(".svg")))
    plt.close()


def plot_det_curve(
    models_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path | str,
    title: str = "Detection Error Tradeoff (DET)",
) -> None:
    """Plots DET curves (False Alarm Rate vs False Rejection Rate) on normal deviate scale."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    for name, (labels, scores) in models_dict.items():
        # DET: FPR = FA (spoof called genuine), FNR = Miss (genuine called spoof)
        fpr, fnr, _ = det_curve(labels, scores, pos_label=BONAFIDE)
        # Safely clip to avoid log(0) divergence when error rate is 0.0%
        fpr_pct = np.clip(fpr * 100.0, 0.01, 100.0)
        fnr_pct = np.clip(fnr * 100.0, 0.01, 100.0)
        ax.plot(fpr_pct, fnr_pct, lw=2, label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([0.01, 100.0])
    ax.set_ylim([0.01, 100.0])
    ax.set_xlabel("False Alarm Probability (%) [Synthetic called Genuine]", fontsize=11)
    ax.set_ylabel("Miss Probability (%) [Genuine called Synthetic]", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out))
    plt.savefig(str(out.with_suffix(".svg")))
    plt.close()


def plot_score_distributions(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_path: Path | str,
    model_name: str = "Model",
) -> None:
    """Plots score histograms and KDEs for bona fide vs spoof classes with operating threshold."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    bf_scores = scores[labels == BONAFIDE]
    sp_scores = scores[labels == SPOOF]

    sns.histplot(sp_scores, color=PALETTE["synthetic"], label="Spoof (Synthetic)", kde=True, stat="density", bins=40, alpha=0.5, ax=ax)
    sns.histplot(bf_scores, color=PALETTE["genuine"], label="Bona Fide (Genuine)", kde=True, stat="density", bins=40, alpha=0.5, ax=ax)

    ax.axvline(threshold, color="black", linestyle="--", lw=2, label=f"EER Threshold ({threshold:.3f})")
    ax.set_xlabel("Predicted Countermeasure Score (higher = more genuine)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Score Separation Distribution: {model_name}", fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out))
    plt.savefig(str(out.with_suffix(".svg")))
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path | str,
    title: str = "Confusion Matrix",
) -> None:
    """Plots normalized confusion matrix with raw counts."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    labels = ["Spoof (0)", "Bona Fide (1)"]
    annot = np.array([
        [f"{cm[0, 0]}\n({cm_norm[0, 0]:.1%})", f"{cm[0, 1]}\n({cm_norm[0, 1]:.1%})"],
        [f"{cm[1, 0]}\n({cm_norm[1, 0]:.1%})", f"{cm[1, 1]}\n({cm_norm[1, 1]:.1%})"],
    ])

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("Ground Truth Class", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out))
    plt.savefig(str(out.with_suffix(".svg")))
    plt.close()
