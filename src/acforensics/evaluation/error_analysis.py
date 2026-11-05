"""Forensic error audit: False Positives & False Negatives broken down by attack, speaker, and quality.

Author: Chunit Bansal (IIT2024223)
"""

from pathlib import Path
from typing import Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    UNSEEN_ATTACKS,
    DEFAULT_RESULTS_DIR,
)
from acforensics.evaluation.metrics import compute_eer
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def plot_per_attack_eer(attack_eers: dict[str, float], output_path: Path | str) -> None:
    """Plots bar chart of EER across individual unseen eval attacks (A07 to A19)."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    attacks = sorted(attack_eers.keys())
    eers = [attack_eers[a] for a in attacks]

    bars = ax.bar(attacks, eers, color="#3B82F6", edgecolor="black", alpha=0.85)

    for i, a in enumerate(attacks):
        if a in ("A16", "A19"):
            bars[i].set_color("#F59E0B")

    ax.axhline(np.mean(eers), color="#EF4444", linestyle="--", lw=1.5, label=f"Mean Attack EER ({np.mean(eers):.2f}%)")
    ax.set_xlabel("Attack Algorithm (A07 - A19 on ASVspoof 2019 LA Eval)", fontsize=11)
    ax.set_ylabel("Equal Error Rate (EER %)", fontsize=11)
    ax.set_title("Per-Attack Vulnerability Breakdown (Amber = Known Attack Variants)", fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out))
    plt.savefig(str(out.with_suffix(".svg")))
    plt.close()


def run_error_audit(
    eval_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    output_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> dict[str, Any]:
    """Performs deep error breakdown and identifies worst-case false alarms and missed detections.

    Args:
        eval_df: DataFrame of evaluation metadata.
        scores: Predicted continuous scores (higher = bona fide).
        threshold: Operating decision threshold (derived from Dev EER).
        output_dir: Directory where tables and figures will be saved.

    Returns:
        Structured audit dictionary with findings and tables.
    """
    out_dir = Path(output_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = eval_df.copy().reset_index(drop=True)
    df["score"] = scores
    df["predicted_label"] = (scores >= threshold).astype(int)

    # Classifications
    # False Positive: Spoof classified as Bona Fide (score >= threshold)
    # False Negative: Bona Fide classified as Spoof (score < threshold)
    df["is_fp"] = (df["label"] == SPOOF) & (df["predicted_label"] == BONAFIDE)
    df["is_fn"] = (df["label"] == BONAFIDE) & (df["predicted_label"] == SPOOF)

    # 1. Per-Attack EER Calculation
    bf_scores = df[df["label"] == BONAFIDE]["score"].values
    attack_eers = {}
    attack_counts = {}

    for attack_id in df[df["label"] == SPOOF]["attack_id"].unique():
        sp_scores = df[df["attack_id"] == attack_id]["score"].values
        if len(sp_scores) > 0 and len(bf_scores) > 0:
            labels_comb = np.concatenate([np.ones_like(bf_scores), np.zeros_like(sp_scores)])
            scores_comb = np.concatenate([bf_scores, sp_scores])
            eer_val, _ = compute_eer(labels_comb, scores_comb)
            attack_eers[attack_id] = float(eer_val)
            attack_counts[attack_id] = len(sp_scores)

    # Plot per-attack breakdown
    plot_per_attack_eer(attack_eers, figures_dir / "per_attack_eer.png")

    if "file_id" not in df.columns and "path" in df.columns:
        df["file_id"] = [Path(p).stem for p in df["path"]]
    if "speaker_id" not in df.columns and "speaker" in df.columns:
        df["speaker_id"] = df["speaker"]

    top_fps = df[df["is_fp"]].sort_values("score", ascending=False).head(10)
    top_fns = df[df["is_fn"]].sort_values("score", ascending=True).head(10)

    worst_cases = pd.concat([
        top_fps.assign(error_type="False Positive (Synthetic Called Genuine)"),
        top_fns.assign(error_type="False Negative (Genuine Called Synthetic)"),
    ])
    worst_cases_cols = ["file_id", "speaker_id", "attack_id", "label", "score", "error_type"]
    available_cols = [c for c in worst_cases_cols if c in worst_cases.columns]
    worst_cases[available_cols].to_csv(tables_dir / "worst_case_errors.csv", index=False)

    n_fp = int(df["is_fp"].sum())
    n_fn = int(df["is_fn"].sum())
    most_vulnerable_attack = max(attack_eers.items(), key=lambda x: x[1])[0] if attack_eers else "N/A"
    most_vulnerable_eer = attack_eers.get(most_vulnerable_attack, 0.0)
    least_vulnerable_attack = min(attack_eers.items(), key=lambda x: x[1])[0] if attack_eers else "N/A"
    least_vulnerable_eer = attack_eers.get(least_vulnerable_attack, 0.0)

    known_keys = [k for k in ("A16", "A19") if k in attack_eers]
    unseen_keys = [k for k in attack_eers if k not in ("A16", "A19")]
    mean_known_eer = float(np.mean([attack_eers[k] for k in known_keys])) if known_keys else 0.0
    mean_unseen_eer = float(np.mean([attack_eers[k] for k in unseen_keys])) if unseen_keys else 0.0

    findings = [
        f"Audited {len(df)} evaluation trials at dev-calibrated threshold {threshold:.4f}.",
        f"Detected {n_fp} False Positives ({n_fp / max(1, int(np.sum(df['label'] == SPOOF))) * 100:.2f}% spoof miss rate) and {n_fn} False Negatives ({n_fn / max(1, int(np.sum(df['label'] == BONAFIDE))) * 100:.2f}% genuine false alarm rate).",
        f"Highest vulnerability observed on attack {most_vulnerable_attack} ({most_vulnerable_eer:.2f}% EER); lowest on {least_vulnerable_attack} ({least_vulnerable_eer:.2f}% EER).",
        f"Known attack variants (A16, A19) achieved mean EER {mean_known_eer:.2f}%, compared to {mean_unseen_eer:.2f}% across unseen generation algorithms (delta: {mean_unseen_eer - mean_known_eer:+.2f} pts).",
        f"Worst-case false alarms and missed detections logged to worst_case_errors.csv for inspection.",
    ]

    summary = {
        "applied_threshold": threshold,
        "total_trials": len(df),
        "false_positive_count": n_fp,
        "false_negative_count": n_fn,
        "per_attack_eer": attack_eers,
        "most_vulnerable_attack": most_vulnerable_attack,
        "most_vulnerable_eer": most_vulnerable_eer,
        "least_vulnerable_attack": least_vulnerable_attack,
        "least_vulnerable_eer": least_vulnerable_eer,
        "known_variants_mean_eer": mean_known_eer,
        "unseen_attacks_mean_eer": mean_unseen_eer,
        "plain_language_findings": findings,
    }
    return summary
