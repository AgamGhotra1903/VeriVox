"""CRNN Training Engine with AMP, SpecAugment, Dev-EER Early Stopping, and Platt Calibration.

Author: Agam Ghotra (IIT2024226)
"""

from pathlib import Path
from typing import Any, Optional
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_RESULTS_DIR,
)
from acforensics.models.crnn import CRNNForensics
from acforensics.models.calibration import PlattCalibrator
from acforensics.training.losses import ClassWeightedCrossEntropyLoss, FocalLoss
from acforensics.training.augment import SpecAugment
from acforensics.evaluation.metrics import compute_eer, compute_roc_auc
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Calculates inverse frequency weights for cross-entropy to handle ASVspoof imbalance."""
    counts = np.bincount(labels)
    total = len(labels)
    # Weight = total / (num_classes * count)
    weights = total / (len(counts) * counts.astype(float) + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_on_dev(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluates CRNN model on dev split and calculates dev EER, threshold, and ROC-AUC.

    Continuous score convention:
        Score = logit[BONAFIDE] - logit[SPOOF]  (Log-Likelihood Ratio)
        Higher score corresponds directly to higher likelihood of genuine audio.

    Returns:
        Tuple of (dev_loss, dev_eer, dev_threshold, all_scores, all_labels).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            # Log-Likelihood Ratio score for bona fide
            scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int64)

    eer, threshold = compute_eer(labels_arr, scores_arr)
    auc = compute_roc_auc(labels_arr, scores_arr)

    return avg_loss, eer, threshold, scores_arr, labels_arr


def plot_training_history(history: list[dict[str, Any]], output_path: Path | str) -> None:
    """Plots and saves loss, dev EER, and dev AUC curves across training epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    dev_loss = [h["dev_loss"] for h in history]
    dev_eer = [h["dev_eer"] for h in history]
    dev_auc = [h["dev_auc"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=200)

    # 1. Loss Curves
    axes[0].plot(epochs, train_loss, label="Train Loss", color="#EF4444", lw=2)
    axes[0].plot(epochs, dev_loss, label="Dev Loss", color="#2DD4BF", lw=2, linestyle="--")
    axes[0].set_title("Cross-Entropy Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Dev EER Curve
    axes[1].plot(epochs, dev_eer, label="Dev EER (%)", color="#F59E0B", lw=2)
    best_epoch_idx = np.argmin(dev_eer)
    axes[1].axvline(epochs[best_epoch_idx], color="grey", linestyle=":", label=f"Best ({dev_eer[best_epoch_idx]:.2f}%)")
    axes[1].set_title("Validation Equal Error Rate (EER)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("EER (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Dev ROC-AUC Curve
    axes[2].plot(epochs, dev_auc, label="Dev ROC-AUC", color="#22C55E", lw=2)
    axes[2].set_title("Validation ROC-AUC", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out))
    plt.savefig(str(out.with_suffix(".svg")))
    plt.close()


def train_crnn(
    model: CRNNForensics,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 8,
    use_amp: bool = True,
    use_spec_augment: bool = True,
    loss_type: str = "weighted_cross_entropy",
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINTS_DIR,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    model_config: Optional[dict[str, Any]] = None,
    feature_config: Optional[dict[str, Any]] = None,
    norm_stats: Optional[dict[str, Any]] = None,
) -> tuple[CRNNForensics, PlattCalibrator, dict[str, Any]]:
    """Runs end-to-end CRNN training with model selection on Dev EER.

    Returns:
        Tuple of (best_model, fitted_calibrator, training_summary_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set up Loss
    train_labels = []
    for b in train_loader:
        train_labels.extend(b["label"].numpy())
    class_weights = compute_class_weights(np.array(train_labels)).to(device)

    if loss_type == "weighted_cross_entropy":
        criterion = ClassWeightedCrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal":
        criterion = FocalLoss(gamma=2.0, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    spec_augment = SpecAugment().to(device) if use_spec_augment else None

    # Mixed precision scaler (GPU only)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "crnn_best.pt"

    history = []
    best_dev_eer = float("inf")
    best_dev_thresh = 0.0
    best_epoch = 0
    best_dev_scores = None
    best_dev_labels = None
    patience_counter = 0

    logger.info("Commencing CRNN training for %d epochs on device: %s", epochs, device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for batch in pbar:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)

            if spec_augment is not None:
                features = spec_augment(features)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits, _ = model(features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(labels)
            n_samples += len(labels)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss = running_loss / n_samples

        # Evaluate on Dev split
        dev_loss, dev_eer, dev_thresh, dev_scores, dev_labels = evaluate_on_dev(model, dev_loader, device)

        logger.info(
            "Epoch %02d | Train Loss: %.4f | Dev Loss: %.4f | Dev EER: %.2f%% | Dev Thresh: %.4f",
            epoch,
            train_loss,
            dev_loss,
            dev_eer,
            dev_thresh,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_eer": dev_eer,
            "dev_thresh": dev_thresh,
            "dev_auc": compute_roc_auc(dev_labels, dev_scores),
        })

        # Model selection: Check if dev EER improved
        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            best_dev_thresh = dev_thresh
            best_epoch = epoch
            best_dev_scores = dev_scores
            best_dev_labels = dev_labels
            patience_counter = 0

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_eer": dev_eer,
                "dev_threshold": dev_thresh,
                "model_config": model_config,
                "feature_config": feature_config,
                "normalization_stats": norm_stats,
            }, str(best_ckpt_path))
            logger.info("Saved new best model checkpoint (Dev EER: %.2f%%) to %s", dev_eer, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", early_stopping_patience)
                break

    # Load best weights
    best_checkpoint = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    # Fit Platt Calibrator on Dev scores
    calibrator = PlattCalibrator()
    calibrator.fit(best_dev_scores, best_dev_labels)
    calibrator.save(ckpt_dir / "calibrator.joblib")

    # Update checkpoint with calibrator
    best_checkpoint["calibrator_state"] = calibrator
    torch.save(best_checkpoint, str(best_ckpt_path))

    # Save training logs & curves
    res_dir = Path(results_dir)
    log_csv = res_dir / "logs" / "crnn_training_log.csv"
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    plot_training_history(history, res_dir / "figures" / "crnn_training_curves.png")

    summary = {
        "best_epoch": best_epoch,
        "best_dev_eer": best_dev_eer,
        "best_dev_threshold": best_dev_thresh,
        "history": history,
    }
    return model, calibrator, summary
