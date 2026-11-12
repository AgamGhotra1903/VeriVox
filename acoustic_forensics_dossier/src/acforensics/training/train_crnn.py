"""CRNN Training Engine with AMP, Dev-EER Early Stopping, and Platt Calibration.

Features:
- Adam / AdamW optimizers with L2 weight decay.
- Learning rate schedulers: ReduceLROnPlateau on dev EER (lower is better) or Cosine Annealing.
- Gradient norm clipping at 5.0.
- Early stopping monitoring validation EER with patience=5 and best checkpoint restoration.
- Class-weighted Cross-Entropy loss addressing ASVspoof class imbalance (~8:1 spoof:bonafide).
- Strict data split leakage protection: asserts eval split is never used for training or early stopping.
- Reproducibility tracking with deterministic seeds and per-run JSON logging.

"""

from pathlib import Path
from typing import Any, Optional
import csv
import json
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
from acforensics.utils.seeds import set_seed

logger = get_logger(__name__)


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Calculates inverse class frequency weights to handle severe ASVspoof imbalance.

    In ASVspoof 2019 Logical Access, spoof utterances vastly outnumber genuine utterances (~9:1).
    Inverse frequency weight formula:
        w_c = N_total / (N_classes * N_c)
    This scales gradients evenly across genuine and spoof trials so minority genuine speech
    is not suppressed.
    """
    counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(counts) * counts.astype(float) + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_on_dev(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluates CRNN model on dev split and calculates dev loss, dev EER, threshold, and ROC-AUC.

    Continuous score convention:
        Score = logit[BONAFIDE] - logit[SPOOF] (Log-Likelihood Ratio)
        Higher score corresponds directly to higher likelihood of bona fide (genuine) audio.

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

            if features.ndim == 3:
                features = features.unsqueeze(1)
            assert features.ndim == 4, f"Dev evaluation expected 4D feature batch, got {features.shape}"
            logits, _ = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(all_labels))
    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int64)

    eer, threshold = compute_eer(labels_arr, scores_arr)
    return avg_loss, eer, threshold, scores_arr, labels_arr


def plot_training_history(history: list[dict[str, Any]], output_path: Path | str) -> None:
    """Plots and saves loss curves and validation EER over training epochs."""
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    dev_losses = [h["dev_loss"] for h in history]
    dev_eers = [h["dev_eer"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

    ax1.plot(epochs, train_losses, label="Train Loss", color="#3B82F6", lw=2)
    ax1.plot(epochs, dev_losses, label="Dev Loss", color="#EF4444", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    ax2.plot(epochs, dev_eers, label="Dev EER (%)", color="#10B981", lw=2)
    best_idx = np.argmin(dev_eers)
    ax2.scatter(
        epochs[best_idx],
        dev_eers[best_idx],
        color="#F59E0B",
        s=80,
        zorder=5,
        label=f"Best Checkpoint ({dev_eers[best_idx]:.2f}%)",
    )
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Equal Error Rate (%)", fontsize=11)
    ax2.set_title("Validation EER (Early Stopping Metric)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True)

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
    optimizer_type: str = "adamw",
    scheduler_type: str = "plateau",
    seed: int = 42,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 5,
    grad_clip_norm: float = 5.0,
    use_amp: bool = True,
    use_spec_augment: bool = True,
    loss_type: str = "weighted_cross_entropy",
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINTS_DIR,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    model_config: Optional[dict[str, Any]] = None,
    feature_config: Optional[dict[str, Any]] = None,
    norm_stats: Optional[dict[str, Any]] = None,
    checkpoint_name: str = "crnn_best.pt",
) -> tuple[CRNNForensics, PlattCalibrator, dict[str, Any]]:
    """Runs end-to-end CRNN training with model selection strictly on Dev EER.

    Leakage Protection:
        Asserts that train_loader and dev_loader are separate and eval split is never used
        for training or early stopping.

    Returns:
        Tuple of (best_model, fitted_calibrator, training_summary_dict).
    """
    train_split_name = getattr(getattr(train_loader, "dataset", None), "split", "train")
    dev_split_name = getattr(getattr(dev_loader, "dataset", None), "split", "dev")
    assert str(train_split_name).lower() == "train", f"Train loader split must be 'train', got {train_split_name}"
    assert str(dev_split_name).lower() == "dev", f"Dev loader split must be 'dev', got {dev_split_name}"

    set_seed(seed=seed, deterministic=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_labels = []
    for b in train_loader:
        train_labels.extend(b["label"].numpy())
    class_weights = compute_class_weights(np.array(train_labels)).to(device)
    logger.info("Computed class weights for loss: %s (spoof: %.2f, bona fide: %.2f)",
                class_weights.cpu().numpy(), class_weights[SPOOF].item(), class_weights[BONAFIDE].item())

    if loss_type == "weighted_cross_entropy":
        criterion = ClassWeightedCrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal":
        criterion = FocalLoss(gamma=2.0, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )

    spec_augment = SpecAugment().to(device) if use_spec_augment else None

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / checkpoint_name

    history = []
    best_dev_eer = float("inf")
    best_dev_thresh = 0.0
    best_epoch = 0
    best_dev_scores = None
    best_dev_labels = None
    patience_counter = 0

    logger.info(
        "Commencing CRNN training: %d epochs | lr: %e | weight_decay: %e | opt: %s | sched: %s | device: %s",
        epochs, lr, weight_decay, optimizer_type, scheduler_type, device,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for batch in pbar:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)

            if features.ndim == 3:
                features = features.unsqueeze(1)
            assert features.ndim == 4, f"Training expected 4D feature batch, got {features.shape}"

            if spec_augment is not None:
                features = spec_augment(features)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits, _ = model(features)
                assert logits.shape == (len(labels), 2), f"Logits shape mismatch: {logits.shape}"
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(labels)
            n_samples += len(labels)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / max(1, n_samples)

        dev_loss, dev_eer, dev_thresh, dev_scores, dev_labels = evaluate_on_dev(model, dev_loader, device)
        dev_auc = compute_roc_auc(dev_labels, dev_scores)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(dev_eer)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %02d | Train Loss: %.4f | Dev Loss: %.4f | Dev EER: %.2f%% | Dev Thresh: %.4f | LR: %e",
            epoch, train_loss, dev_loss, dev_eer, dev_thresh, current_lr,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_eer": dev_eer,
            "dev_thresh": dev_thresh,
            "dev_auc": dev_auc,
            "lr": current_lr,
        })

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            best_dev_thresh = dev_thresh
            best_epoch = epoch
            best_dev_scores = dev_scores
            best_dev_labels = dev_labels
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_eer": dev_eer,
                "dev_threshold": dev_thresh,
                "seed": seed,
                "model_config": model_config,
                "feature_config": feature_config,
                "training_hyperparameters": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "optimizer": optimizer_type,
                    "scheduler": scheduler_type,
                    "grad_clip_norm": grad_clip_norm,
                    "batch_size": train_loader.batch_size,
                    "early_stopping_patience": early_stopping_patience,
                },
                "normalization_stats": norm_stats,
            }, str(best_ckpt_path))
            logger.info("Saved new best model checkpoint (Dev EER: %.2f%%) to %s", dev_eer, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered after %d epochs without Dev EER improvement.", early_stopping_patience)
                break

    logger.info("Restoring best checkpoint from epoch %d (Dev EER: %.2f%%)", best_epoch, best_dev_eer)
    best_checkpoint = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    # 6. Fit Platt Calibrator exclusively on Dev scores
    calibrator = PlattCalibrator()
    if best_dev_scores is not None and best_dev_labels is not None:
        calibrator.fit(best_dev_scores, best_dev_labels)
        calibrator_path = ckpt_dir / f"calibrator_{Path(checkpoint_name).stem}.joblib"
        calibrator.save(calibrator_path)
        best_checkpoint["calibrator_state"] = calibrator
        torch.save(best_checkpoint, str(best_ckpt_path))

    res_dir = Path(results_dir)
    feature_name = (feature_config or {}).get("feature_type", "lfcc")
    run_log_path = res_dir / "metrics" / f"crnn_training_run_{feature_name}_seed{seed}.json"
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "model": "CRNNForensics",
        "features": feature_name,
        "seed": seed,
        "lr": lr,
        "batch_size": train_loader.batch_size,
        "epochs_trained": len(history),
        "weight_decay": weight_decay,
        "optimizer": optimizer_type,
        "scheduler": scheduler_type,
        "grad_clip_norm": grad_clip_norm,
        "early_stopping_patience": early_stopping_patience,
        "best_epoch": best_epoch,
        "best_dev_eer": best_dev_eer,
        "best_dev_threshold": best_dev_thresh,
        "history": history,
    }

    with open(run_log_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)
    logger.info("Saved run metadata to %s", run_log_path)

    plot_training_history(history, res_dir / "figures" / f"crnn_{feature_name}_training_curves.png")

    summary = {
        "best_epoch": best_epoch,
        "best_dev_eer": best_dev_eer,
        "best_dev_threshold": best_dev_thresh,
        "checkpoint_path": str(best_ckpt_path),
        "history": history,
    }
    return model, calibrator, summary
