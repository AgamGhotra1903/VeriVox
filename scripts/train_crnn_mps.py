import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.models.crnn import CRNNForensics
from acforensics.models.calibration import PlattCalibrator
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.training.augment import SpecAugment
from acforensics.training.train_crnn import compute_class_weights
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_metrics_at_threshold,
    compute_min_tdcf,
    bootstrap_confidence_interval,
)
from acforensics.evaluation.provenance import (
    compute_file_sha256,
    get_git_commit,
    guard_results_and_provenance,
    guard_no_smoke_test,
)

def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    tot_loss = 0.0
    tot_cnt = 0
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for b_x, b_y in loader:
            b_x = b_x.to(device=device, dtype=torch.float32).unsqueeze(1)
            b_y = b_y.to(device)
            logits, _ = model(b_x)
            loss = criterion(logits, b_y)
            tot_loss += loss.item() * len(b_y)
            tot_cnt += len(b_y)
            scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(b_y.cpu().numpy())
    avg_loss = tot_loss / max(1, tot_cnt)
    scores_arr = np.concatenate(all_scores, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    return avg_loss, scores_arr, labels_arr

def plot_curves(history: list[dict], out_path: Path) -> None:
    eps = [h["epoch"] for h in history]
    tr_l = [h["train_loss"] for h in history]
    dv_l = [h["dev_loss"] for h in history]
    dv_e = [h["dev_eer"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)
    ax1.plot(eps, tr_l, label="Train Loss", color="#2563EB", lw=2)
    ax1.plot(eps, dv_l, label="Dev Loss", color="#DC2626", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    ax2.plot(eps, dv_e, label="Dev EER (%)", color="#059669", lw=2)
    b_idx = int(np.argmin(dv_e))
    ax2.scatter(eps[b_idx], dv_e[b_idx], color="#D97706", s=80, zorder=5, label=f"Best ({dv_e[b_idx]:.2f}%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("EER (%)")
    ax2.set_title("Dev EER Curve")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path))
    plt.close()

def run_training(
    feature_type: str = "lfcc",
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 4,
    device_name: str = "auto",
    use_spec_augment: bool = True,
    resume_path: str = None,
    dev_asv: str = "data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt",
    eval_asv: str = "data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
    n_bootstraps: int = 1000,
) -> dict:
    set_all_seeds(seed)
    if device_name == "auto":
        device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    tr_cache = Path(f"data/cache/{feature_type}_train.npz")
    dv_cache = Path(f"data/cache/{feature_type}_dev.npz")
    ev_cache = Path(f"data/cache/{feature_type}_eval.npz")
    guard_no_smoke_test(str(tr_cache))
    guard_no_smoke_test(str(dv_cache))
    guard_no_smoke_test(str(ev_cache))

    print(f"Loading {feature_type.upper()} cached datasets and pre-applying per-utterance CMVN...")
    d_tr = np.load(tr_cache)
    tr_x = per_utterance_cmvn(torch.from_numpy(d_tr["features"]))
    tr_y = torch.from_numpy(d_tr["labels"])

    d_dv = np.load(dv_cache)
    dv_x = per_utterance_cmvn(torch.from_numpy(d_dv["features"]))
    dv_y = torch.from_numpy(d_dv["labels"])

    d_ev = np.load(ev_cache)
    ev_x = per_utterance_cmvn(torch.from_numpy(d_ev["features"]))
    ev_y = torch.from_numpy(d_ev["labels"])

    in_dim = tr_x.shape[1]
    train_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(dv_x, dv_y), batch_size=64, shuffle=False)
    eval_loader = DataLoader(TensorDataset(ev_x, ev_y), batch_size=128, shuffle=False)

    class_w = compute_class_weights(d_tr["labels"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    model = CRNNForensics(
        in_channels=1,
        in_freq_dim=in_dim,
        features=feature_type,
        use_gru=True,
        use_attention=True,
        pooling="attention",
        proj_dim=128,
        gru_hidden=128,
        num_gru_layers=2,
        gru_dropout=0.3,
        conv_dropout=0.1,
        classifier_dropout=0.3,
        num_attn_heads=4,
        num_classes=2,
        init_weights=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    augmenter = SpecAugment(freq_mask_max=8, time_mask_max=20, freq_masks_num=2, time_masks_num=2).to(device) if use_spec_augment else None

    start_epoch = 1
    best_dev_eer = float("inf")
    best_dev_loss = float("inf")
    best_dev_thresh = 0.0
    best_epoch = 0
    patience_cnt = 0
    history = []

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"crnn_{feature_type}_seed{seed}.pt"
    best_overall_ckpt = ckpt_dir / f"crnn_{feature_type}_best.pt"
    log_file = Path(f"results/logs/crnn_{feature_type}_seed{seed}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if resume_path and Path(resume_path).exists():
        cp = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(cp["model_state_dict"])
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        start_epoch = cp["epoch"] + 1
        best_dev_eer = cp.get("best_dev_eer", cp.get("dev_eer", float("inf")))
        best_epoch = cp.get("best_epoch", cp.get("epoch", 0))
        history = []
        for past_ep in range(1, start_epoch):
            past_cp_path = ckpt_dir / f"crnn_{feature_type}_epoch{past_ep}.pt"
            if past_cp_path.exists():
                past_cp = torch.load(str(past_cp_path), map_location="cpu", weights_only=False)
                history.append({
                    "epoch": past_ep,
                    "duration_sec": 0.0,
                    "train_loss": past_cp.get("train_loss", 0.0),
                    "dev_loss": past_cp.get("dev_loss", 0.0),
                    "dev_eer": past_cp.get("dev_eer", 0.0),
                    "dev_threshold": past_cp.get("dev_threshold", 0.0),
                    "dev_min_tdcf": past_cp.get("dev_min_tdcf", 0.0),
                    "dev_auc": past_cp.get("dev_auc", 0.0),
                    "lr": lr,
                })
        print(f"Resumed from {resume_path} at epoch {start_epoch} (history reconstructed: {len(history)} epochs)")

    print(f"Starting CRNN-{feature_type.upper()} training on {device} (epochs={epochs}, lr={lr}, seed={seed})")

    t_train_start = time.time()
    for ep in range(start_epoch, epochs + 1):
        t_ep_start = time.time()
        model.train()
        running_loss = 0.0
        n_samples = 0
        for b_x, b_y in train_loader:
            b_x = b_x.to(device=device, dtype=torch.float32).unsqueeze(1)
            b_y = b_y.to(device)
            if augmenter:
                b_x = augmenter(b_x)
            optimizer.zero_grad()
            logits, _ = model(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * len(b_y)
            n_samples += len(b_y)

        train_loss = running_loss / max(1, n_samples)
        dev_loss, dev_scores, dev_labels = evaluate_split(model, dev_loader, device, criterion)

        bf_dev = dev_scores[dev_labels == BONAFIDE]
        sp_dev = dev_scores[dev_labels == SPOOF]
        dev_eer, dev_th = compute_eer(dev_labels, dev_scores)
        dev_auc = compute_roc_auc(dev_labels, dev_scores)
        dev_tdcf, _ = compute_min_tdcf(dev_labels, dev_scores, asv_score_file=dev_asv)

        scheduler.step(dev_eer)
        curr_lr = optimizer.param_groups[0]["lr"]
        ep_duration = time.time() - t_ep_start

        log_msg = (
            f"Epoch {ep:02d}/{epochs:02d} ({ep_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | "
            f"Dev EER: {dev_eer:.4f}% | Dev min-tDCF: {dev_tdcf:.4f} | "
            f"Dev AUC: {dev_auc:.4f} | LR: {curr_lr:.2e}"
        )
        print(log_msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

        history.append({
            "epoch": ep,
            "duration_sec": ep_duration,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_eer": dev_eer,
            "dev_threshold": dev_th,
            "dev_min_tdcf": dev_tdcf,
            "dev_auc": dev_auc,
            "lr": curr_lr,
        })

        ep_ckpt = ckpt_dir / f"crnn_{feature_type}_epoch{ep}.pt"
        torch.save({
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_eer": dev_eer,
            "dev_threshold": dev_th,
            "dev_min_tdcf": dev_tdcf,
            "dev_auc": dev_auc,
            "best_dev_eer": best_dev_eer,
            "best_epoch": best_epoch,
            "seed": seed,
            "features": feature_type,
        }, str(ep_ckpt))

        if dev_eer < best_dev_eer or (abs(dev_eer - best_dev_eer) < 1e-5 and dev_loss < best_dev_loss):
            best_dev_eer = dev_eer
            best_dev_loss = dev_loss
            best_dev_thresh = dev_th
            best_epoch = ep
            patience_cnt = 0
            best_state = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_eer": dev_eer,
                "dev_threshold": dev_th,
                "dev_min_tdcf": dev_tdcf,
                "dev_auc": dev_auc,
                "seed": seed,
                "features": feature_type,
            }
            torch.save(best_state, str(best_ckpt))
            if seed == 42:
                torch.save(best_state, str(best_overall_ckpt))
            print(f" -> New best model for seed {seed} at epoch {ep} (Dev EER: {dev_eer:.4f}%, Dev min-tDCF: {dev_tdcf:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping triggered after {patience} epochs without Dev EER improvement.")
                break

    tot_train_time = time.time() - t_train_start
    print(f"Training completed in {tot_train_time:.1f}s. Restoring best checkpoint from epoch {best_epoch}...")
    cp_best = torch.load(str(best_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(cp_best["model_state_dict"])
    model.eval()

    _, best_dev_scores, best_dev_labels = evaluate_split(model, dev_loader, device, criterion)
    calibrator = PlattCalibrator()
    calibrator.fit(best_dev_scores, best_dev_labels)
    calib_path = ckpt_dir / f"calibrator_crnn_{feature_type}_seed{seed}.joblib"
    calibrator.save(calib_path)
    if seed == 42:
        calibrator.save(ckpt_dir / f"calibrator_crnn_{feature_type}_best.joblib")

    print("Evaluating best checkpoint on frozen EVAL split (71,237 trials)...")
    t_ev_start = time.time()
    _, eval_scores, eval_labels = evaluate_split(model, eval_loader, device, criterion)
    eval_inference_time = time.time() - t_ev_start

    dev_eer, dev_th = compute_eer(best_dev_labels, best_dev_scores)
    dev_tdcf, _ = compute_min_tdcf(best_dev_labels, best_dev_scores, asv_score_file=dev_asv)
    dev_auc = compute_roc_auc(best_dev_labels, best_dev_scores)

    eval_opt_eer, eval_opt_th = compute_eer(eval_labels, eval_scores)
    eval_tdcf, _ = compute_min_tdcf(eval_labels, eval_scores, asv_score_file=eval_asv)
    eval_auc = compute_roc_auc(eval_labels, eval_scores)

    def eer_wrapper(l, s):
        return compute_eer(l, s)[0]

    def tdcf_dev_wrapper(l, s):
        return compute_min_tdcf(l, s, asv_score_file=dev_asv)[0]

    def tdcf_eval_wrapper(l, s):
        return compute_min_tdcf(l, s, asv_score_file=eval_asv)[0]

    print(f"Calculating {n_bootstraps} bootstrap confidence intervals...")
    dev_eer_ci = bootstrap_confidence_interval(eer_wrapper, best_dev_labels, best_dev_scores, n_bootstraps=n_bootstraps, seed=seed)
    dev_tdcf_ci = bootstrap_confidence_interval(tdcf_dev_wrapper, best_dev_labels, best_dev_scores, n_bootstraps=n_bootstraps, seed=seed)

    eval_eer_ci = bootstrap_confidence_interval(eer_wrapper, eval_labels, eval_scores, n_bootstraps=n_bootstraps, seed=seed)
    eval_tdcf_ci = bootstrap_confidence_interval(tdcf_eval_wrapper, eval_labels, eval_scores, n_bootstraps=n_bootstraps, seed=seed)

    eval_at_dev_th = compute_metrics_at_threshold(eval_labels, eval_scores, best_dev_thresh)
    eval_at_opt_th = compute_metrics_at_threshold(eval_labels, eval_scores, eval_opt_th)

    plot_curves(history, Path(f"results/figures/crnn_{feature_type}_training_curves_seed{seed}.png"))
    if seed == 42:
        plot_curves(history, Path(f"results/figures/crnn_{feature_type}_training_curves.png"))

    manifest_sha = compute_file_sha256("data/manifests/train.parquet")
    git_hash = get_git_commit()
    ts = datetime.now(timezone.utc).isoformat()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ckpt_sha256 = compute_file_sha256(str(best_ckpt))

    run_payload = {
        "dataset_name": "ASVspoof 2019 LA",
        "manifest_sha256": manifest_sha,
        "git_commit": git_hash,
        "seed": seed,
        "timestamp": ts,
        "model": f"CRNN-{feature_type.upper()}",
        "features": feature_type,
        "parameters": param_count,
        "training_time_sec": tot_train_time,
        "eval_inference_time_sec": eval_inference_time,
        "checkpoint_path": str(best_ckpt),
        "checkpoint_sha256": ckpt_sha256,
        "dev": {
            "best_epoch": best_epoch,
            "eer": float(dev_eer),
            "eer_threshold": float(best_dev_thresh),
            "roc_auc": float(dev_auc),
            "min_tdcf": float(dev_tdcf),
            "eer_ci_95": [float(dev_eer_ci[0]), float(dev_eer_ci[1])],
            "min_tdcf_ci_95": [float(dev_tdcf_ci[0]), float(dev_tdcf_ci[1])],
        },
        "eval": {
            "eer": float(eval_opt_eer),
            "applied_threshold": float(best_dev_thresh),
            "optimal_threshold": float(eval_opt_th),
            "roc_auc": float(eval_auc),
            "min_tdcf": float(eval_tdcf),
            "eer_ci_95": [float(eval_eer_ci[0]), float(eval_eer_ci[1])],
            "min_tdcf_ci_95": [float(eval_tdcf_ci[0]), float(eval_tdcf_ci[1])],
            "metrics_at_dev_threshold": eval_at_dev_th,
            "metrics_at_optimal_threshold": eval_at_opt_th,
        },
        "history": history,
    }

    guard_results_and_provenance(run_payload)
    out_json = Path(f"results/metrics/crnn_{feature_type}_results_seed{seed}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)

    if seed == 42:
        with open(f"results/metrics/crnn_{feature_type}_results.json", "w", encoding="utf-8") as f:
            json.dump(run_payload, f, indent=2)

    return run_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, choices=["lfcc", "logmel"], default="lfcc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [args.seed]
    all_runs = []
    for s in seeds:
        print(f"\n==================== RUN SEED {s} ({args.features.upper()}) ====================")
        res = run_training(
            feature_type=args.features,
            seed=s,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device_name=args.device,
            use_spec_augment=not args.no_augment,
            resume_path=args.resume,
            n_bootstraps=args.bootstraps,
        )
        all_runs.append(res)

    if len(all_runs) > 1:
        dev_eers = [r["dev"]["eer"] for r in all_runs]
        eval_eers = [r["eval"]["eer"] for r in all_runs]
        eval_tdcfs = [r["eval"]["min_tdcf"] for r in all_runs]
        eval_aucs = [r["eval"]["roc_auc"] for r in all_runs]
        summary_payload = {
            "dataset_name": "ASVspoof 2019 LA",
            "manifest_sha256": all_runs[0]["manifest_sha256"],
            "git_commit": all_runs[0]["git_commit"],
            "seed": seeds[0],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": f"CRNN-{args.features.upper()}",
            "features": args.features,
            "seeds": seeds,
            "metrics": {
                "dev_eer_mean": float(np.mean(dev_eers)),
                "dev_eer_std": float(np.std(dev_eers)),
                "eval_eer_mean": float(np.mean(eval_eers)),
                "eval_eer_std": float(np.std(eval_eers)),
                "eval_min_tdcf_mean": float(np.mean(eval_tdcfs)),
                "eval_min_tdcf_std": float(np.std(eval_tdcfs)),
                "eval_auc_mean": float(np.mean(eval_aucs)),
                "eval_auc_std": float(np.std(eval_aucs)),
            },
            "individual_runs": all_runs,
        }
        guard_results_and_provenance(summary_payload)
        with open(f"results/metrics/crnn_{args.features}_multiseed_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        with open("results/metrics/crnn_multiseed_results.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\nMulti-seed summary saved to results/metrics/crnn_{args.features}_multiseed_summary.json and crnn_multiseed_results.json")
