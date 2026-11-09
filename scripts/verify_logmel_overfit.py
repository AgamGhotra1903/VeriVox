import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.evaluation.provenance import (
    create_provenance_record,
    guard_results_and_provenance,
)
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics


def run_logmel_overfit_check(
    cache_path: str = "data/cache/logmel_train.npz",
    out_json: str = "results/metrics/logmel_overfit_check.json",
    batch_size: int = 64,
    steps: int = 60,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(cache_path)
    all_feats = data["features"]
    all_labels = data["labels"]

    idx_pos = np.where(all_labels == BONAFIDE)[0][: batch_size // 2]
    idx_neg = np.where(all_labels == SPOOF)[0][: batch_size // 2]
    sel_idx = np.concatenate([idx_pos, idx_neg])

    b_raw = all_feats[sel_idx].astype(np.float32)
    b_lbls = all_labels[sel_idx].astype(np.int64)

    raw_min = float(np.min(b_raw))
    raw_max = float(np.max(b_raw))
    raw_mean = float(np.mean(b_raw))
    raw_std = float(np.std(b_raw))
    has_nan = bool(np.isnan(b_raw).any())
    has_inf = bool(np.isinf(b_raw).any())

    t_raw = torch.from_numpy(b_raw)
    t_norm = per_utterance_cmvn(t_raw).unsqueeze(1)
    norm_mean = float(t_norm.mean().item())
    norm_std = float(t_norm.std().item())

    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    t_norm = t_norm.to(device)
    t_targets = torch.from_numpy(b_lbls).to(device)

    model = CRNNForensics(in_channels=1, in_freq_dim=80, features="logmel").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    loss_history = []
    dead_relu_counts = []

    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        logits, _ = model(t_norm)
        loss = criterion(logits, t_targets)
        loss.backward()
        optimizer.step()

        cur_loss = float(loss.item())
        loss_history.append(cur_loss)

        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = float((preds == t_targets).float().mean().item() * 100.0)
                dead_fraction = float((logits <= 0).float().mean().item())
                dead_relu_counts.append(dead_fraction)
                print(f"Step {step:02d}: Loss = {cur_loss:.5f}, Accuracy = {acc:.1f}%")

    final_loss = loss_history[-1]
    final_acc = float((logits.argmax(dim=-1) == t_targets).float().mean().item() * 100.0)

    prov = create_provenance_record("logmel_overfit_check", cache_path, seed=seed)
    payload = {
        **prov,
        "n_samples": batch_size,
        "subset": True,
        "subset_flag": True,
        "raw_statistics": {
            "min_db": raw_min,
            "max_db": raw_max,
            "mean_db": raw_mean,
            "std_db": raw_std,
            "has_nan": has_nan,
            "has_inf": has_inf,
        },
        "normalized_statistics": {
            "mean": norm_mean,
            "std": norm_std,
        },
        "overfit_experiment": {
            "batch_size": batch_size,
            "steps": steps,
            "learning_rate": lr,
            "initial_loss": loss_history[0],
            "final_loss": final_loss,
            "final_accuracy_pct": final_acc,
            "converged_to_zero": bool(final_loss < 0.05),
        },
        "loss_history": loss_history,
    }

    guard_results_and_provenance(payload)

    p_out = Path(out_json)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Log-Mel overfit verification saved to {p_out} (Final loss: {final_loss:.6f}, Acc: {final_acc:.1f}%)")
    return payload


if __name__ == "__main__":
    run_logmel_overfit_check()
