import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.models.crnn import CRNNForensics
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_min_tdcf,
)
from acforensics.evaluation.provenance import (
    compute_file_sha256,
    get_git_commit,
    guard_results_and_provenance,
    guard_no_smoke_test,
)

def evaluate_features_on_model(
    model: nn.Module,
    feats: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    asv_path: str,
    batch_size: int = 128,
) -> dict:
    loader = DataLoader(TensorDataset(feats, labels), batch_size=batch_size, shuffle=False)
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for b_x, b_y in loader:
            b_x = per_utterance_cmvn(b_x.to(device=device, dtype=torch.float32)).unsqueeze(1)
            logits, _ = model(b_x)
            scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(b_y.numpy())
    scores_arr = np.concatenate(all_scores, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    eer, th = compute_eer(labels_arr, scores_arr)
    auc = compute_roc_auc(labels_arr, scores_arr)
    tdcf, _ = compute_min_tdcf(labels_arr, scores_arr, asv_score_file=asv_path)
    return {
        "eer": float(eer),
        "threshold": float(th),
        "roc_auc": float(auc),
        "min_tdcf": float(tdcf),
    }

def apply_spectral_occlusion(feats: torch.Tensor, min_band: int, max_band: int, n_filters: int = 20) -> torch.Tensor:
    cloned = feats.clone()
    for delta_idx in [0, 1, 2]:
        start = (delta_idx * n_filters) + min_band
        end = (delta_idx * n_filters) + max_band
        cloned[:, start:end, :] = 0.0
    return cloned

def apply_lowpass_occlusion(feats: torch.Tensor, cutoff_khz: float, n_filters: int = 20, max_khz: float = 8.0) -> torch.Tensor:
    cloned = feats.clone()
    cutoff_band = int(round((cutoff_khz / max_khz) * n_filters))
    for delta_idx in [0, 1, 2]:
        start = (delta_idx * n_filters) + cutoff_band
        end = (delta_idx * n_filters) + n_filters
        cloned[:, start:end, :] = 0.0
    return cloned

def run_band_analysis(
    checkpoint_path: str = "checkpoints/crnn_lfcc_best.pt",
    dev_cache_path: str = "data/cache/lfcc_dev.npz",
    dev_asv: str = "data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt",
    out_dir: str = "results",
) -> dict:
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    guard_no_smoke_test(checkpoint_path)
    guard_no_smoke_test(dev_cache_path)

    d_dv = np.load(dev_cache_path)
    raw_feats = torch.from_numpy(d_dv["features"])
    raw_labels = torch.from_numpy(d_dv["labels"])

    model = CRNNForensics(in_channels=1, in_freq_dim=60, features="lfcc").to(device)
    cp = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    print("Evaluating baseline (full 0-8 kHz bandwidth)...")
    baseline_res = evaluate_features_on_model(model, raw_feats, raw_labels, device, dev_asv)
    base_eer = baseline_res["eer"]

    lowpass_cutoffs = [7.0, 6.0, 4.0]
    lowpass_results = {}
    for cut in lowpass_cutoffs:
        print(f"Evaluating Low-Pass at {cut:.1f} kHz...")
        occluded = apply_lowpass_occlusion(raw_feats, cutoff_khz=cut, n_filters=20, max_khz=8.0)
        res = evaluate_features_on_model(model, occluded, raw_labels, device, dev_asv)
        res["delta_eer"] = float(res["eer"] - base_eer)
        lowpass_results[f"lowpass_{int(cut)}khz"] = res

    band_spans = [
        ("band_0_2khz", 0, 5, "0-2 kHz"),
        ("band_2_4khz", 5, 10, "2-4 kHz"),
        ("band_4_6khz", 10, 15, "4-6 kHz"),
        ("band_6_8khz", 15, 20, "6-8 kHz"),
    ]
    occlusion_results = {}
    for key, b_start, b_end, label in band_spans:
        print(f"Evaluating Occlusion of {label}...")
        occluded = apply_spectral_occlusion(raw_feats, min_band=b_start, max_band=b_end, n_filters=20)
        res = evaluate_features_on_model(model, occluded, raw_labels, device, dev_asv)
        res["delta_eer"] = float(res["eer"] - base_eer)
        res["label"] = label
        occlusion_results[key] = res

    manifest_sha = compute_file_sha256("data/manifests/train.parquet")
    git_hash = get_git_commit()
    ts = datetime.now(timezone.utc).isoformat()

    full_payload = {
        "dataset_name": "ASVspoof 2019 LA",
        "manifest_sha256": manifest_sha,
        "git_commit": git_hash,
        "seed": 42,
        "timestamp": ts,
        "model": "CRNN-LFCC",
        "baseline_full_bandwidth": baseline_res,
        "lowpass_analysis": lowpass_results,
        "band_occlusion_analysis": occlusion_results,
    }

    guard_results_and_provenance(full_payload)
    out_p = Path(out_dir) / "metrics" / "band_analysis_results.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    rows = [{
        "Condition": "Full Bandwidth (0-8 kHz)",
        "Dev EER (%)": f"{baseline_res['eer']:.2f}",
        "Delta EER (pts)": "0.00",
        "Dev min-tDCF": f"{baseline_res['min_tdcf']:.4f}",
        "Dev ROC-AUC": f"{baseline_res['roc_auc']:.4f}",
    }]
    for k, v in lowpass_results.items():
        cut_label = k.replace("lowpass_", "Low-Pass < ").replace("khz", " kHz")
        rows.append({
            "Condition": cut_label,
            "Dev EER (%)": f"{v['eer']:.2f}",
            "Delta EER (pts)": f"{v['delta_eer']:+.2f}",
            "Dev min-tDCF": f"{v['min_tdcf']:.4f}",
            "Dev ROC-AUC": f"{v['roc_auc']:.4f}",
        })
    for k, v in occlusion_results.items():
        rows.append({
            "Condition": f"Occlude {v['label']}",
            "Dev EER (%)": f"{v['eer']:.2f}",
            "Delta EER (pts)": f"{v['delta_eer']:+.2f}",
            "Dev min-tDCF": f"{v['min_tdcf']:.4f}",
            "Dev ROC-AUC": f"{v['roc_auc']:.4f}",
        })

    df_bands = pd.DataFrame(rows)
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "table_band_analysis.md", "w", encoding="utf-8") as f:
        f.write(df_bands.to_markdown(index=False) + "\n")
    with open(tables_dir / "table_band_analysis.tex", "w", encoding="utf-8") as f:
        f.write(df_bands.to_latex(index=False) + "\n")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    conds = [r["Condition"] for r in rows[1:]]
    deltas = [float(r["Delta EER (pts)"]) for r in rows[1:]]
    colors = ["#EF4444" if d > 5.0 else "#F59E0B" if d > 1.0 else "#3B82F6" for d in deltas]
    bars = ax.barh(conds, deltas, color=colors)
    ax.set_xlabel(r"$\Delta$ Dev EER (percentage points increase)")
    ax.set_title("Spectral Band Impact on CRNN-LFCC Decision Making")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"+{d:.2f}%", va="center", fontsize=9)
    plt.tight_layout()
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fig_dir / "band_analysis_degradation.png"))
    plt.close()

    print("Band analysis completed successfully.")
    return full_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crnn_lfcc_best.pt")
    parser.add_argument("--cache", type=str, default="data/cache/lfcc_dev.npz")
    parser.add_argument("--asv", type=str, default="data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()
    run_band_analysis(
        checkpoint_path=args.checkpoint,
        dev_cache_path=args.cache,
        dev_asv=args.asv,
        out_dir=args.out_dir,
    )

