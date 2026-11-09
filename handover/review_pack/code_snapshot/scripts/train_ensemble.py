import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_curve, auc
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.models.crnn import CRNNForensics
from acforensics.models.calibration import PlattCalibrator, compute_expected_calibration_error
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.ensemble.stacking import NonNegativeStackedEnsemble
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

def score_crnn(model: nn.Module, feats: torch.Tensor, device: torch.device, batch_size: int = 128) -> np.ndarray:
    loader = DataLoader(TensorDataset(feats), batch_size=batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        for (b_x,) in loader:
            b_x = per_utterance_cmvn(b_x.to(device=device, dtype=torch.float32)).unsqueeze(1)
            logits, _ = model(b_x)
            s = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
            scores.append(s)
    return np.concatenate(scores, axis=0)

def score_gmm(feats_list: list[np.ndarray], gmm_bf: GaussianMixture, gmm_sp: GaussianMixture) -> np.ndarray:
    scores = np.zeros(len(feats_list), dtype=np.float64)
    for i, f in enumerate(feats_list):
        t_f = f.T
        ll_bf = np.mean(gmm_bf.score_samples(t_f))
        ll_sp = np.mean(gmm_sp.score_samples(t_f))
        scores[i] = ll_bf - ll_sp
    return scores

def fit_non_negative_meta_learner(Z: np.ndarray, y: np.ndarray, C: float = 1.0) -> tuple[np.ndarray, float]:
    M = Z.shape[1]
    def obj(params):
        w = params[:M]
        b = params[M]
        logits = np.clip(Z @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)) + (0.5 / C) * np.sum(w ** 2)
        grad_l = (p - y) / len(y)
        grad_w = Z.T @ grad_l + (1.0 / C) * w
        grad_b = np.sum(grad_l)
        return loss, np.append(grad_w, grad_b)

    init = np.zeros(M + 1, dtype=np.float64)
    bounds = [(0.0, None) for _ in range(M)] + [(None, None)]
    res = minimize(obj, init, method="L-BFGS-B", bounds=bounds, jac=True)
    w_opt = res.x[:M]
    b_opt = float(res.x[M])
    return w_opt, b_opt

def run_ensemble_training(
    dev_asv: str = "data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt",
    eval_asv: str = "data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
    crnn_ckpt: str = "checkpoints/crnn_lfcc_best.pt",
    seed: int = 42,
    n_bootstraps: int = 1000,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    guard_no_smoke_test(dev_asv)
    guard_no_smoke_test(eval_asv)
    guard_no_smoke_test(crnn_ckpt)

    print("Loading precomputed LFCC caches...")
    d_tr = np.load("data/cache/lfcc_train.npz")
    tr_feats = torch.from_numpy(d_tr["features"])
    tr_pooled = d_tr["pooled"]
    tr_y = d_tr["labels"]

    d_dv = np.load("data/cache/lfcc_dev.npz")
    dv_feats = torch.from_numpy(d_dv["features"])
    dv_pooled = d_dv["pooled"]
    dv_y = d_dv["labels"]

    d_ev = np.load("data/cache/lfcc_eval.npz")
    ev_feats = torch.from_numpy(d_ev["features"])
    ev_pooled = d_ev["pooled"]
    ev_y = d_ev["labels"]

    print("Splitting Train into Train-Fit (80%) and Train-Calibration (20%) with strict speaker disjointness...")
    tr_manifest_path = Path("data/manifests/train.parquet")
    if tr_manifest_path.exists():
        tr_df = pd.read_parquet(tr_manifest_path)
        train_speakers = tr_df["speaker"].to_numpy()
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fit_idx, cal_idx = next(sgkf.split(tr_pooled, tr_y, groups=train_speakers))
        tr_spk_set = set(train_speakers[fit_idx])
        cal_spk_set = set(train_speakers[cal_idx])
        assert len(tr_spk_set.intersection(cal_spk_set)) == 0, "Speaker overlap detected between fit and cal!"
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fit_idx, cal_idx = next(skf.split(tr_pooled, tr_y))

    cal_feats = tr_feats[cal_idx]
    cal_y = tr_y[cal_idx]

    print("Loading and scoring CRNN-LFCC on cal, dev, and eval splits...")
    crnn_model = CRNNForensics(in_channels=1, in_freq_dim=60, features="lfcc").to(device)
    cp = torch.load(crnn_ckpt, map_location=device, weights_only=False)
    crnn_model.load_state_dict(cp["model_state_dict"])
    crnn_model.eval()

    crnn_cal_scores = score_crnn(crnn_model, cal_feats, device)
    crnn_dev_scores = score_crnn(crnn_model, dv_feats, device)
    crnn_eval_scores = score_crnn(crnn_model, ev_feats, device)

    print("Training classical models on Train-Fit (20,304 samples)...")
    fit_features_np = d_tr["features"][fit_idx]
    fit_labels_np = tr_y[fit_idx]
    fit_pooled_np = tr_pooled[fit_idx]
    cal_pooled_np = tr_pooled[cal_idx]

    fit_bf_frames = np.concatenate([f.T for f, l in zip(fit_features_np, fit_labels_np) if l == BONAFIDE], axis=0)
    fit_sp_frames = np.concatenate([f.T for f, l in zip(fit_features_np, fit_labels_np) if l == SPOOF], axis=0)
    bf_sub = fit_bf_frames[np.random.choice(len(fit_bf_frames), min(200000, len(fit_bf_frames)), replace=False)]
    sp_sub = fit_sp_frames[np.random.choice(len(fit_sp_frames), min(200000, len(fit_sp_frames)), replace=False)]

    gmm_bf = GaussianMixture(n_components=64, covariance_type="diag", reg_covar=1e-3, max_iter=100, random_state=seed)
    gmm_sp = GaussianMixture(n_components=64, covariance_type="diag", reg_covar=1e-3, max_iter=100, random_state=seed + 1)
    gmm_bf.fit(bf_sub)
    gmm_sp.fit(sp_sub)

    gmm_cal_scores = score_gmm(list(d_tr["features"][cal_idx]), gmm_bf, gmm_sp)
    gmm_dev_scores = score_gmm(list(d_dv["features"]), gmm_bf, gmm_sp)
    gmm_eval_scores = score_gmm(list(d_ev["features"]), gmm_bf, gmm_sp)

    print("Training Random Forest on Train-Fit pooled features...")
    scaler_pooled = StandardScaler()
    fit_p_scaled = scaler_pooled.fit_transform(fit_pooled_np)
    cal_p_scaled = scaler_pooled.transform(cal_pooled_np)
    dev_p_scaled = scaler_pooled.transform(dv_pooled)
    ev_p_scaled = scaler_pooled.transform(ev_pooled)

    rf = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    rf.fit(fit_p_scaled, fit_labels_np)
    rf_cal_scores = rf.predict_proba(cal_p_scaled)[:, BONAFIDE]
    rf_dev_scores = rf.predict_proba(dev_p_scaled)[:, BONAFIDE]
    rf_eval_scores = rf.predict_proba(ev_p_scaled)[:, BONAFIDE]

    print("Training SVM and Logistic Regression on PCA features...")
    pca = PCA(n_components=0.95, random_state=seed)
    fit_pca = pca.fit_transform(fit_p_scaled)
    cal_pca = pca.transform(cal_p_scaled)
    dev_pca = pca.transform(dev_p_scaled)
    ev_pca = pca.transform(ev_p_scaled)

    sub_n = min(10000, len(fit_pca))
    sub_idx = np.random.choice(len(fit_pca), sub_n, replace=False)
    svm = SVC(C=10.0, gamma="auto", kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    svm.fit(fit_pca[sub_idx], fit_labels_np[sub_idx])
    svm_cal_scores = svm.predict_proba(cal_pca)[:, BONAFIDE]
    svm_dev_scores = svm.predict_proba(dev_pca)[:, BONAFIDE]
    svm_eval_scores = svm.predict_proba(ev_pca)[:, BONAFIDE]

    lr = LogisticRegression(C=100.0, max_iter=1000, class_weight="balanced", random_state=seed)
    lr.fit(fit_pca, fit_labels_np)
    lr_cal_scores = lr.predict_proba(cal_pca)[:, BONAFIDE]
    lr_dev_scores = lr.predict_proba(dev_pca)[:, BONAFIDE]
    lr_eval_scores = lr.predict_proba(ev_pca)[:, BONAFIDE]

    model_names = [
        "CRNN-LFCC",
        "GMM-LFCC",
        "RandomForest-LFCC",
        "SVM-LFCC",
        "LogisticRegression-LFCC",
    ]

    cal_matrix = np.column_stack([crnn_cal_scores, gmm_cal_scores, rf_cal_scores, svm_cal_scores, lr_cal_scores])
    dev_matrix = np.column_stack([crnn_dev_scores, gmm_dev_scores, rf_dev_scores, svm_dev_scores, lr_dev_scores])
    eval_matrix = np.column_stack([crnn_eval_scores, gmm_eval_scores, rf_eval_scores, svm_eval_scores, lr_eval_scores])

    print("Fitting Non-Negative Meta-Learner on Train-Calibration score matrix...")
    ensemble_scaler = StandardScaler()
    Z_cal = ensemble_scaler.fit_transform(cal_matrix)

    weights_opt, intercept_opt = fit_non_negative_meta_learner(Z_cal, cal_y, C=1.0)
    for name, w in zip(model_names, weights_opt):
        print(f"  -> Model '{name}': weight = {w:.4f}")
    print(f"  -> Intercept: {intercept_opt:.4f}")

    ensemble = NonNegativeStackedEnsemble(model_names, weights_opt, intercept_opt, ensemble_scaler)
    ckpt_dir = Path("checkpoints/ensemble")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, str(ckpt_dir / "stacked_ensemble.joblib"))
    classical_bundle = {
        "scaler_pooled": scaler_pooled,
        "rf": rf,
        "pca": pca,
        "svm": svm,
        "lr": lr,
        "gmm_bf": gmm_bf,
        "gmm_sp": gmm_sp,
    }
    joblib.dump(classical_bundle, str(ckpt_dir / "classical_bundle.joblib"))

    print("Evaluating models on Dev split to determine operating thresholds...")
    dev_stack_scores = ensemble.predict_score(dev_matrix)
    dev_stack_eer, dev_stack_th = compute_eer(dv_y, dev_stack_scores)
    dev_stack_tdcf, _ = compute_min_tdcf(dv_y, dev_stack_scores, asv_score_file=dev_asv)
    dev_stack_auc = compute_roc_auc(dv_y, dev_stack_scores)

    dev_mean_scores = ensemble.predict_baseline_mean(dev_matrix)
    dev_mean_eer, dev_mean_th = compute_eer(dv_y, dev_mean_scores)
    dev_mean_tdcf, _ = compute_min_tdcf(dv_y, dev_mean_scores, asv_score_file=dev_asv)
    dev_mean_auc = compute_roc_auc(dv_y, dev_mean_scores)

    dev_crnn_eer, dev_crnn_th = compute_eer(dv_y, crnn_dev_scores)
    dev_crnn_tdcf, _ = compute_min_tdcf(dv_y, crnn_dev_scores, asv_score_file=dev_asv)
    dev_crnn_auc = compute_roc_auc(dv_y, crnn_dev_scores)

    print("Fitting Platt Calibrator on Dev Fused Scores...")
    platt = PlattCalibrator()
    platt.fit(dev_stack_scores, dv_y)
    platt.save(ckpt_dir / "calibrator_stacked_ensemble.joblib")

    print("Evaluating frozen models on Eval split (71,237 trials)...")
    eval_stack_scores = ensemble.predict_score(eval_matrix)
    eval_mean_scores = ensemble.predict_baseline_mean(eval_matrix)

    eval_stack_eer_opt, _ = compute_eer(ev_y, eval_stack_scores)
    eval_stack_tdcf, _ = compute_min_tdcf(ev_y, eval_stack_scores, asv_score_file=eval_asv)
    eval_stack_auc = compute_roc_auc(ev_y, eval_stack_scores)
    eval_stack_metrics = compute_metrics_at_threshold(ev_y, eval_stack_scores, dev_stack_th)

    eval_mean_eer_opt, _ = compute_eer(ev_y, eval_mean_scores)
    eval_mean_tdcf, _ = compute_min_tdcf(ev_y, eval_mean_scores, asv_score_file=eval_asv)
    eval_mean_auc = compute_roc_auc(ev_y, eval_mean_scores)

    eval_crnn_eer_opt, _ = compute_eer(ev_y, crnn_eval_scores)
    eval_crnn_tdcf, _ = compute_min_tdcf(ev_y, crnn_eval_scores, asv_score_file=eval_asv)
    eval_crnn_auc = compute_roc_auc(ev_y, crnn_eval_scores)

    eval_calibrated_probs = platt.predict_proba(eval_stack_scores)
    eval_ece = compute_expected_calibration_error(ev_y, eval_calibrated_probs, n_bins=10)
    eval_brier = float(brier_score_loss(ev_y, eval_calibrated_probs))

    def eer_fn(l, s):
        return compute_eer(l, s)[0]

    def tdcf_fn(l, s):
        return compute_min_tdcf(l, s, asv_score_file=eval_asv)[0]

    print(f"Calculating {n_bootstraps} bootstrap confidence intervals for all 3 systems...")
    stack_eer_ci = bootstrap_confidence_interval(eer_fn, ev_y, eval_stack_scores, n_bootstraps=n_bootstraps, seed=seed)
    stack_tdcf_ci = bootstrap_confidence_interval(tdcf_fn, ev_y, eval_stack_scores, n_bootstraps=n_bootstraps, seed=seed)

    mean_eer_ci = bootstrap_confidence_interval(eer_fn, ev_y, eval_mean_scores, n_bootstraps=n_bootstraps, seed=seed)
    mean_tdcf_ci = bootstrap_confidence_interval(tdcf_fn, ev_y, eval_mean_scores, n_bootstraps=n_bootstraps, seed=seed)

    crnn_eer_ci = bootstrap_confidence_interval(eer_fn, ev_y, crnn_eval_scores, n_bootstraps=n_bootstraps, seed=seed)
    crnn_tdcf_ci = bootstrap_confidence_interval(tdcf_fn, ev_y, crnn_eval_scores, n_bootstraps=n_bootstraps, seed=seed)

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    prob_true, prob_pred = calibration_curve(ev_y, eval_calibrated_probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(prob_pred, prob_true, "s-", color="#2563EB", lw=2, label=f"Stacked Ensemble (ECE={eval_ece:.4f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Fraction of Positives")
    ax.set_title(f"Reliability Diagram (Brier Score: {eval_brier:.4f})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(fig_dir / "ensemble_reliability_diagram.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)
    fpr_s, tpr_s, _ = roc_curve(ev_y, eval_stack_scores)
    fpr_m, tpr_m, _ = roc_curve(ev_y, eval_mean_scores)
    fpr_c, tpr_c, _ = roc_curve(ev_y, crnn_eval_scores)
    ax.plot(fpr_s, tpr_s, color="#DC2626", lw=2, label=f"Stacked Ensemble (AUC={eval_stack_auc:.4f})")
    ax.plot(fpr_m, tpr_m, color="#D97706", lw=1.5, linestyle="--", label=f"Mean Baseline Fusion (AUC={eval_mean_auc:.4f})")
    ax.plot(fpr_c, tpr_c, color="#2563EB", lw=1.5, linestyle=":", label=f"Best Single: CRNN-LFCC (AUC={eval_crnn_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison: Stacked vs Mean vs Single Model")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(str(fig_dir / "ensemble_roc_curves.png"))
    plt.close()

    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    weights_rows = []
    for name, w in zip(model_names, weights_opt):
        weights_rows.append({
            "Base Model": name,
            "Non-Negative Weight": f"{w:.4f}",
            "Status": "Retained" if w > 0.0 else "Zeroed",
        })
    df_w = pd.DataFrame(weights_rows)
    with open(tables_dir / "meta_learner_weights.md", "w", encoding="utf-8") as f:
        f.write(df_w.to_markdown(index=False) + "\n")
    with open(tables_dir / "meta_learner_weights.tex", "w", encoding="utf-8") as f:
        f.write(df_w.to_latex(index=False) + "\n")

    comp_rows = [
        {
            "System": "Stacked Ensemble (Meta-Learner)",
            "Dev EER (%)": f"{dev_stack_eer:.2f}",
            "Eval EER (%) [95% CI]": f"{eval_stack_eer_opt:.2f} [{stack_eer_ci[0]:.2f}, {stack_eer_ci[1]:.2f}]",
            "Eval min-tDCF [95% CI]": f"{eval_stack_tdcf:.4f} [{stack_tdcf_ci[0]:.4f}, {stack_tdcf_ci[1]:.4f}]",
            "Eval ROC-AUC": f"{eval_stack_auc:.4f}",
            "Dev Applied Thresh": f"{dev_stack_th:.4f}",
        },
        {
            "System": "Baseline Mean Fusion (Unweighted)",
            "Dev EER (%)": f"{dev_mean_eer:.2f}",
            "Eval EER (%) [95% CI]": f"{eval_mean_eer_opt:.2f} [{mean_eer_ci[0]:.2f}, {mean_eer_ci[1]:.2f}]",
            "Eval min-tDCF [95% CI]": f"{eval_mean_tdcf:.4f} [{mean_tdcf_ci[0]:.4f}, {mean_tdcf_ci[1]:.4f}]",
            "Eval ROC-AUC": f"{eval_mean_auc:.4f}",
            "Dev Applied Thresh": f"{dev_mean_th:.4f}",
        },
        {
            "System": "Best Single Model (CRNN-LFCC)",
            "Dev EER (%)": f"{dev_crnn_eer:.2f}",
            "Eval EER (%) [95% CI]": f"{eval_crnn_eer_opt:.2f} [{crnn_eer_ci[0]:.2f}, {crnn_eer_ci[1]:.2f}]",
            "Eval min-tDCF [95% CI]": f"{eval_crnn_tdcf:.4f} [{crnn_tdcf_ci[0]:.4f}, {crnn_tdcf_ci[1]:.4f}]",
            "Eval ROC-AUC": f"{eval_crnn_auc:.4f}",
            "Dev Applied Thresh": f"{dev_crnn_th:.4f}",
        },
    ]
    df_comp = pd.DataFrame(comp_rows)
    with open(tables_dir / "ensemble_comparison.md", "w", encoding="utf-8") as f:
        f.write(df_comp.to_markdown(index=False) + "\n")
    with open(tables_dir / "ensemble_comparison.tex", "w", encoding="utf-8") as f:
        f.write(df_comp.to_latex(index=False) + "\n")

    manifest_sha = compute_file_sha256("data/manifests/train.parquet")
    git_hash = get_git_commit()
    ts = datetime.now(timezone.utc).isoformat()

    full_payload = {
        "dataset_name": "ASVspoof 2019 LA",
        "manifest_sha256": manifest_sha,
        "git_commit": git_hash,
        "seed": seed,
        "timestamp": ts,
        "base_models": model_names,
        "weights": {name: float(w) for name, w in zip(model_names, weights_opt)},
        "intercept": intercept_opt,
        "calibration": {
            "eval_ece": float(eval_ece),
            "eval_brier_score": float(eval_brier),
        },
        "stacked_ensemble": {
            "dev": {
                "eer": float(dev_stack_eer),
                "threshold": float(dev_stack_th),
                "min_tdcf": float(dev_stack_tdcf),
                "roc_auc": float(dev_stack_auc),
            },
            "eval": {
                "eer": float(eval_stack_eer_opt),
                "applied_threshold": float(dev_stack_th),
                "min_tdcf": float(eval_stack_tdcf),
                "roc_auc": float(eval_stack_auc),
                "eer_ci_95": [float(stack_eer_ci[0]), float(stack_eer_ci[1])],
                "min_tdcf_ci_95": [float(stack_tdcf_ci[0]), float(stack_tdcf_ci[1])],
                "metrics_at_dev_threshold": eval_stack_metrics,
            },
        },
        "mean_baseline_fusion": {
            "dev": {
                "eer": float(dev_mean_eer),
                "threshold": float(dev_mean_th),
                "min_tdcf": float(dev_mean_tdcf),
                "roc_auc": float(dev_mean_auc),
            },
            "eval": {
                "eer": float(eval_mean_eer_opt),
                "applied_threshold": float(dev_mean_th),
                "min_tdcf": float(eval_mean_tdcf),
                "roc_auc": float(eval_mean_auc),
                "eer_ci_95": [float(mean_eer_ci[0]), float(mean_eer_ci[1])],
                "min_tdcf_ci_95": [float(mean_tdcf_ci[0]), float(mean_tdcf_ci[1])],
            },
        },
        "best_single_model": {
            "dev": {
                "eer": float(dev_crnn_eer),
                "threshold": float(dev_crnn_th),
                "min_tdcf": float(dev_crnn_tdcf),
                "roc_auc": float(dev_crnn_auc),
            },
            "eval": {
                "eer": float(eval_crnn_eer_opt),
                "applied_threshold": float(dev_crnn_th),
                "min_tdcf": float(eval_crnn_tdcf),
                "roc_auc": float(eval_crnn_auc),
                "eer_ci_95": [float(crnn_eer_ci[0]), float(crnn_eer_ci[1])],
                "min_tdcf_ci_95": [float(crnn_tdcf_ci[0]), float(crnn_tdcf_ci[1])],
            },
        },
    }

    guard_results_and_provenance(full_payload)
    out_json = Path("results/metrics/ensemble_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("Ensemble training and benchmark completed successfully.")
    return full_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crnn-ckpt", type=str, default="checkpoints/crnn_lfcc_best.pt")
    parser.add_argument("--dev-asv", type=str, default="data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt")
    parser.add_argument("--eval-asv", type=str, default="data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()

    run_ensemble_training(
        dev_asv=args.dev_asv,
        eval_asv=args.eval_asv,
        crnn_ckpt=args.crnn_ckpt,
        seed=args.seed,
        n_bootstraps=args.bootstraps,
    )

