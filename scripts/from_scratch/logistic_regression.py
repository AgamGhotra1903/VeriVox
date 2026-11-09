from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

_repo = Path(__file__).resolve().parent.parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.evaluation.metrics import compute_eer, compute_roc_auc

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

def compute_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2_reg: float = 0.0) -> float:
    p = sigmoid(X @ w)
    eps = 1e-15
    bce = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
    reg = 0.5 * l2_reg * np.sum(w[1:] ** 2)
    return float(bce + reg)

def compute_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2_reg: float = 0.0) -> np.ndarray:
    n = len(y)
    p = sigmoid(X @ w)
    grad = (X.T @ (p - y)) / n
    if l2_reg > 0.0:
        reg_term = l2_reg * w
        reg_term[0] = 0.0
        grad += reg_term
    return grad

def fit_batch_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iters: int = 300, l2_reg: float = 0.01) -> tuple[np.ndarray, list[float]]:
    w = np.zeros(X.shape[1], dtype=np.float64)
    losses = []
    for _ in range(n_iters):
        grad = compute_gradient(X, y, w, l2_reg)
        w -= lr * grad
        losses.append(compute_loss(X, y, w, l2_reg))
    return w, losses

def fit_sgd(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 15, l2_reg: float = 0.01, seed: int = 42) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1], dtype=np.float64)
    losses = []
    n = len(y)
    for _ in range(epochs):
        indices = rng.permutation(n)
        for idx in indices:
            xi = X[idx : idx + 1]
            yi = y[idx : idx + 1]
            grad = compute_gradient(xi, yi, w, l2_reg)
            w -= lr * grad
        losses.append(compute_loss(X, y, w, l2_reg))
    return w, losses

def fit_minibatch_gd(X: np.ndarray, y: np.ndarray, batch_size: int = 64, lr: float = 0.05, epochs: int = 30, l2_reg: float = 0.01, seed: int = 42) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1], dtype=np.float64)
    losses = []
    n = len(y)
    for _ in range(epochs):
        indices = rng.permutation(n)
        for start_idx in range(0, n, batch_size):
            b_idx = indices[start_idx : start_idx + batch_size]
            xb = X[b_idx]
            yb = y[b_idx]
            grad = compute_gradient(xb, yb, w, l2_reg)
            w -= lr * grad
        losses.append(compute_loss(X, y, w, l2_reg))
    return w, losses

def fit_newton(X: np.ndarray, y: np.ndarray, n_iters: int = 15, l2_reg: float = 0.01, damping: float = 1e-3) -> tuple[np.ndarray, list[float]]:
    w = np.zeros(X.shape[1], dtype=np.float64)
    losses = [compute_loss(X, y, w, l2_reg)]
    n, d = X.shape
    for _ in range(n_iters):
        p = sigmoid(X @ w)
        w_diag = p * (1.0 - p)
        H = (X.T @ (w_diag[:, None] * X)) / n
        if l2_reg > 0.0:
            reg_mat = l2_reg * np.eye(d)
            reg_mat[0, 0] = 0.0
            H += reg_mat
        H += damping * np.eye(d)
        grad = compute_gradient(X, y, w, l2_reg)
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(H) @ grad
        w -= delta
        losses.append(compute_loss(X, y, w, l2_reg))
    return w, losses

def run_scratch_logistic_regression(seed: int = 42, n_train: int = 5000) -> dict:
    rng = np.random.default_rng(seed)

    d_tr = np.load("data/cache/lfcc_train.npz")
    d_dv = np.load("data/cache/lfcc_dev.npz")

    tr_pooled = d_tr["pooled"]
    tr_labels = d_tr["labels"]
    dv_pooled = d_dv["pooled"]
    dv_labels = d_dv["labels"]

    sub_idx = rng.choice(len(tr_labels), size=min(n_train, len(tr_labels)), replace=False)
    X_tr_sub = tr_pooled[sub_idx]
    y_tr_sub = tr_labels[sub_idx]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_sub)
    X_dv_scaled = scaler.transform(dv_pooled)

    X_tr_b = np.hstack([np.ones((len(X_tr_scaled), 1)), X_tr_scaled])
    X_dv_b = np.hstack([np.ones((len(X_dv_scaled), 1)), X_dv_scaled])

    print("Fitting Scratch Logistic Regression with 4 Optimizers...")
    l2 = 0.01

    w_bgd, loss_bgd = fit_batch_gd(X_tr_b, y_tr_sub, lr=0.1, n_iters=250, l2_reg=l2)
    s_bgd = X_dv_b @ w_bgd
    eer_bgd, _ = compute_eer(dv_labels, s_bgd)

    w_sgd, loss_sgd = fit_sgd(X_tr_b, y_tr_sub, lr=0.01, epochs=20, l2_reg=l2, seed=seed)
    s_sgd = X_dv_b @ w_sgd
    eer_sgd, _ = compute_eer(dv_labels, s_sgd)

    w_mb, loss_mb = fit_minibatch_gd(X_tr_b, y_tr_sub, batch_size=64, lr=0.05, epochs=30, l2_reg=l2, seed=seed)
    s_mb = X_dv_b @ w_mb
    eer_mb, _ = compute_eer(dv_labels, s_mb)

    w_newt, loss_newt = fit_newton(X_tr_b, y_tr_sub, n_iters=15, l2_reg=l2)
    s_newt = X_dv_b @ w_newt
    eer_newt, _ = compute_eer(dv_labels, s_newt)

    print("Fitting scikit-learn reference Logistic Regression...")
    sk_lr = SklearnLR(C=1.0 / (l2 * len(y_tr_sub)), penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed)
    sk_lr.fit(X_tr_scaled, y_tr_sub)
    s_sk = sk_lr.decision_function(X_dv_scaled)
    eer_sk, _ = compute_eer(dv_labels, s_sk)

    delta_eer_newt = abs(eer_newt - eer_sk)
    print(f"Dev EER comparison: Sklearn={eer_sk:.2f}%, Newton Scratch={eer_newt:.2f}%, Delta={delta_eer_newt:.2f} pts")

    fig_dir = Path("results/figures/from_scratch")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.plot(loss_bgd, label=f"Batch GD (Final EER: {eer_bgd:.2f}%)", color="#2563EB", lw=2)
    ax.plot(np.linspace(0, len(loss_bgd) - 1, len(loss_sgd)), loss_sgd, label=f"SGD (Final EER: {eer_sgd:.2f}%)", color="#D97706", lw=2)
    ax.plot(np.linspace(0, len(loss_bgd) - 1, len(loss_mb)), loss_mb, label=f"Mini-Batch GD (Final EER: {eer_mb:.2f}%)", color="#059669", lw=2)
    ax.plot(np.linspace(0, len(loss_bgd) - 1, len(loss_newt)), loss_newt, label=f"Newton-Raphson (Final EER: {eer_newt:.2f}%)", color="#DC2626", lw=2, linestyle="--")
    ax.set_xlabel("Relative Iteration Progress")
    ax.set_ylabel("Regularized BCE Loss")
    ax.set_title("From-Scratch Logistic Regression: Optimizer Convergence")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "logistic_regression_loss_curves.png")
    plt.close()

    print("Executing 5-Fold Speaker-Grouped CV for L1 (Lasso) vs L2 (Ridge)...")
    df_manifest = pd.read_parquet("data/manifests/train.parquet")
    speakers_sub = df_manifest["speaker"].iloc[sub_idx].values

    c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    gkf = GroupKFold(n_splits=5)

    l1_cv_eers = []
    l2_cv_eers = []
    l1_nonzero_weights = []
    l2_nonzero_weights = []

    for c in c_values:
        fold_l1 = []
        fold_l2 = []
        nz_l1 = []
        nz_l2 = []
        for tr_fold, val_fold in gkf.split(X_tr_scaled, y_tr_sub, groups=speakers_sub):
            m_l1 = SklearnLR(C=c, penalty="l1", solver="saga", max_iter=500, random_state=seed)
            m_l1.fit(X_tr_scaled[tr_fold], y_tr_sub[tr_fold])
            s1 = m_l1.decision_function(X_tr_scaled[val_fold])
            fold_l1.append(compute_eer(y_tr_sub[val_fold], s1)[0])
            nz_l1.append(np.sum(np.abs(m_l1.coef_) > 1e-4))

            m_l2 = SklearnLR(C=c, penalty="l2", solver="lbfgs", max_iter=500, random_state=seed)
            m_l2.fit(X_tr_scaled[tr_fold], y_tr_sub[tr_fold])
            s2 = m_l2.decision_function(X_tr_scaled[val_fold])
            fold_l2.append(compute_eer(y_tr_sub[val_fold], s2)[0])
            nz_l2.append(np.sum(np.abs(m_l2.coef_) > 1e-4))

        l1_cv_eers.append(float(np.mean(fold_l1)))
        l2_cv_eers.append(float(np.mean(fold_l2)))
        l1_nonzero_weights.append(float(np.mean(nz_l1)))
        l2_nonzero_weights.append(float(np.mean(nz_l2)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
    ax1.plot(c_values, l1_cv_eers, marker="o", label="L1 (Lasso, SAGA)", color="#DC2626", lw=2)
    ax1.plot(c_values, l2_cv_eers, marker="s", label="L2 (Ridge, L-BFGS)", color="#2563EB", lw=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("Inverse Regularization Strength C")
    ax1.set_ylabel("5-Fold Speaker CV EER (%)")
    ax1.set_title("Regularization Strength vs Speaker-CV EER")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(c_values, l1_nonzero_weights, marker="o", label="L1 Non-Zero Weights (Sparsity)", color="#DC2626", lw=2)
    ax2.plot(c_values, l2_nonzero_weights, marker="s", label="L2 Non-Zero Weights (Dense)", color="#2563EB", lw=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Inverse Regularization Strength C")
    ax2.set_ylabel("Active Features Count (out of 120)")
    ax2.set_title("L1 Feature Sparsity vs L2 Shrinkage")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "lr_ridge_vs_lasso.png")
    plt.close()

    metrics_payload = {
        "dataset_name": "ASVspoof 2019 LA Pooled LFCC",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_train_subsample": n_train,
        "scratch_optimizers": {
            "batch_gd": {"dev_eer": float(eer_bgd), "final_loss": float(loss_bgd[-1])},
            "sgd": {"dev_eer": float(eer_sgd), "final_loss": float(loss_sgd[-1])},
            "mini_batch_gd": {"dev_eer": float(eer_mb), "final_loss": float(loss_mb[-1])},
            "newton": {"dev_eer": float(eer_newt), "final_loss": float(loss_newt[-1])},
        },
        "sklearn_reference": {
            "dev_eer": float(eer_sk),
            "delta_newton_vs_sklearn_pts": float(delta_eer_newt),
            "matches_within_0_5_pts": bool(delta_eer_newt <= 0.5),
        },
        "cv_regularization_sweep": {
            "c_values": c_values,
            "l1_cv_eers": l1_cv_eers,
            "l2_cv_eers": l2_cv_eers,
            "l1_nonzero_weights": l1_nonzero_weights,
            "l2_nonzero_weights": l2_nonzero_weights,
            "best_c_l1": float(c_values[int(np.argmin(l1_cv_eers))]),
            "best_c_l2": float(c_values[int(np.argmin(l2_cv_eers))]),
        },
    }

    out_json = Path("results/metrics/syllabus_scratch_logistic_regression.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("Scratch Logistic Regression suite completed successfully!")
    return metrics_payload

if __name__ == "__main__":
    run_scratch_logistic_regression()

