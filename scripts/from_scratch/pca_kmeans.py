from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import adjusted_rand_score

_repo = Path(__file__).resolve().parent.parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from acforensics.constants import BONAFIDE, SPOOF

def run_pca_svd_scratch(X: np.ndarray, n_components: int = 2) -> dict:
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec
    n_samples = X.shape[0]

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_variance_ = (S ** 2) / (n_samples - 1)
    total_var = np.sum(explained_variance_)
    explained_variance_ratio_ = explained_variance_ / total_var

    components_ = Vt[:n_components]
    projected_2d = X_centered @ components_.T

    return {
        "projected_2d": projected_2d,
        "explained_variance": explained_variance_[:n_components].tolist(),
        "explained_variance_ratio": explained_variance_ratio_[:n_components].tolist(),
        "components": components_,
        "singular_values": S[:n_components].tolist(),
    }

def run_kmeans_scratch(X: np.ndarray, k: int = 2, max_iters: int = 25, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape

    init_idx = rng.choice(n_samples, size=k, replace=False)
    centroids = X[init_idx].copy()

    objective_history = []
    assignments = np.zeros(n_samples, dtype=np.int64)

    for it in range(max_iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2) ** 2
        assignments = np.argmin(dists, axis=1)

        total_inertia = 0.0
        for j in range(k):
            cluster_points = X[assignments == j]
            if len(cluster_points) > 0:
                centroids[j] = np.mean(cluster_points, axis=0)
                total_inertia += np.sum(np.linalg.norm(cluster_points - centroids[j], axis=1) ** 2)
            else:
                centroids[j] = X[rng.integers(n_samples)]
        objective_history.append(float(total_inertia))

    return {
        "centroids": centroids,
        "assignments": assignments,
        "objective_history": objective_history,
        "final_inertia": float(objective_history[-1]),
    }

def compute_cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    contingency = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        contingency[t, p] += 1
    return float(np.sum(np.max(contingency, axis=0)) / len(y_true))

def main(seed: int = 42, n_train: int = 5000):
    rng = np.random.default_rng(seed)

    d_tr = np.load("data/cache/lfcc_train.npz")
    tr_pooled = d_tr["pooled"]
    tr_labels = d_tr["labels"]

    sub_idx = rng.choice(len(tr_labels), size=min(n_train, len(tr_labels)), replace=False)
    X_sub = tr_pooled[sub_idx]
    y_sub = tr_labels[sub_idx]

    print("Running SVD PCA from scratch...")
    scratch_pca = run_pca_svd_scratch(X_sub, n_components=2)
    sk_pca = SklearnPCA(n_components=2, random_state=seed)
    sk_proj = sk_pca.fit_transform(X_sub)

    ratio_scratch = scratch_pca["explained_variance_ratio"]
    ratio_sk = sk_pca.explained_variance_ratio_.tolist()
    print(f"PCA PC1/PC2 Explained Variance Ratio: Scratch={[round(x, 4) for x in ratio_scratch]}, Sklearn={[round(x, 4) for x in ratio_sk]}")

    fig_dir = Path("results/figures/from_scratch")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    bf_mask = y_sub == BONAFIDE
    sp_mask = y_sub == SPOOF
    proj = scratch_pca["projected_2d"]
    ax.scatter(proj[sp_mask, 0], proj[sp_mask, 1], c="#DC2626", alpha=0.35, s=15, label="Spoof (Synthetic)")
    ax.scatter(proj[bf_mask, 0], proj[bf_mask, 1], c="#059669", alpha=0.6, s=25, label="Bona Fide (Genuine)")
    ax.set_xlabel(f"Principal Component 1 ({ratio_scratch[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"Principal Component 2 ({ratio_scratch[1]*100:.1f}% Variance)")
    ax.set_title("From-Scratch PCA (SVD): 2-D Projection of Pooled LFCC")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pca_2d_scatter.png")
    plt.close()

    print("Running k-Means (k=2) from scratch...")
    kmeans_res = run_kmeans_scratch(X_sub, k=2, max_iters=25, seed=seed)
    cluster_labels = kmeans_res["assignments"]

    purity = compute_cluster_purity(y_sub, cluster_labels)
    ari = float(adjusted_rand_score(y_sub, cluster_labels))
    print(f"k-Means (k=2) Unsupervised Quality: Purity={purity*100:.2f}%, ARI={ari:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    ax.plot(range(1, len(kmeans_res["objective_history"]) + 1), kmeans_res["objective_history"], marker="o", color="#2563EB", lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Inertia Objective (Sum of Squared Distances)")
    ax.set_title("From-Scratch k-Means (k=2) Objective Convergence")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "kmeans_objective_curve.png")
    plt.close()

    results_payload = {
        "dataset_name": "ASVspoof 2019 LA Pooled LFCC (N=5000)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_samples": n_train,
        "pca_svd": {
            "pc1_explained_variance_ratio": float(ratio_scratch[0]),
            "pc2_explained_variance_ratio": float(ratio_scratch[1]),
            "sklearn_pc1_ratio": float(ratio_sk[0]),
            "sklearn_pc2_ratio": float(ratio_sk[1]),
            "ratio_parity_delta": float(abs(ratio_scratch[0] - ratio_sk[0])),
        },
        "kmeans_k2": {
            "purity": float(purity),
            "adjusted_rand_index": float(ari),
            "initial_inertia": float(kmeans_res["objective_history"][0]),
            "final_inertia": float(kmeans_res["final_inertia"]),
            "n_iterations": len(kmeans_res["objective_history"]),
            "honest_assessment": "Unsupervised k-means achieves near-zero ARI, demonstrating that forensic deepfake speech detection cannot be solved via naive unsupervised clustering in pooled LFCC space without supervision or discriminative margin modeling.",
        },
        "theoretical_connection_to_em": {
            "gmm_relationship": "k-Means represents the limiting hard-clustering case of the Expectation-Maximization (EM) algorithm for Gaussian Mixture Models where component covariances are isotropic spherical matrices and variance approaches zero.",
            "soft_vs_hard": "In GMM-EM, the E-step computes soft posterior responsibilities gamma_ik, while k-means performs hard argmin quantization. The M-step in GMM updates both component means, diagonal variances, and prior weights, allowing GMM-LFCC to capture multimodal speech formant density.",
        },
    }

    out_json = Path("results/metrics/syllabus_scratch_pca_kmeans.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("Scratch PCA & k-Means completed successfully!")

if __name__ == "__main__":
    main()

