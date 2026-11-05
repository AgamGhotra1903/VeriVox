import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_min_tdcf,
    compute_metrics_at_threshold,
    bootstrap_confidence_interval,
)
from acforensics.evaluation.provenance import (
    compute_file_sha256,
    get_git_commit,
    guard_results_and_provenance,
)

def subsample_frames(frames: np.ndarray, max_frames: int, seed: int) -> np.ndarray:
    if len(frames) <= max_frames:
        return frames
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(frames), size=max_frames, replace=False)
    return frames[idx]

def extract_frames_by_class(features: np.ndarray, labels: np.ndarray, target_label: int) -> np.ndarray:
    idx = np.where(labels == target_label)[0]
    sub = features[idx]
    n, c, t = sub.shape
    trans = np.transpose(sub, (0, 2, 1)).reshape(-1, c)
    return np.asarray(trans, dtype=np.float64)

def score_gmm(features: np.ndarray, gmm_bf: GaussianMixture, gmm_sp: GaussianMixture) -> np.ndarray:
    n_utts = len(features)
    scores = np.empty(n_utts, dtype=np.float64)
    for i in range(n_utts):
        frames = np.asarray(features[i].T, dtype=np.float64)
        ll_bf = gmm_bf.score_samples(frames)
        ll_sp = gmm_sp.score_samples(frames)
        scores[i] = float(np.mean(ll_bf - ll_sp))
    return scores

def stratified_subsample(x: np.ndarray, y: np.ndarray, max_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_n:
        return x, y
    rng = np.random.RandomState(seed)
    bf_idx = np.where(y == BONAFIDE)[0]
    sp_idx = np.where(y == SPOOF)[0]
    n_bf = int(max_n * (len(bf_idx) / len(y)))
    n_sp = max_n - n_bf
    chosen_bf = rng.choice(bf_idx, size=min(n_bf, len(bf_idx)), replace=False)
    chosen_sp = rng.choice(sp_idx, size=min(n_sp, len(sp_idx)), replace=False)
    all_chosen = np.concatenate([chosen_bf, chosen_sp])
    rng.shuffle(all_chosen)
    return x[all_chosen], y[all_chosen]

def evaluate_model_on_splits(
    model_name: str,
    feat_name: str,
    dev_scores: np.ndarray,
    dev_labels: np.ndarray,
    eval_scores: np.ndarray,
    eval_labels: np.ndarray,
    dev_asv: str,
    eval_asv: str,
    best_params: dict,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    dev_eer, dev_th = compute_eer(dev_labels, dev_scores)
    dev_auc = compute_roc_auc(dev_labels, dev_scores)
    dev_tdcf, _ = compute_min_tdcf(dev_labels, dev_scores, asv_score_file=dev_asv)
    dev_m_at_th = compute_metrics_at_threshold(dev_labels, dev_scores, dev_th)

    dev_eer_ci = bootstrap_confidence_interval(lambda l, s: compute_eer(l, s)[0], dev_labels, dev_scores, n_bootstraps=n_bootstraps, seed=seed)
    dev_tdcf_ci = bootstrap_confidence_interval(lambda l, s: compute_min_tdcf(l, s, asv_score_file=dev_asv)[0], dev_labels, dev_scores, n_bootstraps=n_bootstraps, seed=seed)

    eval_eer, eval_opt_th = compute_eer(eval_labels, eval_scores)
    eval_auc = compute_roc_auc(eval_labels, eval_scores)
    eval_tdcf, _ = compute_min_tdcf(eval_labels, eval_scores, asv_score_file=eval_asv)
    eval_m_dev_th = compute_metrics_at_threshold(eval_labels, eval_scores, dev_th)
    eval_m_opt_th = compute_metrics_at_threshold(eval_labels, eval_scores, eval_opt_th)

    eval_eer_ci = bootstrap_confidence_interval(lambda l, s: compute_eer(l, s)[0], eval_labels, eval_scores, n_bootstraps=n_bootstraps, seed=seed)
    eval_tdcf_ci = bootstrap_confidence_interval(lambda l, s: compute_min_tdcf(l, s, asv_score_file=eval_asv)[0], eval_labels, eval_scores, n_bootstraps=n_bootstraps, seed=seed)

    return {
        "model": model_name,
        "features": feat_name,
        "best_params": best_params,
        "dev": {
            "eer": float(dev_eer),
            "eer_threshold": float(dev_th),
            "roc_auc": float(dev_auc),
            "min_tdcf": float(dev_tdcf),
            "eer_ci_95": [float(dev_eer_ci[0]), float(dev_eer_ci[1])],
            "min_tdcf_ci_95": [float(dev_tdcf_ci[0]), float(dev_tdcf_ci[1])],
            "metrics_at_threshold": dev_m_at_th,
        },
        "eval": {
            "eer": float(eval_eer),
            "eer_optimal_threshold": float(eval_opt_th),
            "applied_threshold": float(dev_th),
            "roc_auc": float(eval_auc),
            "min_tdcf": float(eval_tdcf),
            "eer_ci_95": [float(eval_eer_ci[0]), float(eval_eer_ci[1])],
            "min_tdcf_ci_95": [float(eval_tdcf_ci[0]), float(eval_tdcf_ci[1])],
            "metrics_at_dev_threshold": eval_m_dev_th,
            "metrics_at_optimal_threshold": eval_m_opt_th,
        },
    }

def run_classical_sweep(
    dev_asv: str,
    eval_asv: str,
    out_json: str,
    seed: int = 42,
    n_bootstraps: int = 1000,
) -> dict:
    all_results = {}
    baselines_payload_models = {}

    for feat_type in ["lfcc", "logmel"]:
        tr_npz = np.load(f"data/cache/{feat_type}_train.npz")
        dv_npz = np.load(f"data/cache/{feat_type}_dev.npz")
        ev_npz = np.load(f"data/cache/{feat_type}_eval.npz")

        tr_labels = tr_npz["labels"]
        dv_labels = dv_npz["labels"]
        ev_labels = ev_npz["labels"]

        tr_features = tr_npz["features"]
        dv_features = dv_npz["features"]
        ev_features = ev_npz["features"]

        tr_pooled = tr_npz["pooled"]
        dv_pooled = dv_npz["pooled"]
        ev_pooled = ev_npz["pooled"]

        print(f"[{feat_type.upper()}] Training GMM (K=64)...")
        bf_f = subsample_frames(extract_frames_by_class(tr_features, tr_labels, BONAFIDE), 250000, seed)
        sp_f = subsample_frames(extract_frames_by_class(tr_features, tr_labels, SPOOF), 250000, seed + 1)
        gmm_bf = GaussianMixture(n_components=64, covariance_type="diag", reg_covar=1e-3, max_iter=100, random_state=seed)
        gmm_sp = GaussianMixture(n_components=64, covariance_type="diag", reg_covar=1e-3, max_iter=100, random_state=seed + 1)
        gmm_bf.fit(bf_f)
        gmm_sp.fit(sp_f)

        gmm_dev_scores = score_gmm(dv_features, gmm_bf, gmm_sp)
        gmm_eval_scores = score_gmm(ev_features, gmm_bf, gmm_sp)
        gmm_res = evaluate_model_on_splits(
            "GMM", feat_type, gmm_dev_scores, dv_labels, gmm_eval_scores, ev_labels,
            dev_asv, eval_asv, {"n_components": 64, "reg_covar": 1e-3}, n_bootstraps, seed,
        )
        all_results[f"GMM_{feat_type}"] = gmm_res

        if "GMM" not in baselines_payload_models:
            baselines_payload_models["GMM"] = {}
        baselines_payload_models["GMM"][feat_type] = {
            "dev": gmm_res["dev"],
            "eval": gmm_res["eval"],
        }

        print(f"[{feat_type.upper()}] Preprocessing pooled features with StandardScaler + PCA(0.95)...")
        scaler = StandardScaler()
        tr_scaled = scaler.fit_transform(tr_pooled)
        dv_scaled = scaler.transform(dv_pooled)
        ev_scaled = scaler.transform(ev_pooled)

        pca = PCA(n_components=0.95, random_state=seed)
        tr_pca = pca.fit_transform(tr_scaled)
        dv_pca = pca.transform(dv_scaled)
        ev_pca = pca.transform(ev_scaled)
        print(f"[{feat_type.upper()}] PCA retained {tr_pca.shape[1]} components from {tr_scaled.shape[1]}")

        print(f"[{feat_type.upper()}] Tuning Logistic Regression on Dev...")
        c_candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        best_lr_auc = -1.0
        best_lr_model = None
        best_lr_c = 1.0
        for c in c_candidates:
            lr = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed)
            lr.fit(tr_pca, tr_labels)
            pred_dev = lr.predict_proba(dv_pca)[:, 1]
            auc_val = compute_roc_auc(dv_labels, pred_dev)
            if auc_val > best_lr_auc:
                best_lr_auc = auc_val
                best_lr_model = lr
                best_lr_c = c

        lr_dev_scores = best_lr_model.predict_proba(dv_pca)[:, 1]
        lr_eval_scores = best_lr_model.predict_proba(ev_pca)[:, 1]
        lr_res = evaluate_model_on_splits(
            "LogisticRegression", feat_type, lr_dev_scores, dv_labels, lr_eval_scores, ev_labels,
            dev_asv, eval_asv, {"C": best_lr_c}, n_bootstraps, seed,
        )
        all_results[f"LogisticRegression_{feat_type}"] = lr_res
        if "LogisticRegression" not in baselines_payload_models:
            baselines_payload_models["LogisticRegression"] = {}
        baselines_payload_models["LogisticRegression"][feat_type] = {
            "dev": lr_res["dev"],
            "eval": lr_res["eval"],
        }

        print(f"[{feat_type.upper()}] Tuning RBF-SVM on Dev (subsample 10k)...")
        sub_x, sub_y = stratified_subsample(tr_pca, tr_labels, max_n=10000, seed=seed)
        svm_c_candidates = [0.1, 1.0, 10.0]
        svm_gamma_candidates = ["scale", "auto", 0.01]
        best_svm_auc = -1.0
        best_svm_model = None
        best_svm_params = {}
        for c in svm_c_candidates:
            for gamma in svm_gamma_candidates:
                svm = SVC(C=c, gamma=gamma, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
                svm.fit(sub_x, sub_y)
                pred_dev = svm.predict_proba(dv_pca)[:, 1]
                auc_val = compute_roc_auc(dv_labels, pred_dev)
                if auc_val > best_svm_auc:
                    best_svm_auc = auc_val
                    best_svm_model = svm
                    best_svm_params = {"C": c, "gamma": gamma}

        svm_dev_scores = best_svm_model.predict_proba(dv_pca)[:, 1]
        svm_eval_scores = best_svm_model.predict_proba(ev_pca)[:, 1]
        svm_res = evaluate_model_on_splits(
            "SVM", feat_type, svm_dev_scores, dv_labels, svm_eval_scores, ev_labels,
            dev_asv, eval_asv, best_svm_params, n_bootstraps, seed,
        )
        all_results[f"SVM_{feat_type}"] = svm_res
        if "SVM" not in baselines_payload_models:
            baselines_payload_models["SVM"] = {}
        baselines_payload_models["SVM"][feat_type] = {
            "dev": svm_res["dev"],
            "eval": svm_res["eval"],
        }

        print(f"[{feat_type.upper()}] Tuning Random Forest on Dev...")
        rf_candidates = [
            (100, 15),
            (200, 15),
            (100, 25),
            (200, None),
        ]
        best_rf_auc = -1.0
        best_rf_model = None
        best_rf_params = {}
        for n_est, max_d in rf_candidates:
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
            rf.fit(tr_scaled, tr_labels)
            pred_dev = rf.predict_proba(dv_scaled)[:, 1]
            auc_val = compute_roc_auc(dv_labels, pred_dev)
            if auc_val > best_rf_auc:
                best_rf_auc = auc_val
                best_rf_model = rf
                best_rf_params = {"n_estimators": n_est, "max_depth": max_d}

        rf_dev_scores = best_rf_model.predict_proba(dv_scaled)[:, 1]
        rf_eval_scores = best_rf_model.predict_proba(ev_scaled)[:, 1]
        rf_res = evaluate_model_on_splits(
            "RandomForest", feat_type, rf_dev_scores, dv_labels, rf_eval_scores, ev_labels,
            dev_asv, eval_asv, best_rf_params, n_bootstraps, seed,
        )
        all_results[f"RandomForest_{feat_type}"] = rf_res
        if "RandomForest" not in baselines_payload_models:
            baselines_payload_models["RandomForest"] = {}
        baselines_payload_models["RandomForest"][feat_type] = {
            "dev": rf_res["dev"],
            "eval": rf_res["eval"],
        }

    manifest_sha = compute_file_sha256("data/manifests/train.parquet")
    git_hash = get_git_commit()
    ts = datetime.now(timezone.utc).isoformat()

    full_payload = {
        "dataset_name": "ASVspoof 2019 LA",
        "manifest_sha256": manifest_sha,
        "git_commit": git_hash,
        "seed": seed,
        "timestamp": ts,
        "sweep_results": all_results,
    }

    guard_results_and_provenance(full_payload)

    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    base_payload = {
        "dataset_name": "ASVspoof 2019 LA",
        "manifest_sha256": manifest_sha,
        "git_commit": git_hash,
        "seed": seed,
        "timestamp": ts,
        **baselines_payload_models,
    }
    guard_results_and_provenance(base_payload)
    with open(out_p.parent / "baselines_results.json", "w", encoding="utf-8") as f:
        json.dump(base_payload, f, indent=2)

    return full_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-asv", type=str, default="data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt")
    parser.add_argument("--eval-asv", type=str, default="data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
    parser.add_argument("--out-json", type=str, default="results/metrics/classical_sweep.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    res = run_classical_sweep(
        dev_asv=args.dev_asv,
        eval_asv=args.eval_asv,
        out_json=args.out_json,
        seed=args.seed,
        n_bootstraps=args.bootstraps,
    )
    print("Classical sweep completed successfully.")

