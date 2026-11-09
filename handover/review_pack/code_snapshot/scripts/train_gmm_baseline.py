import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture

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

def score_all(features: np.ndarray, gmm_bf: GaussianMixture, gmm_sp: GaussianMixture) -> np.ndarray:
    n_utts = len(features)
    scores = np.empty(n_utts, dtype=np.float64)
    for i in range(n_utts):
        frames = np.asarray(features[i].T, dtype=np.float64)
        ll_bf = gmm_bf.score_samples(frames)
        ll_sp = gmm_sp.score_samples(frames)
        scores[i] = float(np.mean(ll_bf - ll_sp))
    return scores

def run_pipeline(
    train_cache: str,
    dev_cache: str,
    eval_cache: str,
    dev_asv: str,
    eval_asv: str,
    out_json: str,
    n_components: int = 64,
    max_frames_per_class: int = 250000,
    seed: int = 42,
    n_bootstraps: int = 1000,
) -> dict:
    train_npz = np.load(train_cache)
    tr_feat = train_npz["features"]
    tr_lab = train_npz["labels"]

    bf_frames = extract_frames_by_class(tr_feat, tr_lab, BONAFIDE)
    sp_frames = extract_frames_by_class(tr_feat, tr_lab, SPOOF)

    bf_sub = subsample_frames(bf_frames, max_frames_per_class, seed)
    sp_sub = subsample_frames(sp_frames, max_frames_per_class, seed + 1)

    gmm_bf = GaussianMixture(n_components=n_components, covariance_type="diag", reg_covar=1e-3, max_iter=100, random_state=seed)
    gmm_sp = GaussianMixture(n_components=n_components, covariance_type="diag", reg_covar=1e-3, max_iter=100, random_state=seed + 1)

    gmm_bf.fit(bf_sub)
    gmm_sp.fit(sp_sub)

    dev_npz = np.load(dev_cache)
    dev_feat = dev_npz["features"]
    dev_lab = dev_npz["labels"]
    dev_scores = score_all(dev_feat, gmm_bf, gmm_sp)

    dev_eer, dev_th = compute_eer(dev_lab, dev_scores)
    dev_auc = compute_roc_auc(dev_lab, dev_scores)
    dev_tdcf, _ = compute_min_tdcf(dev_lab, dev_scores, asv_score_file=dev_asv)
    dev_metrics_at_th = compute_metrics_at_threshold(dev_lab, dev_scores, dev_th)

    dev_eer_ci = bootstrap_confidence_interval(lambda l, s: compute_eer(l, s)[0], dev_lab, dev_scores, n_bootstraps=n_bootstraps, seed=seed)
    dev_tdcf_ci = bootstrap_confidence_interval(lambda l, s: compute_min_tdcf(l, s, asv_score_file=dev_asv)[0], dev_lab, dev_scores, n_bootstraps=n_bootstraps, seed=seed)

    eval_npz = np.load(eval_cache)
    eval_feat = eval_npz["features"]
    eval_lab = eval_npz["labels"]
    eval_scores = score_all(eval_feat, gmm_bf, gmm_sp)

    eval_eer, eval_opt_th = compute_eer(eval_lab, eval_scores)
    eval_auc = compute_roc_auc(eval_lab, eval_scores)
    eval_tdcf, _ = compute_min_tdcf(eval_lab, eval_scores, asv_score_file=eval_asv)
    eval_metrics_at_dev_th = compute_metrics_at_threshold(eval_lab, eval_scores, dev_th)
    eval_metrics_at_opt_th = compute_metrics_at_threshold(eval_lab, eval_scores, eval_opt_th)

    eval_eer_ci = bootstrap_confidence_interval(lambda l, s: compute_eer(l, s)[0], eval_lab, eval_scores, n_bootstraps=n_bootstraps, seed=seed)
    eval_tdcf_ci = bootstrap_confidence_interval(lambda l, s: compute_min_tdcf(l, s, asv_score_file=eval_asv)[0], eval_lab, eval_scores, n_bootstraps=n_bootstraps, seed=seed)

    manifest_sha = compute_file_sha256("data/manifests/train.parquet")
    git_hash = get_git_commit()
    ts = datetime.now(timezone.utc).isoformat()

    results = {
        "dataset_name": "ASVspoof 2019 LA",
        "manifest_sha256": manifest_sha,
        "git_commit": git_hash,
        "seed": seed,
        "timestamp": ts,
        "model": "LFCC-GMM",
        "n_components": n_components,
        "covariance_type": "diag",
        "max_frames_per_class": max_frames_per_class,
        "dev": {
            "num_trials": len(dev_lab),
            "eer": float(dev_eer),
            "eer_threshold": float(dev_th),
            "roc_auc": float(dev_auc),
            "min_tdcf": float(dev_tdcf),
            "eer_ci_95": [float(dev_eer_ci[0]), float(dev_eer_ci[1])],
            "min_tdcf_ci_95": [float(dev_tdcf_ci[0]), float(dev_tdcf_ci[1])],
            "metrics_at_threshold": dev_metrics_at_th,
        },
        "eval": {
            "num_trials": len(eval_lab),
            "eer": float(eval_eer),
            "eer_optimal_threshold": float(eval_opt_th),
            "operating_threshold_from_dev": float(dev_th),
            "roc_auc": float(eval_auc),
            "min_tdcf": float(eval_tdcf),
            "eer_ci_95": [float(eval_eer_ci[0]), float(eval_eer_ci[1])],
            "min_tdcf_ci_95": [float(eval_tdcf_ci[0]), float(eval_tdcf_ci[1])],
            "metrics_at_dev_threshold": eval_metrics_at_dev_th,
            "metrics_at_optimal_threshold": eval_metrics_at_opt_th,
        },
    }

    guard_results_and_provenance(results)

    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    baselines_payload = {
        "dataset_name": "ASVspoof 2019 LA",
        "manifest_sha256": manifest_sha,
        "git_commit": git_hash,
        "seed": seed,
        "timestamp": ts,
        "GMM": {
            "lfcc": {
                "dev": {
                    "eer": float(dev_eer),
                    "eer_threshold": float(dev_th),
                    "roc_auc": float(dev_auc),
                    "min_tdcf": float(dev_tdcf),
                },
                "eval": {
                    "eer": float(eval_eer),
                    "roc_auc": float(eval_auc),
                    "min_tdcf": float(eval_tdcf),
                    "applied_threshold": float(dev_th),
                },
            }
        },
    }
    with open(out_p.parent / "baselines_results.json", "w", encoding="utf-8") as f:
        json.dump(baselines_payload, f, indent=2)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=str, default="data/cache/lfcc_train.npz")
    parser.add_argument("--dev-cache", type=str, default="data/cache/lfcc_dev.npz")
    parser.add_argument("--eval-cache", type=str, default="data/cache/lfcc_eval.npz")
    parser.add_argument("--dev-asv", type=str, default="data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt")
    parser.add_argument("--eval-asv", type=str, default="data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
    parser.add_argument("--out-json", type=str, default="results/metrics/gmm_baseline.json")
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    res = run_pipeline(
        train_cache=args.train_cache,
        dev_cache=args.dev_cache,
        eval_cache=args.eval_cache,
        dev_asv=args.dev_asv,
        eval_asv=args.eval_asv,
        out_json=args.out_json,
        n_components=args.n_components,
        seed=args.seed,
        n_bootstraps=args.bootstraps,
    )
    print("Baseline evaluation completed successfully.")
    print("Dev EER:", res["dev"]["eer"], "Dev min-tDCF:", res["dev"]["min_tdcf"])
    print("Eval EER:", res["eval"]["eer"], "Eval min-tDCF:", res["eval"]["min_tdcf"])
