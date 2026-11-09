from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_min_tdcf,
    compute_metrics_at_threshold,
    bootstrap_confidence_interval,
)

def evaluate_classifier_full(model, X_dev, y_dev, X_eval, y_eval, dev_asv, eval_asv, model_name: str) -> dict:
    if hasattr(model, "predict_proba"):
        s_dev = model.predict_proba(X_dev)[:, BONAFIDE]
        s_eval = model.predict_proba(X_eval)[:, BONAFIDE]
    elif hasattr(model, "decision_function"):
        s_dev = model.decision_function(X_dev)
        s_eval = model.decision_function(X_eval)
    else:
        s_dev = model.predict(X_dev).astype(float)
        s_eval = model.predict(X_eval).astype(float)

    dev_eer, dev_th = compute_eer(y_dev, s_dev)
    dev_auc = compute_roc_auc(y_dev, s_dev)
    dev_tdcf, _ = compute_min_tdcf(y_dev, s_dev, asv_score_file=dev_asv)

    eval_eer, eval_opt_th = compute_eer(y_eval, s_eval)
    eval_auc = compute_roc_auc(y_eval, s_eval)
    eval_tdcf, _ = compute_min_tdcf(y_eval, s_eval, asv_score_file=eval_asv)

    eval_eer_ci = bootstrap_confidence_interval(lambda l, s: compute_eer(l, s)[0], y_eval, s_eval, n_bootstraps=500, seed=42)

    at_dev_th = compute_metrics_at_threshold(y_eval, s_eval, dev_th)
    preds = (s_eval >= dev_th).astype(int)
    cm = confusion_matrix(y_eval, preds, labels=[BONAFIDE, SPOOF]).tolist()

    return {
        "model_name": model_name,
        "dev_eer": float(dev_eer),
        "dev_threshold": float(dev_th),
        "dev_min_tdcf": float(dev_tdcf),
        "dev_roc_auc": float(dev_auc),
        "eval_eer": float(eval_eer),
        "eval_eer_ci_95": [float(eval_eer_ci[0]), float(eval_eer_ci[1])],
        "eval_min_tdcf": float(eval_tdcf),
        "eval_roc_auc": float(eval_auc),
        "metrics_at_dev_threshold": {
            "accuracy": float(at_dev_th["accuracy"]),
            "precision": float(at_dev_th["precision"]),
            "recall": float(at_dev_th["recall"]),
            "f1": float(at_dev_th["f1"]),
        },
        "confusion_matrix_at_dev_th": cm,
    }

def main(seed: int = 42):
    print("Loading cached pooled LFCC features for syllabus baseline models...")
    d_tr = np.load("data/cache/lfcc_train.npz")
    d_dv = np.load("data/cache/lfcc_dev.npz")
    d_ev = np.load("data/cache/lfcc_eval.npz")

    tr_pooled = d_tr["pooled"]
    tr_labels = d_tr["labels"]
    dv_pooled = d_dv["pooled"]
    dv_labels = d_dv["labels"]
    ev_pooled = d_ev["pooled"]
    ev_labels = d_ev["labels"]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(tr_pooled)
    X_dv = scaler.transform(dv_pooled)
    X_ev = scaler.transform(ev_pooled)

    dev_asv = "data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
    eval_asv = "data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"

    results = {}

    print("1. Evaluating Gaussian Naive Bayes...")
    gnb = GaussianNB()
    gnb.fit(X_tr, tr_labels)
    results["GaussianNB"] = evaluate_classifier_full(gnb, X_dv, dv_labels, X_ev, ev_labels, dev_asv, eval_asv, "Gaussian Naive Bayes")
    print(f"GaussianNB: Dev EER={results['GaussianNB']['dev_eer']:.2f}%, Eval EER={results['GaussianNB']['eval_eer']:.2f}%")

    print("2. Tuning k-NN (k in [1, 3, 5, 9, 15, 25]) on Dev split...")
    k_candidates = [1, 3, 5, 9, 15, 25]
    best_k = 5
    best_k_dev_eer = float("inf")
    knn_dev_eers = []
    sub_knn = 10000
    rng = np.random.default_rng(seed)
    sub_knn_idx = rng.choice(len(tr_labels), min(sub_knn, len(tr_labels)), replace=False)

    for k in k_candidates:
        knn_temp = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn_temp.fit(X_tr[sub_knn_idx], tr_labels[sub_knn_idx])
        s_dv = knn_temp.predict_proba(X_dv)[:, BONAFIDE]
        e_dv, _ = compute_eer(dv_labels, s_dv)
        knn_dev_eers.append(float(e_dv))
        print(f"k={k}: Dev EER={e_dv:.2f}%")
        if e_dv < best_k_dev_eer:
            best_k_dev_eer = e_dv
            best_k = k

    print(f"Optimal k selected by Dev EER: k={best_k} (Dev EER: {best_k_dev_eer:.2f}%)")
    best_knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    best_knn.fit(X_tr[sub_knn_idx], tr_labels[sub_knn_idx])
    results[f"k-NN (k={best_k})"] = evaluate_classifier_full(best_knn, X_dv, dv_labels, X_ev, ev_labels, dev_asv, eval_asv, f"k-Nearest Neighbors (k={best_k})")

    print("3. Decision Tree: Depth Sweep and Cost-Complexity Pruning...")
    depths = [2, 4, 6, 8, 10, 15, 20, 30]
    dt_train_eers = []
    dt_dev_eers = []
    best_depth = 6
    best_depth_dev_eer = float("inf")

    for d in depths:
        dt_temp = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=seed, class_weight="balanced")
        dt_temp.fit(X_tr, tr_labels)
        s_tr = dt_temp.predict_proba(X_tr)[:, BONAFIDE]
        s_dv = dt_temp.predict_proba(X_dv)[:, BONAFIDE]
        e_tr, _ = compute_eer(tr_labels, s_tr)
        e_dv, _ = compute_eer(dv_labels, s_dv)
        dt_train_eers.append(float(e_tr))
        dt_dev_eers.append(float(e_dv))
        if e_dv < best_depth_dev_eer:
            best_depth_dev_eer = e_dv
            best_depth = d

    print(f"Optimal Decision Tree Depth by Dev EER: depth={best_depth} (Dev EER: {best_depth_dev_eer:.2f}%)")
    best_dt = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth, random_state=seed, class_weight="balanced")
    best_dt.fit(X_tr, tr_labels)
    results[f"DecisionTree (depth={best_depth})"] = evaluate_classifier_full(best_dt, X_dv, dv_labels, X_ev, ev_labels, dev_asv, eval_asv, f"Decision Tree (Entropy, depth={best_depth})")

    fig_dir = Path("results/figures/syllabus")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.plot(depths, dt_train_eers, marker="o", label="Train EER (Memorization)", color="#2563EB", lw=2)
    ax.plot(depths, dt_dev_eers, marker="s", label="Dev EER (Generalization)", color="#DC2626", lw=2)
    ax.scatter(best_depth, best_depth_dev_eer, color="#D97706", s=100, zorder=5, label=f"Optimal Depth={best_depth}")
    ax.set_xlabel("Decision Tree Maximum Depth")
    ax.set_ylabel("EER (%)")
    ax.set_title("Bias-Variance Tradeoff: Decision Tree Depth Overfitting")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "bias_variance_tree_depth.png")
    plt.close()

    print("4. Evaluating SVM C Overfitting Curve...")
    c_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    svm_train_eers = []
    svm_dev_eers = []
    sub_svm = 5000
    sub_svm_idx = rng.choice(len(tr_labels), min(sub_svm, len(tr_labels)), replace=False)

    for c in c_list:
        svm_temp = SVC(C=c, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
        svm_temp.fit(X_tr[sub_svm_idx], tr_labels[sub_svm_idx])
        s_tr = svm_temp.predict_proba(X_tr[sub_svm_idx])[:, BONAFIDE]
        s_dv = svm_temp.predict_proba(X_dv)[:, BONAFIDE]
        e_tr, _ = compute_eer(tr_labels[sub_svm_idx], s_tr)
        e_dv, _ = compute_eer(dv_labels, s_dv)
        svm_train_eers.append(float(e_tr))
        svm_dev_eers.append(float(e_dv))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.plot(c_list, svm_train_eers, marker="o", label="Train EER (N=5000)", color="#2563EB", lw=2)
    ax.plot(c_list, svm_dev_eers, marker="s", label="Dev EER", color="#DC2626", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("SVM Inverse Regularization Parameter C")
    ax.set_ylabel("EER (%)")
    ax.set_title("Bias-Variance Tradeoff: SVM Margin vs Overfitting")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "bias_variance_svm_c.png")
    plt.close()

    print("5. Evaluating AdaBoost Classifier...")
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=seed)
    adaboost.fit(X_tr, tr_labels)
    results["AdaBoost"] = evaluate_classifier_full(adaboost, X_dv, dv_labels, X_ev, ev_labels, dev_asv, eval_asv, "AdaBoost (100 Estimators)")
    print(f"AdaBoost: Dev EER={results['AdaBoost']['dev_eer']:.2f}%, Eval EER={results['AdaBoost']['eval_eer']:.2f}%")

    out_json = Path("results/metrics/syllabus_models_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    table_md_path = Path("results/tables/table_syllabus_models.md")
    table_tex_path = Path("results/tables/table_syllabus_models.tex")
    
    md_lines = [
        "| Model (Syllabus Architecture) | Dev EER (%) | Eval EER (%) [95% CI] | min-tDCF | ROC-AUC | Dev Thresh Acc (%) | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tex_lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Model & Dev EER (\\%) & Eval EER (\\%) & min-tDCF & ROC-AUC & Accuracy & Precision & Recall & F1 \\\\",
        "\\midrule",
    ]

    for k, v in results.items():
        m = v["metrics_at_dev_threshold"]
        ci_str = f"[{v['eval_eer_ci_95'][0]:.2f}, {v['eval_eer_ci_95'][1]:.2f}]"
        md_lines.append(f"| {k} | {v['dev_eer']:.2f} | {v['eval_eer']:.2f} {ci_str} | {v['eval_min_tdcf']:.4f} | {v['eval_roc_auc']:.4f} | {m['accuracy']*100:.2f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
        tex_lines.append(f"{k} & {v['dev_eer']:.2f} & {v['eval_eer']:.2f} & {v['eval_min_tdcf']:.4f} & {v['eval_roc_auc']:.4f} & {m['accuracy']*100:.2f}\\% & {m['precision']:.3f} & {m['recall']:.3f} & {m['f1']:.3f} \\\\")

    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])

    table_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    table_tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    print(f"Syllabus models evaluation finished! Results saved to {out_json} and {table_md_path}")

if __name__ == "__main__":
    main()

