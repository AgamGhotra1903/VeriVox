"""Fit Stacker CLI: Meta-Learner Training strictly on DEV scores.

Features:
- Standardizes raw base model scores via StandardScaler fitted on DEV only.
- Fits L2-regularized Logistic Regression meta-learner with CV on DEV only.
- Compares Stacked Ensemble against simple baseline fusion (mean of z-scores).
- Evaluates on EVAL split with everything frozen (zero leakage).
- Outputs meta-learner coefficient table to results/tables/.
- Saves fitted StackedEnsemble pipeline to checkpoints/ensemble/stacked_ensemble.joblib.

Author: Agam Ghotra (IIT2024226)
"""

import argparse
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd

from acforensics.constants import DEFAULT_RESULTS_DIR, DEFAULT_CHECKPOINTS_DIR
from acforensics.ensemble.scores import load_score_matrix
from acforensics.ensemble.stacking import StackedEnsemble
from acforensics.evaluation.metrics import full_evaluation
from acforensics.evaluation.plots import plot_roc_curve, plot_confusion_matrix
from acforensics.utils.io import save_json
from acforensics.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fit Stacked Ensemble Meta-Learner")
    parser.add_argument("--scores-dir", type=str, default="results/scores", help="Directory containing score matrices")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke mode")
    args = parser.parse_args()

    setup_logging()

    scores_dir = Path(args.scores_dir)
    assert (scores_dir / "dev_scores.npy").exists(), f"Dev scores missing at {scores_dir / 'dev_scores.npy'}. Run scripts/build_scores.py first."
    assert (scores_dir / "eval_scores.npy").exists(), f"Eval scores missing at {scores_dir / 'eval_scores.npy'}. Run scripts/build_scores.py first."

    # 1. Load DEV and EVAL score matrices
    dev_scores, dev_labels, dev_ids, model_names = load_score_matrix(scores_dir, "dev")
    eval_scores, eval_labels, eval_ids, _ = load_score_matrix(scores_dir, "eval")

    # Strict leakage assertion: disjoint IDs
    assert len(set(dev_ids).intersection(set(eval_ids))) == 0, "CRITICAL: dev_ids and eval_ids overlap!"

    # 2. Fit Stacked Ensemble strictly on DEV
    logger.info("Fitting StackedEnsemble on DEV scores only (zero leakage into eval)...")
    stacker = StackedEnsemble(model_names=model_names, seed=args.seed)
    stacker.fit(dev_scores, dev_labels)

    ckpt_dir = Path("checkpoints/ensemble")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stacker.save(ckpt_dir / "stacked_ensemble.joblib")

    # 3. Save Meta-Learner Weights Table
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    weights_df = stacker.get_coefficients_table()
    weights_df.to_csv(tables_dir / "meta_learner_weights.csv", index=False)
    weights_df.to_markdown(tables_dir / "meta_learner_weights.md", index=False)
    logger.info("Saved meta-learner weights to %s", tables_dir / "meta_learner_weights.csv")

    # 4. Dev Evaluation to determine operating thresholds
    # A. Stacked Ensemble
    dev_fused_scores = stacker.predict_score(dev_scores)
    dev_stacker_report = full_evaluation(dev_labels, dev_fused_scores, compute_ci=False)
    dev_stacker_thresh = dev_stacker_report["eer_threshold"]

    # B. Mean Baseline Fusion
    dev_mean_scores = stacker.predict_baseline_mean(dev_scores)
    dev_mean_report = full_evaluation(dev_labels, dev_mean_scores, compute_ci=False)
    dev_mean_thresh = dev_mean_report["eer_threshold"]

    # 5. Eval Evaluation strictly using Dev Thresholds
    asv_eval_file = Path("data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
    asv_path_str = str(asv_eval_file) if asv_eval_file.exists() else None

    # Base models single evaluation on Eval
    summary_rows = []
    roc_dict = {}

    for idx, m_name in enumerate(model_names):
        m_dev_scores = dev_scores[:, idx]
        m_dev_rep = full_evaluation(dev_labels, m_dev_scores, compute_ci=False)
        m_dev_th = m_dev_rep["eer_threshold"]

        m_eval_scores = eval_scores[:, idx]
        m_eval_rep = full_evaluation(eval_labels, m_eval_scores, dev_threshold=m_dev_th, asv_score_file=asv_path_str)
        roc_dict[m_name] = (eval_labels, m_eval_scores)

        summary_rows.append({
            "System": f"Single: {m_name}",
            "Type": "Base Model",
            "Dev EER (%)": f"{m_dev_rep['eer']:.2f}",
            "Eval EER (%)": f"{m_eval_rep['eer']:.2f}",
            "Eval ROC-AUC": f"{m_eval_rep['roc_auc']:.4f}",
            "Eval min-tDCF": f"{m_eval_rep['min_tdcf']:.4f}",
            "Applied Threshold": f"{m_dev_th:.4f} (Dev)",
        })

    # Baseline Mean Fusion evaluation on Eval
    eval_mean_scores = stacker.predict_baseline_mean(eval_scores)
    eval_mean_rep = full_evaluation(eval_labels, eval_mean_scores, dev_threshold=dev_mean_thresh, asv_score_file=asv_path_str)
    roc_dict["Mean Fusion (Z-Score)"] = (eval_labels, eval_mean_scores)

    summary_rows.append({
        "System": "Baseline Fusion (Mean Z-Scores)",
        "Type": "Unweighted Average",
        "Dev EER (%)": f"{dev_mean_report['eer']:.2f}",
        "Eval EER (%)": f"{eval_mean_rep['eer']:.2f}",
        "Eval ROC-AUC": f"{eval_mean_rep['roc_auc']:.4f}",
        "Eval min-tDCF": f"{eval_mean_rep['min_tdcf']:.4f}",
        "Applied Threshold": f"{dev_mean_thresh:.4f} (Dev)",
    })

    # Stacked Ensemble evaluation on Eval
    eval_fused_scores = stacker.predict_score(eval_scores)
    eval_stacker_rep = full_evaluation(eval_labels, eval_fused_scores, dev_threshold=dev_stacker_thresh, asv_score_file=asv_path_str)
    roc_dict["Stacked Ensemble (Proposed)"] = (eval_labels, eval_fused_scores)

    summary_rows.append({
        "System": "Stacked Ensemble (Proposed)",
        "Type": "L2 Logistic Regression",
        "Dev EER (%)": f"{dev_stacker_report['eer']:.2f}",
        "Eval EER (%)": f"{eval_stacker_rep['eer']:.2f}",
        "Eval ROC-AUC": f"{eval_stacker_rep['roc_auc']:.4f}",
        "Eval min-tDCF": f"{eval_stacker_rep['min_tdcf']:.4f}",
        "Applied Threshold": f"{dev_stacker_thresh:.4f} (Dev)",
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / "ensemble_comparison.csv", index=False)
    summary_df.to_markdown(tables_dir / "ensemble_comparison.md", index=False)

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(roc_dict, figures_dir / "ensemble_roc_curves.png", title="Base Models vs Fusions on Eval Split")

    # Confusion matrix of Stacked Ensemble at EER threshold
    cm_eval = np.array([
        [eval_stacker_rep["at_applied_threshold"]["tn"], eval_stacker_rep["at_applied_threshold"]["fp"]],
        [eval_stacker_rep["at_applied_threshold"]["fn"], eval_stacker_rep["at_applied_threshold"]["tp"]],
    ])
    plot_confusion_matrix(cm_eval, figures_dir / "stacked_ensemble_confusion_matrix.png", title="Stacked Ensemble Confusion Matrix")

    # Print summary tables
    print("\n" + "=" * 95)
    print("META-LEARNER BASE MODEL COEFFICIENTS (STACKED ENSEMBLE)")
    print("=" * 95)
    print(weights_df.to_string(index=False))
    print("=" * 95)

    print("\n" + "=" * 95)
    print("BASE MODELS VS. FUSIONS PERFORMANCE COMPARISON (ASVspoof 2019 LA)")
    print("=" * 95)
    print(summary_df.to_string(index=False))
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
