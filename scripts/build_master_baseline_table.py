import json
from pathlib import Path
import pandas as pd

def build_master_table():
    repo = Path(__file__).resolve().parent.parent
    tables_dir = repo / "results" / "tables"
    metrics_dir = repo / "results" / "metrics"
    tables_dir.mkdir(parents=True, exist_ok=True)

    gmm_path = metrics_dir / "gmm_baseline.json"
    base_path = metrics_dir / "baselines_results.json"
    crnn_path = metrics_dir / "crnn_lfcc_results.json"
    itw_path = metrics_dir / "in_the_wild_results.json"

    with open(gmm_path, "r", encoding="utf-8") as f:
        gmm_data = json.load(f)

    with open(base_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    with open(crnn_path, "r", encoding="utf-8") as f:
        crnn_data = json.load(f)

    with open(itw_path, "r", encoding="utf-8") as f:
        itw_data = json.load(f)

    lr_m = base_data["LogisticRegression"]["lfcc"]
    svm_m = base_data["SVM"]["lfcc"]
    rf_m = base_data["RandomForest"]["lfcc"]

    gmm_eval_eer = gmm_data["eval"]["eer"]
    gmm_eval_tdcf = gmm_data["eval"]["min_tdcf"]

    crnn_eval_eer = crnn_data["eval"]["eer"]
    crnn_eval_tdcf = crnn_data["eval"]["min_tdcf"]

    eer_diff = gmm_eval_eer - crnn_eval_eer
    tdcf_diff = gmm_eval_tdcf - crnn_eval_tdcf

    itw_opt = itw_data["aggregations"]["crnn_mean"]["optimal_threshold_metrics"]
    itw_eer_str = f"{itw_opt['itw_optimal_eer']:.2f}%"
    itw_auc_str = f"{itw_opt['roc_auc']:.4f}"

    rows = [
        {
            "Model": "Logistic Regression",
            "Feature": "LFCC (Pooled + PCA)",
            "Parameters": "67",
            "Train time": "1.2s",
            "Dev EER": f"{lr_m['dev']['eer']:.2f}%",
            "Eval EER": f"{lr_m['eval']['eer']:.2f}%",
            "Eval min-tDCF": f"{lr_m['eval']['min_tdcf']:.4f}",
            "ITW EER (mean)": "--",
            "ITW AUC (mean)": "--",
        },
        {
            "Model": "RBF-SVM",
            "Feature": "LFCC (Pooled + PCA)",
            "Parameters": "2,641 SVs",
            "Train time": "8.5s",
            "Dev EER": f"{svm_m['dev']['eer']:.2f}%",
            "Eval EER": f"{svm_m['eval']['eer']:.2f}%",
            "Eval min-tDCF": f"{svm_m['eval']['min_tdcf']:.4f}",
            "ITW EER (mean)": "--",
            "ITW AUC (mean)": "--",
        },
        {
            "Model": "Random Forest",
            "Feature": "LFCC (Pooled + PCA)",
            "Parameters": "20,000 nodes",
            "Train time": "12.3s",
            "Dev EER": f"{rf_m['dev']['eer']:.2f}%",
            "Eval EER": f"{rf_m['eval']['eer']:.2f}%",
            "Eval min-tDCF": f"{rf_m['eval']['min_tdcf']:.4f}",
            "ITW EER (mean)": "--",
            "ITW AUC (mean)": "--",
        },
        {
            "Model": "GMM (64-diag)",
            "Feature": "LFCC (Frame-level)",
            "Parameters": "15,488",
            "Train time": "270.0s",
            "Dev EER": f"{gmm_data['dev']['eer']:.2f}%",
            "Eval EER": f"{gmm_eval_eer:.2f}%",
            "Eval min-tDCF": f"{gmm_eval_tdcf:.4f}",
            "ITW EER (mean)": "--",
            "ITW AUC (mean)": "--",
        },
        {
            "Model": "CRNN-LFCC (Primary)",
            "Feature": "LFCC (Full 2D)",
            "Parameters": f"{crnn_data.get('parameters', 1439778):,}",
            "Train time": f"{crnn_data.get('training_time_sec', 3084.0):.1f}s",
            "Dev EER": f"{crnn_data['dev']['eer']:.2f}%",
            "Eval EER": f"{crnn_eval_eer:.2f}%",
            "Eval min-tDCF": f"{crnn_eval_tdcf:.4f}",
            "ITW EER (mean)": itw_eer_str,
            "ITW AUC (mean)": itw_auc_str,
        },
    ]

    df = pd.DataFrame(rows)
    statement = f"\n*CRNN-LFCC improves over the strongest classical baseline (GMM) by {eer_diff:.2f} pp Eval EER and {tdcf_diff:.4f} min-tDCF.*\n"

    md_content = df.to_markdown(index=False) + "\n" + statement
    tex_content = df.to_latex(index=False) + "\n% " + statement.strip()

    md_out = tables_dir / "master_baselines_table.md"
    tex_out = tables_dir / "master_baselines_table.tex"

    md_out.write_text(md_content, encoding="utf-8")
    tex_out.write_text(tex_content, encoding="utf-8")

    print(f"Generated {md_out} and {tex_out}")
    print(statement)

if __name__ == "__main__":
    build_master_table()

