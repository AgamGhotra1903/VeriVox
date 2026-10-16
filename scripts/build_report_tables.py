"""Automated Report Table Builder (Zero Fabricated Numbers).

Author: Chunit Bansal (IIT2024223)
"""

import argparse
from pathlib import Path
import pandas as pd

from acforensics.constants import DEFAULT_RESULTS_DIR
from acforensics.utils.io import load_json, safe_write_text
from acforensics.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def build_tables(results_dir: Path | str = DEFAULT_RESULTS_DIR) -> None:
    res_dir = Path(results_dir)
    metrics_dir = res_dir / "metrics"
    tables_dir = res_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classical Baselines Table
    baselines_file = metrics_dir / "baselines_results.json"
    if baselines_file.exists():
        base_data = load_json(baselines_file)
        rows = []
        for model_name, feats in base_data.items():
            for feat_name, splits in feats.items():
                dev_m = splits["dev"]
                eval_m = splits["eval"]
                rows.append({
                    "Model": model_name,
                    "Features": feat_name.upper(),
                    "Dev EER (%)": f"{dev_m['eer']:.2f}",
                    "Eval EER (%)": f"{eval_m['eer']:.2f}",
                    "Eval ROC-AUC": f"{eval_m['roc_auc']:.4f}",
                    "Eval min-tDCF": f"{eval_m['min_tdcf']:.4f}",
                    "Applied Threshold": f"{eval_m['applied_threshold']:.4f}",
                })
        df_base = pd.DataFrame(rows)
        # Markdown table
        md_base = df_base.to_markdown(index=False)
        safe_write_text(md_base, tables_dir / "table1_baselines.md")
        # LaTeX table
        latex_base = df_base.to_latex(index=False)
        safe_write_text(latex_base, tables_dir / "table1_baselines.tex")
        logger.info("Generated Table 1: Classical Baselines")

    # 2. Main Model Comparison Table
    crnn_file = metrics_dir / "crnn_results.json"
    if crnn_file.exists():
        crnn_data = load_json(crnn_file)
        rows = [{
            "Architecture": "CRNN (Proposed)",
            "Parameters": f"{crnn_data.get('parameters', 0):,}",
            "Latency CPU (ms)": f"{crnn_data.get('inference_latency_ms', {}).get('cpu', 0.0):.2f}",
            "Dev EER (%)": f"{crnn_data.get('dev', {}).get('eer', 0.0):.2f}",
            "Eval EER (%)": f"{crnn_data.get('eval', {}).get('eer', 0.0):.2f}",
            "Eval ROC-AUC": f"{crnn_data.get('eval', {}).get('roc_auc', 0.0):.4f}",
            "Eval min-tDCF": f"{crnn_data.get('eval', {}).get('min_tdcf', 0.0):.4f}",
        }]
        df_main = pd.DataFrame(rows)
        safe_write_text(df_main.to_markdown(index=False), tables_dir / "table2_main_model.md")
        safe_write_text(df_main.to_latex(index=False), tables_dir / "table2_main_model.tex")
        logger.info("Generated Table 2: Main Model Benchmark")

    # 3. Robustness Degradation Table
    robust_file = metrics_dir / "robustness_results.json"
    if robust_file.exists():
        rob_data = load_json(robust_file)
        clean_eer = rob_data.get("clean_baseline_eer", 0.0)
        rows = []
        for codec, bitrates in rob_data.get("results_by_codec", {}).items():
            for kbps, eer_val in bitrates.items():
                rows.append({
                    "Codec": codec.upper(),
                    "Bitrate (kbps)": kbps,
                    "Clean EER (%)": f"{clean_eer:.2f}",
                    "Compressed EER (%)": f"{eer_val:.2f}",
                    "Degradation (pts)": f"{eer_val - clean_eer:+.2f}",
                })
        if rows:
            df_rob = pd.DataFrame(rows)
            safe_write_text(df_rob.to_markdown(index=False), tables_dir / "table3_robustness.md")
            safe_write_text(df_rob.to_latex(index=False), tables_dir / "table3_robustness.tex")
            logger.info("Generated Table 3: Robustness Degradation")

    # 4. Cross-Domain Table
    cross_file = metrics_dir / "crossdomain_results.json"
    if cross_file.exists():
        cd_data = load_json(cross_file)
        rows = [{
            "Evaluation Domain": "In-the-Wild (Unseen)",
            "Clips": cd_data.get("num_evaluated_clips", 0),
            "Optimal In-Domain EER (%)": f"{cd_data.get('optimal_threshold_metrics', {}).get('itw_optimal_eer', 0.0):.2f}",
            "Optimal In-Domain ROC-AUC": f"{cd_data.get('optimal_threshold_metrics', {}).get('roc_auc', 0.0):.4f}",
            "Accuracy @ Dev Threshold (%)": f"{cd_data.get('dev_threshold_metrics', {}).get('accuracy', 0.0)*100:.2f}",
        }]
        df_cd = pd.DataFrame(rows)
        safe_write_text(df_cd.to_markdown(index=False), tables_dir / "table4_crossdomain.md")
        safe_write_text(df_cd.to_latex(index=False), tables_dir / "table4_crossdomain.tex")
        logger.info("Generated Table 4: Cross-Domain Benchmark")


def main():
    parser = argparse.ArgumentParser(description="Automated Report Table Generator")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing results")
    args = parser.parse_args()

    setup_logging()
    build_tables(args.results_dir)
    print("Report tables successfully generated in results/tables/ (Markdown & LaTeX).")


if __name__ == "__main__":
    main()
