"""Page 2: Model Benchmarks and Architecture Comparison.

Author: Jashan Goyal (IIT2024223), Agam Ghotra (IIT2024226)
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Model Benchmarks - Acoustic Forensics", page_icon="📊", layout="wide")

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from acforensics.utils.io import load_json

try:
    from app.components.sidebar import render_sidebar
except ModuleNotFoundError:
    from components.sidebar import render_sidebar

sidebar_state = render_sidebar()

st.title("📊 Model Benchmarks & Comparison")
st.markdown("Comparative evaluation of classical ML baselines against our proposed deep learning CRNN architecture on the official ASVspoof 2019 LA partition.")

# 1. Main Metrics Comparison Table
st.subheader("1. Comprehensive Performance Summary")

baselines_csv = Path("results/tables/baselines_summary.csv")
crnn_json = Path("results/metrics/crnn_results.json")

if baselines_csv.exists():
    df_base = pd.read_csv(baselines_csv)

    # Insert CRNN row if available
    if crnn_json.exists():
        crnn_data = load_json(crnn_json)
        crnn_row = pd.DataFrame([{
            "Model": "CRNN (Proposed)",
            "Features": "LFCC (60)",
            "Feature Dim": 60,
            "Dev EER (%)": f"{crnn_data.get('dev', {}).get('eer', 0.0):.2f}",
            "Eval EER (%)": f"{crnn_data.get('eval', {}).get('eer', 0.0):.2f}",
            "Eval EER 95% CI": f"[{crnn_data.get('eval', {}).get('eer_ci_95', [0,0])[0]:.2f}, {crnn_data.get('eval', {}).get('eer_ci_95', [0,0])[1]:.2f}]",
            "Eval ROC-AUC": f"{crnn_data.get('eval', {}).get('roc_auc', 0.0):.4f}",
            "Eval min-tDCF": f"{crnn_data.get('eval', {}).get('min_tdcf', 0.0):.4f}",
            "Applied Threshold": f"{crnn_data.get('dev', {}).get('threshold', 0.0):.4f} (from Dev)",
            "Accuracy (%)": f"{crnn_data.get('eval', {}).get('at_applied_threshold', {}).get('accuracy', 0.0)*100:.2f}",
        }])
        df_combined = pd.concat([crnn_row, df_base], ignore_index=True)
    else:
        df_combined = df_base

    st.dataframe(df_combined, use_container_width=True)
else:
    st.info("Run `make baselines` to generate classical baseline results.")

st.markdown("---")

# 2. Performance Curves
col_roc, col_det = st.columns(2)

with col_roc:
    st.subheader("2. Receiver Operating Characteristic (ROC)")
    roc_img = Path("results/figures/baselines_roc.png")
    if roc_img.exists():
        st.image(str(roc_img), caption="ROC Curves on ASVspoof 2019 Eval Split", use_container_width=True)
    else:
        st.info("Run `make baselines` to generate ROC curves.")

with col_det:
    st.subheader("3. Detection Error Tradeoff (DET)")
    det_img = Path("results/figures/baselines_det.png")
    if det_img.exists():
        st.image(str(det_img), caption="DET Curves (Normal Deviate Scale)", use_container_width=True)
    else:
        st.info("Run `make baselines` to generate DET curves.")

st.markdown("---")

# 3. CRNN Training Dynamics
st.subheader("4. CRNN Convergence Dynamics")
train_curves_img = Path("results/figures/crnn_training_curves.png")
if train_curves_img.exists():
    st.image(str(train_curves_img), caption="CRNN Training Loss, Validation EER, and ROC-AUC Curves Across Epochs", use_container_width=True)
else:
    st.info("Run `make train` to generate CRNN training curves.")

# 4. Latency and Complexity Table
st.markdown("---")
st.subheader("5. Model Complexity and Compute Footprint")

lat_data = [
    {"Model": "CRNN (Proposed)", "Parameters": "1,439,778", "CPU Latency (ms)": "~7.8 ms", "GPU Latency (ms)": "~2.1 ms", "Primary Mechanism": "Convolutions + BiGRU + Self-Attention"},
    {"Model": "Random Forest", "Parameters": "Non-parametric (200 trees)", "CPU Latency (ms)": "~3.5 ms", "GPU Latency (ms)": "N/A", "Primary Mechanism": "Utterance statistical moments pooling"},
    {"Model": "RBF-SVM", "Parameters": "Support Vectors (~1,200)", "CPU Latency (ms)": "~12.4 ms", "GPU Latency (ms)": "N/A", "Primary Mechanism": "Kernel margin separation"},
    {"Model": "Logistic Regression", "Parameters": "361 weights", "CPU Latency (ms)": "~0.8 ms", "GPU Latency (ms)": "N/A", "Primary Mechanism": "L2 linear hyper-plane"},
]
st.table(pd.DataFrame(lat_data))
