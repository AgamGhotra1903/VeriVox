"""Page 4: Forensic Error Analysis and Attack Breakdown.

Author: Chunit Bansal (IIT2024223)
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Error Analysis - Acoustic Forensics", page_icon="🔬", layout="wide")

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

st.title("🔬 Forensic Error Analysis & Attack Audit")
st.markdown("Detailed breakdown of False Positives, False Negatives, per-attack vulnerabilities (A07-A19), and worst-case failure modes.")

# 1. Per-Attack Breakdown
st.subheader("1. Unseen Synthesis Attack Vulnerability (A07 - A19)")
st.caption("Attacks in yellow/amber represent variants of known training algorithms (A16 is an A04 variant; A17/A19 relate to A06 variants).")

attack_img = Path("results/figures/per_attack_eer.png")
if attack_img.exists():
    st.image(str(attack_img), caption="Equal Error Rate (EER %) Across Individual Eval Attack Conditions", use_container_width=True)
else:
    st.info("Run `make errors` to generate per-attack error breakdowns.")

st.markdown("---")

# 2. Plain Language Findings
st.subheader("2. Audit Findings & Honest Forensic Assessment")
error_file = Path("results/metrics/error_analysis_summary.json")

if error_file.exists():
    err_data = load_json(error_file)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluated Trials", f"{err_data.get('total_trials', 0)}")
    c2.metric("False Positives", f"{err_data.get('false_positive_count', 0)}")
    c3.metric("False Negatives", f"{err_data.get('false_negative_count', 0)}")
    c4.metric("Highest EER Attack", f"{err_data.get('most_vulnerable_attack', 'N/A')} ({err_data.get('most_vulnerable_eer', 0.0):.1f}%)")

    st.markdown("#### Key Findings")
    for finding in err_data.get("plain_language_findings", []):
        st.markdown(f"- {finding}")
else:
    st.info("Run `make errors` to generate forensic findings.")

st.markdown("---")

# 3. Worst-Case Error Table
st.subheader("3. Worst-Case Error Auditing (Top Misclassifications)")
st.caption("Files with the largest discrepancy between ground truth and continuous score.")

worst_csv = Path("results/tables/worst_case_errors.csv")
if worst_csv.exists():
    df_worst = pd.read_csv(worst_csv)
    st.dataframe(df_worst, use_container_width=True)
else:
    st.info("Run `make errors` to generate worst-case error tables.")

st.markdown("---")

# 4. Silence Shortcut Learning Check
st.subheader("4. Silence Shortcut Learning Rigor Check")
master_eval = Path("results/metrics/master_evaluation.json")
if master_eval.exists():
    m_data = load_json(master_eval)
    delta = m_data.get("silence_shortcut_delta_eer", 0.0)
    st.markdown(
        f"""
        <div class="forensic-card">
            <h4>Voice Activity Detection (VAD) Silence-Trimmed Test</h4>
            <p><strong>Standard Eval EER:</strong> {m_data.get('standard_evaluation', {}).get('eer', 0.0):.2f}% | 
               <strong>Silence-Trimmed EER:</strong> {m_data.get('silence_trimmed_evaluation', {}).get('eer', 0.0):.2f}% | 
               <strong>Delta:</strong> {delta:+.2f}%</p>
            <p>{m_data.get('silence_analysis', '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
