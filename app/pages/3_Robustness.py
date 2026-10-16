"""Page 3: Robustness Stress-Testing and Cross-Domain Generalization.

Author: Chunit Bansal (IIT2024223)
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Robustness & Cross-Domain - Acoustic Forensics", page_icon="📉", layout="wide")

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

st.title("📉 Robustness & Stress-Testing")
st.markdown("Quantifying model resilience against real-world lossy compression (MP3, AAC, Opus) and zero-shot out-of-domain distribution shift.")

# Section 1: Codec Compression
st.subheader("1. Lossy Codec Compression Stress-Test")
st.markdown(
    """
    Consumer communication channels (telephony, WhatsApp, YouTube, Discord) re-encode audio with aggressive lossy codecs.
    Lossy psychoacoustic algorithms eliminate high-frequency bands and perturb phase continuity.
    Here we evaluate model degradation from **64 kbps down to 16 kbps**.
    """
)

col_chart, col_table = st.columns([1.2, 1])

with col_chart:
    rob_fig = Path("results/figures/robustness_eer_vs_bitrate.png")
    if rob_fig.exists():
        st.image(str(rob_fig), caption="EER vs Bitrate (MP3, AAC, Opus)", use_container_width=True)
    else:
        st.info("Run `make robustness` to generate codec degradation plots.")

with col_table:
    rob_csv = Path("results/tables/robustness_degradation.csv")
    if rob_csv.exists():
        df_rob = pd.read_csv(rob_csv)
        st.dataframe(df_rob, use_container_width=True)
    else:
        st.info("Run `make robustness` to generate the degradation table.")

st.markdown("---")

# Section 2: In-The-Wild Cross-Domain
st.subheader("2. In-The-Wild Zero-Shot Generalization (Müller et al., Interspeech 2022)")
st.markdown(
    """
    **The Cross-Domain Challenge:** Models trained exclusively on studio-condition benchmarks often catastrophically fail
    when tested on real-world celebrity deepfakes due to acoustic distribution shifts (reverberation, microphone variability, ambient noise).
    """
)

cd_file = Path("results/metrics/crossdomain_results.json")
if cd_file.exists():
    cd_data = load_json(cd_file)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluated Clips", f"{cd_data.get('num_evaluated_clips', 0)}")
    c2.metric("Optimal In-Domain EER", f"{cd_data.get('optimal_threshold_metrics', {}).get('itw_optimal_eer', 0.0):.2f}%")
    c3.metric("Optimal Threshold", f"{cd_data.get('optimal_threshold_metrics', {}).get('itw_optimal_threshold', 0.0):.4f}")
    c4.metric("Cross-Domain Accuracy", f"{cd_data.get('dev_threshold_metrics', {}).get('accuracy', 0.0)*100:.2f}%")

    st.markdown(
        f"""
        <div class="forensic-card">
            <h4>🔬 Analysis of Calibration & Domain Shift</h4>
            <p>{cd_data.get('domain_shift_analysis', '')}</p>
            <p><strong>Note on Tandem DCF:</strong> {cd_data.get('min_tdcf_note', '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Run `make crossdomain` to evaluate the In-The-Wild dataset.")
