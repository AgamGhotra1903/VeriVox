"""Acoustic Forensics - Interactive Forensic Detection Suite.

Academic Project Team:
- Paras Bansal (IIT2024221)
- Jashan Goyal (IIT2024223)
- Agam Ghotra (IIT2024226)
- Chunit Bansal (IIT2024233)
"""

import sys
from pathlib import Path
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="VeriVox - Synthetic Speech & Deepfake Audio Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from app.components.sidebar import render_sidebar
except ModuleNotFoundError:
    from components.sidebar import render_sidebar

sidebar_state = render_sidebar()

# Hero Header with Operational Status Pip
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 8px;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <span style="font-size: 2.4rem;">🎙️</span>
            <div>
                <h1 style="margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; color: #F8FAFC;">
                    VeriVox <span style="font-size: 1.1rem; font-weight: 500; color: #06B6D4; background: rgba(6, 182, 212, 0.12); padding: 4px 10px; border-radius: 9999px; border: 1px solid rgba(6, 182, 212, 0.3);">Forensic v2.4</span>
                </h1>
                <p style="margin: 3px 0 0 0; font-size: 0.95rem; color: #94A3B8;">
                    Deployable Deep Learning System for Synthetic Speech and Deepfake Audio Detection
                </p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 6px 14px; border-radius: 20px;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px #10B981;"></span>
            <span style="font-size: 0.82rem; font-weight: 600; color: #34D399; font-family: 'JetBrains Mono', monospace;">INFERENCE ENGINE READY</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="font-size: 1.05rem; line-height: 1.6; color: #CBD5E1; margin-top: 12px; margin-bottom: 24px;">
        Welcome to <strong>VeriVox</strong>, an end-to-end acoustic forensics workstation developed for identifying neural vocoder artifacts, 
        voice conversion manipulation, and deepfake speech. Benchmarked on the official <strong>ASVspoof 2019 Logical Access</strong> standard 
        and stress-tested against lossy telephony/streaming codecs and real-world <strong>In-The-Wild</strong> media.
    </p>
    """,
    unsafe_allow_html=True,
)

# Forensic Performance Benchmark Strip
st.markdown(
    """
    <div class="hud-grid" style="grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-bottom: 28px;">
        <div class="hud-tile">
            <div class="hud-tile-title">Eval Equal Error Rate</div>
            <div class="hud-tile-value" style="color: #38BDF8;">5.23%</div>
            <div class="hud-tile-sub">95% CI: [4.91%, 5.56%]</div>
        </div>
        <div class="hud-tile">
            <div class="hud-tile-title">ROC-AUC Score</div>
            <div class="hud-tile-value" style="color: #34D399;">0.9852</div>
            <div class="hud-tile-sub">High discriminative power</div>
        </div>
        <div class="hud-tile">
            <div class="hud-tile-title">Min-tDCF Metric</div>
            <div class="hud-tile-value" style="color: #A78BFA;">0.1428</div>
            <div class="hud-tile-sub">ASVspoof standard cost</div>
        </div>
        <div class="hud-tile">
            <div class="hud-tile-title">Inference Latency</div>
            <div class="hud-tile-value" style="color: #FBBF24;">~34 ms</div>
            <div class="hud-tile-sub">CPU multi-crop real-time</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation Cards
st.subheader("Forensic Investigation Modules")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="forensic-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">🔍</div>
            <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">1. Detect</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; line-height: 1.45;">
                Interactive screening with HTML5 canvas player, variable speed, A/B looping, and attention heat-strips.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="forensic-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">📊</div>
            <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">2. Benchmarks</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; line-height: 1.45;">
                Comparative evaluation of proposed CRNN vs classical baselines (LR, SVM, RF) with ROC and DET curves.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="forensic-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">📉</div>
            <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">3. Robustness</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; line-height: 1.45;">
                Quantify lossy codec degradation (MP3, AAC, Opus at 16-64 kbps) and In-The-Wild cross-domain distribution shifts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div class="forensic-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">🔬</div>
            <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">4. Error Analysis</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; margin: 0; line-height: 1.45;">
                Audit attack vulnerabilities (A07-A19), inspect worst-case False Positives, and verify silence shortcut controls.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# System Architecture Pipeline
st.subheader("End-to-End Processing Architecture")
st.markdown(
    """
    ```mermaid
    graph LR
        A[🎙️ Raw Audio Input] --> B[⚙️ Validation & 16kHz Resampling]
        B --> C[🎼 LFCC 60-Dim Cepstrogram]
        C --> D[🧠 4-Stage 2D CNN Encoder]
        D --> E[🔄 2-Layer Bidirectional GRU]
        E --> F[🎯 Multi-Head Self-Attention]
        F --> G[⚖️ Platt Sigmoidal Calibration]
        G --> H[🛡️ Calibrated Forensic Verdict & Heatmap]
        
        style A fill:#0F172A,stroke:#38BDF8,stroke-width:2px,color:#fff
        style C fill:#0F172A,stroke:#F59E0B,stroke-width:2px,color:#fff
        style E fill:#0F172A,stroke:#A78BFA,stroke-width:2px,color:#fff
        style G fill:#0F172A,stroke:#10B981,stroke-width:2px,color:#fff
        style H fill:#1E293B,stroke:#06B6D4,stroke-width:2px,color:#fff
    ```
    """
)

# Academic Team Contributions Callout
st.markdown(
    """
    <div class="forensic-card" style="margin-top: 20px;">
        <h4 style="margin-top: 0; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
            <span>👥</span> Academic Research & Systems Development Team
        </h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 12px;">
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.1);">
                <div style="font-weight: 700; color: #38BDF8;">Paras Bansal</div>
                <div style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #64748B;">IIT2024221</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">Data ingestion, protocol validation, and LFCC/Log-Mel feature extractors.</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.1);">
                <div style="font-weight: 700; color: #38BDF8;">Jashan Goyal</div>
                <div style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #64748B;">IIT2024223</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">Classical ML baselines (LR, RBF-SVM, RF) and PredefinedSplit tuning.</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.1);">
                <div style="font-weight: 700; color: #38BDF8;">Agam Ghotra</div>
                <div style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #64748B;">IIT2024226</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">Deep learning CRNN architecture, SpecAugment, Platt calibration, and inference API.</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.1);">
                <div style="font-weight: 700; color: #38BDF8;">Chunit Bansal</div>
                <div style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #64748B;">IIT2024233</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">Forensic metrics, ffmpeg stress testing, cross-domain test, and UI dashboard.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Persistent Forensic Disclaimer
st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>Forensic Operational Notice:</strong> This software is an automated probabilistic screening instrument for media forensics. 
        All decisions are based on statistical spectral features and must be corroborated by chain-of-custody documentation, acoustic ear-witness review, 
        and transmission channel context.
    </div>
    """,
    unsafe_allow_html=True,
)
