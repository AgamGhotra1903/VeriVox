"""Page 5: About, Methodology, Team Contributions, and Ethical Statement.

Academic Project Team:
- Paras Bansal (IIT2024221)
- Jashan Goyal (IIT2024223)
- Agam Ghotra (IIT2024226)
- Chunit Bansal (IIT2024233)
"""

import sys
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="About & Methodology - VeriVox",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from app.components.sidebar import render_sidebar
except ModuleNotFoundError:
    from components.sidebar import render_sidebar

sidebar_state = render_sidebar()

st.title("ℹ️ About VeriVox & Technical Methodology")
st.markdown("### Applied Machine Learning & Acoustic Forensics for Synthetic Speech Detection")

st.markdown(
    """
    <div class="forensic-card" style="margin-bottom: 24px;">
        <p style="font-size: 1.05rem; line-height: 1.6; color: #E2E8F0; margin: 0;">
            <strong>VeriVox</strong> is an academic machine learning system engineered to discriminate genuine human speech (<em>bona fide</em>) 
            from spoofed speech produced by modern neural text-to-speech vocoders, voice conversion models, and generative audio pipelines. 
            The system combines linear frequency cepstral analysis with deep recurrent convolutional neural networks and empirical Platt probability calibration.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_arch, tab_team, tab_math, tab_bib, tab_ethics = st.tabs([
    "🏗️ System Architecture",
    "👥 Team & Code Ownership",
    "📐 Metric Formulations",
    "📚 Academic References",
    "⚖️ Ethical Boundaries",
])

with tab_arch:
    st.subheader("1. End-to-End Processing Architecture")
    st.markdown(
        """
        The processing pipeline operates deterministically from raw digital audio to calibrated forensic verdict:
        
        1. **Audio Validation & Preprocessing** (`audio_io.py`): Audio is ingested across multiple container formats (WAV, FLAC, MP3, AAC, Opus, M4A), checked for corrupt/silent data, downmixed to mono, resampled to 16,000 Hz, and peak-normalized.
        2. **Analysis Windowing** (`segmentation.py`): Signals are cropped to fixed 4.0-second intervals (64,000 samples) using repeat-padding for short clips and sliding multi-crop overlapping windows for extended utterances.
        3. **Feature Extraction** (`lfcc.py`): Extracts Linear Frequency Cepstral Coefficients (LFCC) across 20 linear filterbank channels plus 1st (delta) and 2nd (delta-delta) order temporal differentials, producing a $60 \\times \\approx 400$ feature tensor.
        4. **CRNN Deep Neural Network** (`crnn.py`):
           - **4-Stage 2D CNN Encoder** ($32 \\rightarrow 64 \\rightarrow 128 \\rightarrow 128$ channels) that compresses spectral bands while preserving fine frame-level timing.
           - **2-Layer Bidirectional GRU** (hidden dim 128) capturing phonemic co-articulation and pitch contour consistency.
           - **4-Head Multi-Head Self-Attention** with residual connections focusing on synthetic transition anomalies.
           - **Learned Attention Pooling** computing frame-level decision weights $\\alpha_t$ and aggregating into a clip-level embedding.
        5. **Platt Calibration** (`calibration.py`): Sigmoidal scaling fitted exclusively on development split log-likelihood scores transforms raw continuous margins into well-calibrated posterior probabilities $P(\\text{bona fide} \\mid X)$.
        """
    )

with tab_team:
    st.subheader("2. Academic Project Team & Module Ownership")
    st.markdown(
        """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-top: 14px;">
            <div class="forensic-card" style="margin: 0;">
                <h4 style="margin: 0; color: #38BDF8;">Paras Bansal</h4>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94A3B8; margin-top: 2px;">IIT2024221</div>
                <hr style="border-color: #334155; margin: 10px 0;">
                <p style="font-size: 0.86rem; color: #CBD5E1; margin: 0; line-height: 1.5;">
                    <strong>Lead Data Engineer:</strong> Protocol parsing, dataset curation (ASVspoof 2019 LA), audio I/O validation, feature extractors (LFCC, Log-Mel, Normalization).
                </p>
                <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #64748B; margin-top: 8px;">
                    <code>protocols.py</code>, <code>audio_io.py</code>, <code>lfcc.py</code>, <code>normalization.py</code>
                </div>
            </div>
            <div class="forensic-card" style="margin: 0;">
                <h4 style="margin: 0; color: #38BDF8;">Jashan Goyal</h4>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94A3B8; margin-top: 2px;">IIT2024223</div>
                <hr style="border-color: #334155; margin: 10px 0;">
                <p style="font-size: 0.86rem; color: #CBD5E1; margin: 0; line-height: 1.5;">
                    <strong>Classical ML Engineer:</strong> Classical ML baselines (LR, RBF-SVM, Random Forest), PredefinedSplit dev hyperparameter tuning, statistical moments pooling.
                </p>
                <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #64748B; margin-top: 8px;">
                    <code>classical.py</code>, <code>statistics.py</code>, <code>run_baselines.py</code>
                </div>
            </div>
            <div class="forensic-card" style="margin: 0;">
                <h4 style="margin: 0; color: #38BDF8;">Agam Ghotra</h4>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94A3B8; margin-top: 2px;">IIT2024226</div>
                <hr style="border-color: #334155; margin: 10px 0;">
                <p style="font-size: 0.86rem; color: #CBD5E1; margin: 0; line-height: 1.5;">
                    <strong>Deep Learning Architect:</strong> CRNN architecture (CNN+BiGRU+Attention), SpecAugment, mixed-precision training engine, Platt calibrator, inference API.
                </p>
                <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #64748B; margin-top: 8px;">
                    <code>crnn.py</code>, <code>train_crnn.py</code>, <code>calibration.py</code>, <code>predictor.py</code>
                </div>
            </div>
            <div class="forensic-card" style="margin: 0;">
                <h4 style="margin: 0; color: #38BDF8;">Chunit Bansal</h4>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94A3B8; margin-top: 2px;">IIT2024233</div>
                <hr style="border-color: #334155; margin: 10px 0;">
                <p style="font-size: 0.86rem; color: #CBD5E1; margin: 0; line-height: 1.5;">
                    <strong>Forensics & UI Engineer:</strong> Evaluation metrics (EER, min-tDCF, ROC-AUC), ffmpeg compression stress tests, In-The-Wild cross-domain test, Streamlit UI.
                </p>
                <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #64748B; margin-top: 8px;">
                    <code>metrics.py</code>, <code>robustness.py</code>, <code>crossdomain.py</code>, <code>app/</code>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_math:
    st.subheader("3. Core Forensic Evaluation Metrics")
    st.markdown(
        """
        - **Equal Error Rate (EER)**: The operating decision threshold $\\theta^*$ where the False Acceptance Rate (synthetic audio accepted as genuine) equals the False Rejection Rate (genuine speech falsely flagged as synthetic):
          $$\\text{FAR}(\\theta^*) = \\text{FRR}(\\theta^*) = \\text{EER}$$
        - **Receiver Operating Characteristic (ROC-AUC)**: The definite integral of the True Positive Rate against the False Positive Rate across all continuous thresholds $\\theta \\in [-\\infty, +\\infty]$:
          $$\\text{AUC} = \\int_{0}^{1} \\text{TPR}(\\text{FPR}^{-1}(t)) \\, dt$$
        - **Minimum Tandem Detection Cost Function (min-tDCF)**: The official ASVspoof countermeasure metric evaluating operational impact on an automatic speaker verification (ASV) system under standardized penalties ($C_{\\text{miss}}=1, C_{\\text{fa}}=10$):
          $$\\text{tDCF}(\\theta) = C_{\\text{miss}} \\cdot P_{\\text{tar}} \\cdot P_{\\text{miss}}^{\\text{cm}}(\\theta) + C_{\\text{fa}} \\cdot P_{\\text{non}} \\cdot P_{\\text{fa}}^{\\text{cm}}(\\theta)$$
        """
    )

with tab_bib:
    st.subheader("4. Published Academic References")
    st.markdown(
        """
        1. **Wang, X., et al.** (2020). *ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech.* Computer Speech & Language, 64, 101114. [arXiv:1911.01601](https://arxiv.org/abs/1911.01601)
        2. **Todisco, M., et al.** (2019). *ASVspoof 2019: Future horizons in spoofed and fake audio detection.* Interspeech 2019, 1008-1012.
        3. **Müller, N., et al.** (2022). *Does Audio Deepfake Detection Generalize?* Interspeech 2022, 2783-2787.
        4. **Pham, V., et al.** (2024). *Deepfake Speech Detection via Multi-Scale Spectral Ensembles.* IEEE IS2 2024. doi: 10.1109/IS262782.2024.10704095.
        5. **Tak, H., et al.** (2021). *End-to-End Spectro-Temporal Graph Convolutional Network for Speech Anti-Spoofing.* Proc. Interspeech 2021, 1014-1018.
        6. **Lavrentyeva, G., et al.** (2019). *STC Antispoofing Systems for the ASVspoof 2019 Challenge.* Proc. Interspeech 2019, 1033-1037.
        """
    )

with tab_ethics:
    st.subheader("5. Ethical Use and Forensic Boundaries")
    st.markdown(
        """
        <div class="forensic-card">
            <h4 style="margin-top: 0; color: #F59E0B;">Dual-Use Awareness & Defensive Disclosure</h4>
            <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.6;">
                Generative voice cloning and neural speech synthesis present profound social benefits (assistive vocal communication, dubbing, linguistic preservation) 
                alongside catastrophic misuse vectors (biometric voice authorization spoofing, impersonation scams, disinformation). 
            </p>
            <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.6;">
                <strong>VeriVox</strong> is engineered strictly for <strong>defensive forensic analysis, academic research, and public integrity verification</strong>. 
                Under no circumstances should automated model scores be utilized as singular, uncorroborated evidence in judicial proceedings without 
                human expert acoustic verification and cryptographic chain-of-custody documentation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
