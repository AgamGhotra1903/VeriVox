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

st.set_page_config(page_title="About & Methodology - Acoustic Forensics", page_icon="ℹ️", layout="wide")

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from app.components.sidebar import render_sidebar
except ModuleNotFoundError:
    from components.sidebar import render_sidebar

sidebar_state = render_sidebar()

st.title("ℹ️ About Acoustic Forensics")
st.markdown("### A Systems-Building Machine Learning Project for Deepfake Audio Detection")

st.markdown(
    """
    **Acoustic Forensics** is an applied machine learning system designed to discriminate between genuine human speech (bona fide)
    and synthetic speech produced by neural vocoders, voice conversion models, and generative audio pipelines.
    """
)

# 1. Methodology & Pipeline
st.subheader("1. System Architecture")
st.markdown(
    """
    The end-to-end processing pipeline operates deterministically from raw audio to calibrated forensic verdict:
    1. **Audio Validation & Preprocessing**: Audio is ingested, validated against corrupt/silent signals, converted to mono, resampled to 16 kHz, and peak-normalized to [-1, 1].
    2. **Analysis Windowing**: Signals are cropped to a fixed 4.0s duration (64,000 samples) using repeat-padding for short audio, and sliding multi-crop overlapping windows for long speech.
    3. **Feature Extraction**: Computes Linear Frequency Cepstral Coefficients (LFCC) across 20 linear filterbank channels plus 1st (delta) and 2nd (delta-delta) order temporal differentials, yielding a 60 x ~400 feature matrix.
    4. **CRNN Deep Architecture**:
       - 4-stage 2D CNN encoder (32 -> 64 -> 128 -> 128 channels) condensing frequency while preserving temporal frames.
       - 2-layer Bidirectional GRU (hidden dim 128) capturing speech co-articulation and pitch contour consistency.
       - 4-head Multi-Head Self-Attention with residual connections.
       - Learned additive attention pooling producing frame-level decision weights.
    5. **Calibration**: Platt scaling (logistic sigmoid fitted exclusively on development split continuous scores) transforms logits into well-calibrated probabilities.
    """
)

st.markdown("---")

# 2. Team and Individual Contributions
st.subheader("2. Team Members & Code Ownership")
st.markdown(
    """
    | Team Member | Roll Number | Primary Role & Codebase Responsibilities |
    |---|---|---|
    | **Paras Bansal** | IIT2024221 | Dataset curation, protocol parsers, robust audio I/O validation, and feature generation module (`protocols.py`, `audio_io.py`, `lfcc.py`, `normalization.py`). |
    | **Jashan Goyal** | IIT2024223 | Classical ML baselines, dev-set PredefinedSplit hyperparameter tuning, statistical moments pooling (`classical.py`, `statistics.py`, `run_baselines.py`). |
    | **Agam Ghotra** | IIT2024226 | Deep learning CRNN architecture, SpecAugment, mixed-precision training loop, Platt calibration, predictor API (`crnn.py`, `train_crnn.py`, `calibration.py`, `predictor.py`). |
    | **Chunit Bansal** | IIT2024233 | Metric suite, ffmpeg compression stress tests, In-The-Wild cross-domain test, error audit, Streamlit forensic UI (`metrics.py`, `robustness.py`, `crossdomain.py`, `error_analysis.py`, `app/`). |
    """
)

st.markdown("---")

# 3. Metric Formulations
st.subheader("3. Core Evaluation Metrics")
st.markdown(
    """
    - **Equal Error Rate (EER)**: The operating point threshold $\\theta$ where the False Acceptance Rate (synthetic audio accepted as genuine) equals the False Rejection Rate (genuine speech falsely rejected).
    - **ROC-AUC**: Area under the Receiver Operating Characteristic curve, measuring separation ability across all possible operating thresholds independent of class prior.
    - **Minimum Tandem Detection Cost Function (min-tDCF)**: Official ASVspoof metric evaluating the impact of the countermeasure on an end-user Automatic Speaker Verification (ASV) system under standardized cost penalties ($C_{miss}=1, C_{fa}=10$).
    """
)

st.markdown("---")

# 4. Academic References
st.subheader("4. Academic References")
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

st.markdown("---")

# 5. Ethical Statement
st.subheader("5. Ethical Use and Forensic Boundaries")
st.markdown(
    """
    **Dual-Use & Responsible Disclosure:** Synthetic speech technology presents beneficial applications (assistive speech synthesis, dubbing, accessibility)
    alongside significant malicious potential (biometric authentication spoofing, disinformation, CEO fraud). This system is developed strictly for
    defensive media forensics and academic research. It must not be weaponized or relied upon as sole legal evidence without corroborating metadata
    and expert human forensic review.
    """
)
