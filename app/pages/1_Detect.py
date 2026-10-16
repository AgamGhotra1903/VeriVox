"""Page 1: Forensic Detection and Single-Clip Inspection.

Author: Agam Ghotra (IIT2024226), Chunit Bansal (IIT2024223)
"""

import json
import sys
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
import streamlit as st
import torch

st.set_page_config(page_title="Forensic Detection - Acoustic Forensics", page_icon="🔍", layout="wide")

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from acforensics.constants import SAMPLE_RATE
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.inference.predictor import ForensicPredictor
from acforensics.evaluation.robustness import compress_and_decompress_audio

try:
    from app.components.sidebar import render_sidebar
    from app.components.plots import (
        create_confidence_gauge,
        create_attention_waveform_plot,
        create_spectrogram_heatmap,
    )
except ModuleNotFoundError:
    from components.sidebar import render_sidebar
    from components.plots import (
        create_confidence_gauge,
        create_attention_waveform_plot,
        create_spectrogram_heatmap,
    )

# Synchronized Global Sidebar
sidebar_state = render_sidebar()

# Cached Predictor Engine
@st.cache_resource
def get_predictor():
    return ForensicPredictor(
        checkpoint_path="checkpoints/crnn_best.pt",
        norm_stats_path="checkpoints/normalization_stats.json",
        calibrator_path="checkpoints/calibrator.joblib",
        device="cpu",
    )

predictor = get_predictor()

st.title("🔍 Audio Authenticity Screener")
st.markdown("Analyze audio for neural vocoder glitches, synthetic phase artifacts, and voice cloning markers.")

active_model_name = sidebar_state.get("active_model", "CRNN (Best)")
active_feat_rep = sidebar_state.get("feature_representation", "LFCC (60 Coefficients)")
active_threshold = float(sidebar_state.get("threshold_override", 5.23))

st.caption(f"Active Screener: **{active_model_name}** | Feature Mode: **{active_feat_rep}** | Decision Threshold: **{active_threshold:.2f}**")

# Input Selection
col_input, col_results = st.columns([1, 1.2])

with col_input:
    st.subheader("1. Audio Source")
    input_method = st.radio("Choose Input Mode", ["Sample Clip", "Upload Audio File", "Record Microphone"], horizontal=True)

    temp_audio_path = None

    if input_method == "Sample Clip":
        curated_dir = Path("app/assets/samples")
        smoke_dir = Path("data/smoke_test")
        sample_candidates = []
        sample_labels = []

        if curated_dir.exists():
            for p in sorted(curated_dir.glob("*.wav")):
                sample_candidates.append(p)
                clean_name = p.stem.replace("sample_", "").replace("_", " ").title()
                sample_labels.append(f"⭐ Curated: {clean_name}")

        if smoke_dir.exists():
            for p in list(smoke_dir.rglob("*.flac"))[:10] + list(smoke_dir.rglob("*.wav"))[:10]:
                sample_candidates.append(p)
                sample_labels.append(f"📁 Benchmark: {p.parent.name}/{p.name}")

        if sample_candidates:
            chosen_idx = st.selectbox(
                "Select Benchmark Sample",
                range(len(sample_candidates)),
                format_func=lambda i: sample_labels[i],
            )
            temp_audio_path = sample_candidates[chosen_idx]
            st.audio(str(temp_audio_path))
        else:
            st.info("No sample clips found. Run `make smoke` or upload an audio file.")

    elif input_method == "Upload Audio File":
        uploaded = st.file_uploader("Upload Audio (WAV, FLAC, MP3, OGG, M4A)", type=["wav", "flac", "mp3", "ogg", "m4a"])
        if uploaded is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}")
            tfile.write(uploaded.read())
            temp_audio_path = Path(tfile.name)
            st.audio(str(temp_audio_path))

    elif input_method == "Record Microphone":
        mic_audio = st.audio_input("Record Voice Sample")
        if mic_audio is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tfile.write(mic_audio.read())
            temp_audio_path = Path(tfile.name)

    # Optional Stress-Test Re-encoding
    st.markdown("---")
    st.subheader("2. Codec Compression Stress-Test")
    st.caption("Simulate real-world compression to observe verdict stability.")
    col_codec, col_kbps = st.columns(2)
    with col_codec:
        stress_codec = st.selectbox("Codec", ["mp3", "aac", "opus"])
    with col_kbps:
        stress_bitrate = st.select_slider("Bitrate (kbps)", options=[16, 24, 32, 48, 64], value=32)

    apply_stress = st.button("Simulate Compression on this Clip")

with col_results:
    st.subheader("2. Forensic Inspection Results")

    if temp_audio_path is not None and Path(temp_audio_path).exists():
        target_eval_path = temp_audio_path

        if apply_stress:
            with st.spinner(f"Re-encoding audio via {stress_codec.upper()} at {stress_bitrate} kbps..."):
                stress_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                ok = compress_and_decompress_audio(
                    input_wav=temp_audio_path,
                    output_wav=stress_file.name,
                    codec=stress_codec,
                    bitrate_kbps=stress_bitrate,
                )
                if ok:
                    target_eval_path = Path(stress_file.name)
                    st.success(f"Clip re-encoded with {stress_codec.upper()} @ {stress_bitrate} kbps!")
                    st.audio(str(target_eval_path))
                else:
                    st.warning("ffmpeg re-encoding failed. Is ffmpeg installed on your system?")

        try:
            pred = predictor.predict(
                target_eval_path,
                threshold_override=active_threshold,
                model_name=active_model_name,
                feature_type=active_feat_rep,
            )

            # Verdict Badge
            if pred.verdict == "LIKELY GENUINE":
                st.markdown(f'<div class="badge-genuine">✅ {pred.verdict}</div>', unsafe_allow_html=True)
            elif pred.verdict == "LIKELY SYNTHETIC":
                st.markdown(f'<div class="badge-synthetic">🚨 {pred.verdict}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-uncertain">⚠️ {pred.verdict}</div>', unsafe_allow_html=True)

            # Confidence Gauge
            st.plotly_chart(create_confidence_gauge(pred.calibrated_probability_bonafide, pred.verdict), use_container_width=True)

            # Summary Metrics Row
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Clip Duration", f"{pred.duration_sec:.2f} s")
            m_col2.metric("Inference Latency", f"{pred.inference_latency_ms:.1f} ms")
            m_col3.metric("Analysis Windows", f"{pred.num_windows_analyzed}")

            # Technical Details Collapsible
            if sidebar_state.get("show_tech", True):
                with st.expander("Technical Forensic Diagnostics", expanded=False):
                    st.write({
                        "active_model": pred.model_name,
                        "feature_representation": pred.feature_type,
                        "raw_score_llr": f"{pred.raw_score:.4f}",
                        "calibrated_prob_bonafide": f"{pred.calibrated_probability_bonafide:.4f}",
                        "applied_threshold": f"{pred.decision_threshold:.4f}",
                        "confidence_percentage": f"{pred.confidence_percentage:.1f}%",
                        "sha256": pred.file_sha256,
                        "timestamp_utc": pred.timestamp_utc,
                    })

            # Downloadable JSON Report
            report_dict = {
                "verdict": pred.verdict,
                "calibrated_probability_bonafide": pred.calibrated_probability_bonafide,
                "confidence_percentage": pred.confidence_percentage,
                "raw_score": pred.raw_score,
                "threshold": pred.decision_threshold,
                "duration_sec": pred.duration_sec,
                "file_sha256": pred.file_sha256,
                "timestamp_utc": pred.timestamp_utc,
                "model": pred.model_name,
                "feature_type": pred.feature_type,
                "window_timeline": pred.window_timeline,
            }
            st.download_button(
                label="📥 Download Forensic Analysis Report (JSON)",
                data=json.dumps(report_dict, indent=2),
                file_name=f"forensic_report_{pred.file_sha256[:8]}.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"Inference error: {e}")
    else:
        st.info("Select a sample clip or upload an audio file to begin analysis.")

# Visual Evidence Section
st.markdown("---")
st.subheader("3. Visual Forensic Evidence")

if temp_audio_path is not None and Path(temp_audio_path).exists():
    res = load_and_validate_audio(temp_audio_path, target_sr=SAMPLE_RATE)
    if res.is_valid and res.waveform is not None:
        wave_np = res.waveform.numpy()
        tab1, tab2, tab3 = st.tabs(["Attention Waveform Heatmap", "Log-Mel Spectrogram", "LFCC Cepstrogram"])

        with tab1:
            st.caption("The attention heat-strip highlights temporal regions where the model concentrated its classification decision.")
            if 'pred' in locals():
                fig_attn = create_attention_waveform_plot(wave_np, pred.attention_weights, sr=SAMPLE_RATE)
                st.plotly_chart(fig_attn, use_container_width=True)

        with tab2:
            st.caption("Log-Mel Spectrogram (80 Mel bands, 0-8000 Hz). Look for unnatural spectral cuts or lack of harmonic decay.")
            feat_logmel = predictor.logmel_extractor(res.waveform).numpy()
            fig_mel = create_spectrogram_heatmap(feat_logmel, title="Log-Mel Spectrogram (80 Bands)", y_title="Mel Frequency Band")
            st.plotly_chart(fig_mel, use_container_width=True)

        with tab3:
            st.caption("LFCC Matrix (60 coefficients with velocity and acceleration deltas). Linear filterbanks reveal high-frequency phase discontinuities.")
            feat_lfcc = predictor.lfcc_extractor(res.waveform).numpy()
            fig_lfcc = create_spectrogram_heatmap(feat_lfcc, title="LFCC (Static + Delta + Delta-Delta)", y_title="Cepstral Coefficient")
            st.plotly_chart(fig_lfcc, use_container_width=True)

st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>Notice:</strong> High-frequency harmonic degradation typical of MP3 and Opus lossy compression may suppress deepfake artifacts. Always cross-reference with lossless uncompressed captures when possible.
    </div>
    """,
    unsafe_allow_html=True,
)
