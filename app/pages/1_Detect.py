"""Page 1: Forensic Detection, Single-Clip Inspection, and HUD Diagnostic Suite.

Author: Agam Ghotra (IIT2024226), Chunit Bansal (IIT2024233)
"""

import json
import sys
import time
from pathlib import Path
import tempfile
import numpy as np
import streamlit as st
import torch

st.set_page_config(
    page_title="Forensic Detection - VeriVox",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    from app.components.audio_player import render_forensic_audio_player
    from app.components.plots import (
        create_confidence_gauge,
        create_attention_waveform_plot,
        create_spectrogram_heatmap,
    )
except ModuleNotFoundError:
    from components.sidebar import render_sidebar
    from components.audio_player import render_forensic_audio_player
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
st.markdown("Precision acoustic forensics workstation for detecting synthetic speech, voice cloning, and neural vocoder artifacts.")

active_model_name = sidebar_state.get("active_model", "CRNN (Best)")
active_feat_rep = sidebar_state.get("feature_representation", "LFCC (60 Coefficients)")
active_threshold = float(sidebar_state.get("threshold_override", 5.23))

st.caption(f"Active Screener: **{active_model_name}** | Feature Mode: **{active_feat_rep}** | Decision Threshold: **{active_threshold:.2f}**")

# Input Selection Row
col_input, col_results = st.columns([1, 1.25])

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
            render_forensic_audio_player(temp_audio_path)
        else:
            st.info("No sample clips found. Run `make smoke` or upload an audio file.")

    elif input_method == "Upload Audio File":
        uploaded = st.file_uploader(
            "Upload Audio (WAV, FLAC, MP3, OGG, M4A, AAC)",
            type=["wav", "flac", "mp3", "ogg", "m4a", "aac"],
            help="High-precision analysis works on standard lossless and compressed formats.",
        )
        if uploaded is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}")
            tfile.write(uploaded.read())
            temp_audio_path = Path(tfile.name)
            render_forensic_audio_player(temp_audio_path)

    elif input_method == "Record Microphone":
        mic_audio = st.audio_input("Record Voice Sample")
        if mic_audio is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tfile.write(mic_audio.read())
            temp_audio_path = Path(tfile.name)
            render_forensic_audio_player(temp_audio_path)

    # Optional Stress-Test Re-encoding
    st.markdown("---")
    st.subheader("2. Codec Compression Stress-Test")
    st.caption("Simulate aggressive social media / messaging compression to observe verdict resilience.")
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
                    render_forensic_audio_player(target_eval_path)
                else:
                    st.warning("ffmpeg re-encoding failed. Is ffmpeg installed on your system?")

        try:
            # Forensic Analysis Pipeline
            with st.spinner("Analyzing acoustic features & calculating confidence..."):
                pred = predictor.predict(
                    target_eval_path,
                    threshold_override=active_threshold,
                    model_name=active_model_name,
                    feature_type=active_feat_rep,
                )

            # 1. Primary Verdict Badge with Glow
            st.markdown('<div class="verdict-container">', unsafe_allow_html=True)
            if pred.verdict == "LIKELY GENUINE":
                st.markdown(f'<div class="badge-genuine">✅ {pred.verdict}</div>', unsafe_allow_html=True)
            elif pred.verdict == "LIKELY SYNTHETIC":
                st.markdown(f'<div class="badge-synthetic">🚨 {pred.verdict}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-uncertain">⚠️ {pred.verdict}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # 2. Confidence Gauge
            st.plotly_chart(create_confidence_gauge(pred.calibrated_probability_bonafide, pred.verdict), use_container_width=True)

            # 3. Precision HUD Cards
            meta = pred.metadata or {}
            orig_sr = meta.get("orig_sample_rate", SAMPLE_RATE)
            bit_depth = meta.get("bit_depth", "16-bit PCM")
            peak_db = meta.get("peak_level_db", 0.0)
            snr_db = meta.get("snr_estimate_db", 30.0)

            st.markdown(
                f"""
                <div class="hud-grid">
                    <div class="hud-tile">
                        <div class="hud-tile-title">Sample Rate</div>
                        <div class="hud-tile-value">{SAMPLE_RATE:,} Hz</div>
                        <div class="hud-tile-sub">Orig: {orig_sr:,} Hz</div>
                    </div>
                    <div class="hud-tile">
                        <div class="hud-tile-title">Bit Depth</div>
                        <div class="hud-tile-value">{bit_depth}</div>
                        <div class="hud-tile-sub">Mono Channel</div>
                    </div>
                    <div class="hud-tile">
                        <div class="hud-tile-title">Peak Level</div>
                        <div class="hud-tile-value">{peak_db:+.1f} dB</div>
                        <div class="hud-tile-sub">dBFS</div>
                    </div>
                    <div class="hud-tile">
                        <div class="hud-tile-title">Est. SNR</div>
                        <div class="hud-tile-value">{snr_db:.1f} dB</div>
                        <div class="hud-tile-sub">Dynamic Range</div>
                    </div>
                    <div class="hud-tile">
                        <div class="hud-tile-title">Inference Time</div>
                        <div class="hud-tile-value">{pred.inference_latency_ms:.1f} ms</div>
                        <div class="hud-tile-sub">{pred.num_windows_analyzed} Window(s)</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 4. Detected Anomaly Highlights
            if pred.anomaly_timestamps:
                st.markdown("##### ⚠️ Detected Anomaly Timestamp Segments")
                for anom in pred.anomaly_timestamps:
                    tag_class = "anomaly-tag-critical" if anom.get("severity") == "CRITICAL" else "anomaly-tag-high"
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 6px; font-size: 0.85rem;">
                            <span class="{tag_class}">WINDOW {anom.get('window_idx', 0)} [{anom.get('start_sec', 0.0):.2f}s - {anom.get('end_sec', 0.0):.2f}s]</span>
                            &nbsp; <strong>Anomaly Score: {anom.get('anomaly_score', 0.0)*100:.1f}%</strong> — {anom.get('anomaly_type', 'Neural Vocoder Clue')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # 5. Technical Details Collapsible
            if sidebar_state.get("show_tech", True):
                with st.expander("🔬 Comprehensive Forensic Diagnostics", expanded=False):
                    st.json(pred.to_dict())

            # 6. Forensic Reports Export (JSON + Text Certificate)
            col_rep_json, col_rep_txt = st.columns(2)
            with col_rep_json:
                st.download_button(
                    label="📥 Export Report (JSON)",
                    data=json.dumps(pred.to_dict(), indent=2),
                    file_name=f"forensic_report_{pred.file_sha256[:8]}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col_rep_txt:
                cert_text = f"""================================================================================
VERIVOX ACOUSTIC FORENSIC EVIDENCE CERTIFICATE
================================================================================
Timestamp (UTC):       {pred.timestamp_utc}
Cryptographic SHA-256: {pred.file_sha256}
Operating Screener:    {pred.model_name}
Feature Mode:          {pred.feature_type}
Duration:              {pred.duration_sec:.2f} seconds
Windows Analyzed:      {pred.num_windows_analyzed}

FORENSIC VERDICT:      {pred.verdict}
Confidence Level:      {pred.confidence_percentage:.2f}%
Authenticity Posterior:{pred.calibrated_probability_bonafide:.4f}
Synthetic Audio Score: {pred.synthetic_score:.4f}
Decision Threshold:    {pred.decision_threshold:.4f}

ACOUSTIC PROPERTIES:
- Sample Rate:         {SAMPLE_RATE} Hz (Original: {orig_sr} Hz)
- Quantization:        {bit_depth}
- Peak Amplitude:      {peak_db:+.2f} dBFS
- Estimated SNR:       {snr_db:.1f} dB

ANOMALY AUDIT:
- Detected Anomalies:  {len(pred.anomaly_timestamps)}
================================================================================
Notice: This report is generated deterministically by VeriVox Acoustic Forensics.
Corroborate with expert forensic ear-witness testimony and channel metadata.
================================================================================
"""
                st.download_button(
                    label="📄 Export Certificate (TXT)",
                    data=cert_text,
                    file_name=f"verivox_certificate_{pred.file_sha256[:8]}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Inference error: {e}")
    else:
        st.info("Select a sample clip or upload an audio file to begin forensic screening.")

# Visual Evidence Section
st.markdown("---")
st.subheader("3. Visual Forensic Evidence & Spectral Heatmaps")

if temp_audio_path is not None and Path(temp_audio_path).exists():
    res = load_and_validate_audio(temp_audio_path, target_sr=SAMPLE_RATE)
    if res.is_valid and res.waveform is not None:
        wave_np = res.waveform.numpy()
        anom_list = pred.anomaly_timestamps if "pred" in locals() else None

        tab1, tab2, tab3 = st.tabs(["Attention Waveform Heatmap", "Log-Mel Spectrogram", "LFCC Cepstrogram"])

        with tab1:
            st.caption("Temporal attention heat-strip highlights regions where the model concentrated its classification decision.")
            if "pred" in locals():
                fig_attn = create_attention_waveform_plot(
                    wave_np,
                    pred.attention_weights,
                    sr=SAMPLE_RATE,
                    anomaly_segments=anom_list,
                )
                st.plotly_chart(fig_attn, use_container_width=True)

        with tab2:
            st.caption("Log-Mel Spectrogram (80 Mel bands, 0-8000 Hz). Look for unnatural high-frequency cuts or unnatural harmonic decay.")
            feat_logmel = predictor.logmel_extractor(res.waveform).numpy()
            fig_mel = create_spectrogram_heatmap(
                feat_logmel,
                title="Log-Mel Spectrogram (80 Bands)",
                y_title="Mel Frequency Band",
                duration_sec=res.duration_sec,
                anomaly_segments=anom_list,
            )
            st.plotly_chart(fig_mel, use_container_width=True)

        with tab3:
            st.caption("LFCC Matrix (60 coefficients with velocity and acceleration deltas). Linear filterbanks reveal phase discontinuities.")
            feat_lfcc = predictor.lfcc_extractor(res.waveform).numpy()
            fig_lfcc = create_spectrogram_heatmap(
                feat_lfcc,
                title="LFCC (Static + Delta + Delta-Delta)",
                y_title="Cepstral Coefficient",
                duration_sec=res.duration_sec,
                anomaly_segments=anom_list,
            )
            st.plotly_chart(fig_lfcc, use_container_width=True)

st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>Notice:</strong> High-frequency harmonic degradation typical of MP3 and Opus lossy compression may suppress deepfake artifacts. Always cross-reference with lossless uncompressed captures when available.
    </div>
    """,
    unsafe_allow_html=True,
)
