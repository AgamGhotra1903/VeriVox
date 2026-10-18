"""Shared sidebar component for all Streamlit pages with synchronized state.

Academic Project Team:
- Paras Bansal (IIT2024221)
- Jashan Goyal (IIT2024223)
- Agam Ghotra (IIT2024226)
- Chunit Bansal (IIT2024233)
"""

from pathlib import Path
import streamlit as st


def render_sidebar() -> dict:
    """Renders persistent sidebar across all multipage views and synchronizes session state."""
    # Custom CSS injection
    css_path = Path("app/assets/style.css")
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Sidebar Brand Header
    st.sidebar.markdown(
        """
        <div style="padding-bottom: 14px; border-bottom: 1px solid var(--border-subtle); margin-bottom: 16px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <h2 style="margin: 0; font-size: 1.5rem; font-weight: 900; color: var(--text-white); display: flex; align-items: center; gap: 8px;">
                    <span>🎙️</span> VeriVox
                </h2>
                <span class="vv-status-pip" style="font-size: 0.62rem; padding: 2px 8px;">
                    <span class="vv-status-dot" style="width: 5px; height: 5px;"></span>
                    CALIBRATED
                </span>
            </div>
            <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 4px;">Acoustic Forensics &amp; Speech Authentication</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize defaults in session state if not present
    if "active_model" not in st.session_state:
        st.session_state["active_model"] = "CRNN (Best)"
    if "feature_representation" not in st.session_state:
        st.session_state["feature_representation"] = "LFCC (60 Coefficients)"
    if "threshold_override" not in st.session_state:
        st.session_state["threshold_override"] = 5.23
    if "show_tech" not in st.session_state:
        st.session_state["show_tech"] = True

    model_options = [
        "CRNN (Best)",
        "Random Forest (LFCC)",
        "RBF-SVM (LFCC)",
        "Logistic Regression (LFCC)",
    ]
    current_model_idx = (
        model_options.index(st.session_state["active_model"])
        if st.session_state["active_model"] in model_options
        else 0
    )

    st.sidebar.markdown(
        '<p style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.08em; margin: 0 0 6px 0;">Active Screener</p>',
        unsafe_allow_html=True,
    )
    active_model = st.sidebar.selectbox(
        "Active Model",
        model_options,
        index=current_model_idx,
        key="sidebar_model_select",
        label_visibility="collapsed",
    )
    st.session_state["active_model"] = active_model

    feature_options = [
        "LFCC (60 Coefficients)",
        "Log-Mel (80 Bands)",
        "Combined (Late Fusion)",
    ]
    current_feat_idx = (
        feature_options.index(st.session_state["feature_representation"])
        if st.session_state["feature_representation"] in feature_options
        else 0
    )

    st.sidebar.markdown(
        '<p style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.08em; margin: 14px 0 6px 0;">Feature Mode</p>',
        unsafe_allow_html=True,
    )
    feature_choice = st.sidebar.selectbox(
        "Feature Representation",
        feature_options,
        index=current_feat_idx,
        key="sidebar_feat_select",
        label_visibility="collapsed",
    )
    st.session_state["feature_representation"] = feature_choice

    st.sidebar.markdown(
        '<p style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.08em; margin: 14px 0 2px 0;">Operating Threshold</p>',
        unsafe_allow_html=True,
    )
    threshold_override = st.sidebar.slider(
        "Operating Decision Threshold",
        min_value=-5.0,
        max_value=15.0,
        value=float(st.session_state["threshold_override"]),
        step=0.1,
        help="Default threshold calibrated on ASVspoof 2019 LA development split at Equal Error Rate (EER = 5.23).",
        key="sidebar_thresh_slider",
        label_visibility="collapsed",
    )
    st.session_state["threshold_override"] = threshold_override

    st.sidebar.caption("Calibrated EER threshold: **θ = 5.23**")

    show_tech = st.sidebar.checkbox(
        "Show Technical Diagnostics",
        value=bool(st.session_state["show_tech"]),
        key="sidebar_tech_cb",
    )
    st.session_state["show_tech"] = show_tech

    # Team footer
    st.sidebar.markdown(
        """
        <div style="margin-top: 28px; padding-top: 14px; border-top: 1px solid var(--border-subtle); font-size: 0.78rem; color: var(--text-muted);">
            <div style="font-weight: 700; color: var(--text-secondary); margin-bottom: 6px; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;">Engineering Team</div>
            <div>• <strong style="color: var(--accent-sky);">Paras Bansal</strong> <span style="color: var(--text-dim);">(IIT2024221)</span></div>
            <div>• <strong style="color: var(--accent-sky);">Jashan Goyal</strong> <span style="color: var(--text-dim);">(IIT2024223)</span></div>
            <div>• <strong style="color: var(--accent-sky);">Agam Ghotra</strong> <span style="color: var(--text-dim);">(IIT2024226)</span></div>
            <div>• <strong style="color: var(--accent-sky);">Chunit Bansal</strong> <span style="color: var(--text-dim);">(IIT2024233)</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return {
        "active_model": active_model,
        "feature_representation": feature_choice,
        "threshold_override": threshold_override,
        "show_tech": show_tech,
    }
