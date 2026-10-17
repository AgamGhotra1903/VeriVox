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
        <div style="padding-bottom: 12px; border-bottom: 1px solid rgba(148, 163, 184, 0.15); margin-bottom: 14px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <h2 style="margin: 0; font-size: 1.55rem; font-weight: 800; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
                    <span>🎙️</span> VeriVox
                </h2>
                <span style="font-size: 0.68rem; font-weight: 700; background: rgba(16, 185, 129, 0.15); color: #34D399; padding: 2px 8px; border-radius: 9999px; border: 1px solid rgba(16, 185, 129, 0.3); font-family: 'JetBrains Mono', monospace;">CALIBRATED</span>
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">Acoustic Forensics & Speech Authentication</div>
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

    st.sidebar.markdown("<p style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.08em; margin: 0 0 6px 0;'>Active Screener</p>", unsafe_allow_html=True)
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

    st.sidebar.markdown("<p style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.08em; margin: 12px 0 6px 0;'>Feature Mode</p>", unsafe_allow_html=True)
    feature_choice = st.sidebar.selectbox(
        "Feature Representation",
        feature_options,
        index=current_feat_idx,
        key="sidebar_feat_select",
        label_visibility="collapsed",
    )
    st.session_state["feature_representation"] = feature_choice

    st.sidebar.markdown("<p style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.08em; margin: 12px 0 2px 0;'>Operating Decision Threshold</p>", unsafe_allow_html=True)
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

    st.sidebar.markdown(
        """
        <div style="margin-top: 24px; padding-top: 14px; border-top: 1px solid rgba(148, 163, 184, 0.15); font-size: 0.82rem; color: #94A3B8;">
            <div style="font-weight: 700; color: #E2E8F0; margin-bottom: 6px;">Academic Engineering Team:</div>
            <div>• <strong style="color: #38BDF8;">Paras Bansal</strong> (IIT2024221)</div>
            <div>• <strong style="color: #38BDF8;">Jashan Goyal</strong> (IIT2024223)</div>
            <div>• <strong style="color: #38BDF8;">Agam Ghotra</strong> (IIT2024226)</div>
            <div>• <strong style="color: #38BDF8;">Chunit Bansal</strong> (IIT2024233)</div>
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
