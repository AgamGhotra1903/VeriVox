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

    st.sidebar.markdown("## 🎙️ **VeriVox**")
    st.sidebar.caption("Acoustic Forensics & Synthetic Speech Detection")

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

    active_model = st.sidebar.selectbox(
        "Active Model",
        model_options,
        index=current_model_idx,
        key="sidebar_model_select",
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

    feature_choice = st.sidebar.selectbox(
        "Feature Representation",
        feature_options,
        index=current_feat_idx,
        key="sidebar_feat_select",
    )
    st.session_state["feature_representation"] = feature_choice

    threshold_override = st.sidebar.slider(
        "Operating Decision Threshold",
        min_value=-5.0,
        max_value=15.0,
        value=float(st.session_state["threshold_override"]),
        step=0.1,
        help="Default threshold calibrated on ASVspoof 2019 LA development split at Equal Error Rate (EER).",
        key="sidebar_thresh_slider",
    )
    st.session_state["threshold_override"] = threshold_override

    show_tech = st.sidebar.checkbox(
        "Show Forensic Technical Details",
        value=bool(st.session_state["show_tech"]),
        key="sidebar_tech_cb",
    )
    st.session_state["show_tech"] = show_tech

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Team Contributions:**
        - **Paras Bansal**: Data curation & normalization
        - **Jashan Goyal**: Classical ML baselines
        - **Agam Ghotra**: CRNN & Inference engine
        - **Chunit Bansal**: Robustness, metrics & demo
        """
    )

    return {
        "active_model": active_model,
        "feature_representation": feature_choice,
        "threshold_override": threshold_override,
        "show_tech": show_tech,
    }
