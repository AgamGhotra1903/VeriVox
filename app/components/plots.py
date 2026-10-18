"""Interactive Plotly visualizers for forensic inspection and benchmark dashboards.

Author: Chunit Bansal (IIT2024223)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_confidence_gauge(probability_bonafide: float, verdict: str) -> go.Figure:
    """Builds a radial gauge displaying calibrated authenticity probability."""
    if verdict == "LIKELY GENUINE":
        bar_color = "#22C55E"
    elif verdict == "LIKELY SYNTHETIC":
        bar_color = "#EF4444"
    else:
        bar_color = "#F59E0B"

    val_pct = probability_bonafide * 100.0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val_pct,
        number={"suffix": "%", "font": {"size": 36, "color": "#FFFFFF"}},
        title={"text": "Authenticity Score (Calibrated)", "font": {"size": 16, "color": "#94A3B8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#161B22",
            "borderwidth": 1,
            "bordercolor": "#30363D",
            "steps": [
                {"range": [0, 35], "color": "rgba(239, 68, 68, 0.25)"},
                {"range": [35, 65], "color": "rgba(245, 158, 11, 0.25)"},
                {"range": [65, 100], "color": "rgba(34, 197, 94, 0.25)"},
            ],
            "threshold": {
                "line": {"color": "#FFFFFF", "width": 3},
                "thickness": 0.8,
                "value": val_pct,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_attention_waveform_plot(
    waveform: np.ndarray,
    attention_weights: np.ndarray,
    sr: int = 16000,
    anomaly_segments: list[dict] = None,
) -> go.Figure:
    """Creates dual-layer plot: audio waveform with overlaid attention heat-strip and anomaly highlights."""
    duration = len(waveform) / sr
    time_wave = np.linspace(0, duration, len(waveform))

    # Downsample waveform for snappy Plotly rendering
    stride = max(1, len(waveform) // 4000)
    time_sub = time_wave[::stride]
    wave_sub = waveform[::stride]

    # Attention weights: shape (T_frames,)
    time_attn = np.linspace(0, duration, len(attention_weights))

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Waveform Amplitude with Anomaly Regions", "Frame Attention Weights (Model Decision Focus)"],
    )

    # Waveform trace
    fig.add_trace(
        go.Scatter(
            x=time_sub,
            y=wave_sub,
            mode="lines",
            line=dict(color="#06B6D4", width=1.2),
            name="Waveform",
            hoverinfo="x+y",
        ),
        row=1, col=1,
    )

    # Attention heat-strip trace
    fig.add_trace(
        go.Scatter(
            x=time_attn,
            y=attention_weights,
            fill="tozeroy",
            mode="lines",
            line=dict(color="#F59E0B", width=2),
            name="Attention Weight α_t",
            hoverinfo="x+y",
        ),
        row=2, col=1,
    )

    # Add anomaly segment highlight boxes if present
    if anomaly_segments:
        for seg in anomaly_segments:
            s_start = seg.get("start_sec", 0.0)
            s_end = seg.get("end_sec", duration)
            severity = seg.get("severity", "HIGH")
            box_color = "rgba(239, 68, 68, 0.22)" if severity == "CRITICAL" else "rgba(245, 158, 11, 0.18)"
            fig.add_vrect(
                x0=s_start,
                x1=s_end,
                fillcolor=box_color,
                layer="below",
                line_width=1,
                line_dash="dot",
                line_color="#EF4444" if severity == "CRITICAL" else "#F59E0B",
                annotation_text=f"Anomaly ({severity})",
                annotation_position="top left",
                annotation_font=dict(size=10, color="#FCA5A5" if severity == "CRITICAL" else "#FDE68A"),
                row=1, col=1,
            )

    fig.update_layout(
        paper_bgcolor="#0B0F17",
        plot_bgcolor="#111827",
        font=dict(color="#F1F5F9"),
        height=360,
        margin=dict(l=30, r=30, t=40, b=30),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1, gridcolor="#1F2937")
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, gridcolor="#1F2937")
    fig.update_yaxes(title_text="Weight α_t", row=2, col=1, gridcolor="#1F2937")

    return fig


def create_spectrogram_heatmap(
    spec_matrix: np.ndarray,
    title: str = "Spectrogram Heatmap",
    y_title: str = "Bands",
    duration_sec: float = None,
    anomaly_segments: list[dict] = None,
) -> go.Figure:
    """Builds interactive heatmap for LFCC or Log-Mel features with calibrated time axis."""
    T = spec_matrix.shape[1] if spec_matrix.ndim > 1 else 100
    x_axis = np.linspace(0, duration_sec, T) if duration_sec else np.arange(T)
    x_label = "Time (seconds)" if duration_sec else "Time Frames (~10ms hop)"

    fig = go.Figure(data=go.Heatmap(
        z=spec_matrix,
        x=x_axis,
        colorscale="Viridis",
        colorbar=dict(title="Energy / dB"),
    ))

    if anomaly_segments and duration_sec:
        for seg in anomaly_segments:
            s_start = seg.get("start_sec", 0.0)
            s_end = seg.get("end_sec", duration_sec)
            fig.add_vrect(
                x0=s_start,
                x1=s_end,
                fillcolor="rgba(239, 68, 68, 0.25)",
                layer="above",
                line_width=1,
                line_color="#EF4444",
                annotation_text="Anomaly",
                annotation_font=dict(size=9, color="#F87171"),
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#F1F5F9")),
        xaxis=dict(title=x_label, gridcolor="#1F2937"),
        yaxis=dict(title=y_title, gridcolor="#1F2937"),
        paper_bgcolor="#0B0F17",
        plot_bgcolor="#111827",
        font=dict(color="#F1F5F9"),
        height=320,
        margin=dict(l=30, r=30, t=40, b=30),
    )
    return fig


def create_robustness_chart(df_rob: pd.DataFrame) -> go.Figure:
    """Builds interactive Plotly dark chart for Lossy Codec Degradation."""
    fig = go.Figure()

    # Baseline line at Clean EER (0.00%)
    fig.add_trace(go.Scatter(
        x=[16, 64],
        y=[0.0, 0.0],
        mode="lines",
        name="Clean Baseline (0.00%)",
        line=dict(color="#94A3B8", dash="dash", width=1.5),
        hoverinfo="skip",
    ))

    color_map = {
        "MP3": "#EF4444",   # Rose
        "AAC": "#38BDF8",   # Sky
        "OPUS": "#10B981",  # Emerald
    }

    for codec in ["MP3", "AAC", "OPUS"]:
        sub = df_rob[df_rob["Codec"].str.upper() == codec]
        if not sub.empty:
            group_sorted = sub.sort_values("Bitrate (kbps)")
            color = color_map.get(codec, "#A78BFA")
            fig.add_trace(go.Scatter(
                x=group_sorted["Bitrate (kbps)"],
                y=group_sorted["Compressed EER (%)"],
                mode="lines+markers",
                name=f"{codec}",
                line=dict(color=color, width=2.5),
                marker=dict(size=8, color=color, symbol="circle"),
                customdata=group_sorted[["Accuracy (%)", "Abs Degradation (pts)"]].to_numpy(),
                hovertemplate="<b>%{data.name}</b><br>Bitrate: %{x} kbps<br>Compressed EER: %{y:.2f}%<br>Accuracy: %{customdata[0]:.1f}%<br>Degradation: %{customdata[1]} pts<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text="Lossy Codec Degradation: EER (%) vs. Bitrate", font=dict(size=14, color="#F8FAFC")),
        xaxis=dict(title="Bitrate (kbps)", gridcolor="#1F2937", tickvals=[16, 24, 32, 48, 64]),
        yaxis=dict(title="Equal Error Rate (EER %)", gridcolor="#1F2937"),
        paper_bgcolor="#0B0F17",
        plot_bgcolor="#111827",
        font=dict(color="#F1F5F9"),
        height=360,
        margin=dict(l=35, r=35, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(17, 24, 39, 0.7)",
            bordercolor="#1F2937",
            borderwidth=1,
        ),
    )
    return fig


def create_per_attack_chart(attack_dict: dict[str, float]) -> go.Figure:
    """Builds interactive Plotly dark bar chart for Per-Attack EER Breakdown."""
    attacks = sorted(attack_dict.keys())
    eers = [float(attack_dict[a]) for a in attacks]

    # Known training variants: A16, A17, A19 -> amber, others -> cyan
    colors = ["#F59E0B" if a in ["A16", "A17", "A19"] else "#06B6D4" for a in attacks]

    fig = go.Figure(go.Bar(
        x=attacks,
        y=eers,
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.2)", width=1),
        ),
        hovertemplate="<b>Attack %{x}</b><br>EER: %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Per-Attack EER Breakdown (Amber = Known Training Variant)", font=dict(size=14, color="#F8FAFC")),
        xaxis=dict(title="Attack Algorithm (A07–A19)", gridcolor="#1F2937"),
        yaxis=dict(title="Equal Error Rate (%)", gridcolor="#1F2937", rangemode="tozero"),
        paper_bgcolor="#0B0F17",
        plot_bgcolor="#111827",
        font=dict(color="#F1F5F9"),
        height=320,
        margin=dict(l=35, r=35, t=45, b=35),
    )
    return fig
