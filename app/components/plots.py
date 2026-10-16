"""Interactive Plotly visualizers for forensic inspection and benchmark dashboards.

Author: Chunit Bansal (IIT2024223)
"""

import numpy as np
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
) -> go.Figure:
    """Creates dual-layer plot: audio waveform with overlaid attention heat-strip."""
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
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Waveform Amplitude", "Frame Attention Weights (Model Decision Focus)"],
    )

    # Waveform trace
    fig.add_trace(
        go.Scatter(
            x=time_sub,
            y=wave_sub,
            mode="lines",
            line=dict(color="#2DD4BF", width=1),
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

    fig.update_layout(
        paper_bgcolor="#161B22",
        plot_bgcolor="#0E1117",
        font=dict(color="#E2E8F0"),
        height=350,
        margin=dict(l=30, r=30, t=40, b=30),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1, gridcolor="#30363D")
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, gridcolor="#30363D")
    fig.update_yaxes(title_text="Weight α_t", row=2, col=1, gridcolor="#30363D")

    return fig


def create_spectrogram_heatmap(
    spec_matrix: np.ndarray,
    title: str = "Spectrogram Heatmap",
    y_title: str = "Bands",
) -> go.Figure:
    """Builds interactive heatmap for LFCC or Log-Mel features."""
    fig = go.Figure(data=go.Heatmap(
        z=spec_matrix,
        colorscale="Viridis",
        colorbar=dict(title="Energy / dB"),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#E2E8F0")),
        xaxis=dict(title="Time Frames (~10ms hop)", gridcolor="#30363D"),
        yaxis=dict(title=y_title, gridcolor="#30363D"),
        paper_bgcolor="#161B22",
        plot_bgcolor="#0E1117",
        font=dict(color="#E2E8F0"),
        height=300,
        margin=dict(l=30, r=30, t=40, b=30),
    )
    return fig
