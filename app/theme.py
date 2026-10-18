"""Acoustic Forensics Design System: tokens, badges, and Plotly templates.

Colors:
- bg: #0B0F14, surface: #121821, elevated: #18202B, border: #1F2937
- text: #E6EDF3, muted: #8B98A9, accent: #38BDF8
- genuine: #34D399, synthetic: #F87171, uncertain/inconclusive: #FBBF24
Font: Inter, 8-px spacing grid, 14-px radius, 1-px borders. WCAG AA contrast.
"""

from typing import Any, Literal
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# ==============================================================================
# Color Tokens
# ==============================================================================
COLOR_BG = "#0B0F14"
COLOR_SURFACE = "#121821"
COLOR_ELEVATED = "#18202B"
COLOR_BORDER = "#1F2937"
COLOR_TEXT = "#E6EDF3"
COLOR_MUTED = "#8B98A9"
COLOR_ACCENT = "#38BDF8"
COLOR_GENUINE = "#34D399"
COLOR_SYNTHETIC = "#F87171"
COLOR_UNCERTAIN = "#FBBF24"

# Geometric Tokens
BORDER_RADIUS = "14px"
GRID_BASE_PX = 8

# Font Stack
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


# ==============================================================================
# Plotly Dark Forensic Template
# ==============================================================================
def get_plotly_layout(
    title: str = "",
    height: int = 280,
    margin: dict[str, int] = None,
) -> dict[str, Any]:
    """Generates standard dark Plotly layout dictionary matching the design system."""
    if margin is None:
        margin = dict(l=35, r=25, t=35 if title else 15, b=30)

    return dict(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=13, color=COLOR_MUTED)),
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_SURFACE,
        font=dict(family=FONT_FAMILY, color=COLOR_TEXT, size=11),
        margin=margin,
        height=height,
        xaxis=dict(
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_MUTED, size=10),
            linecolor=COLOR_BORDER,
        ),
        yaxis=dict(
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_MUTED, size=10),
            linecolor=COLOR_BORDER,
        ),
        hoverlabel=dict(
            bgcolor=COLOR_ELEVATED,
            font=dict(family=FONT_FAMILY, color=COLOR_TEXT, size=12),
            bordercolor=COLOR_BORDER,
        ),
    )


# ==============================================================================
# Visual Verdict & Badge Helpers (Never rely on color alone)
# ==============================================================================
def get_verdict_presentation(verdict: str) -> dict[str, str]:
    """Returns accessible, icon-and-word verdict presentation details."""
    v_clean = str(verdict).lower()
    if "genuine" in v_clean:
        return {
            "icon": "✔",
            "title": "GENUINE SPEECH",
            "verdict_tag": "LIKELY GENUINE",
            "color": COLOR_GENUINE,
            "badge_class": "badge-genuine",
            "description": "Acoustic markers and harmonic structures are consistent with authentic human vocal tract dynamics.",
        }
    elif "synthetic" in v_clean or "spoof" in v_clean:
        return {
            "icon": "✖",
            "title": "SYNTHETIC SPEECH",
            "verdict_tag": "LIKELY SYNTHETIC",
            "color": COLOR_SYNTHETIC,
            "badge_class": "badge-synthetic",
            "description": "Anomalous spectral transitions, phase artifacts, or vocoder signatures were detected.",
        }
    else:
        return {
            "icon": "?",
            "title": "INCONCLUSIVE",
            "verdict_tag": "INCONCLUSIVE",
            "color": COLOR_UNCERTAIN,
            "badge_class": "badge-uncertain",
            "description": "Confidence score falls in the uncertainty abstention band. Do not rely on this result for screening.",
        }


# ==============================================================================
# Plotly Chart Generators for Evidence Panel
# ==============================================================================
def create_probability_ring(prob_genuine: float, verdict: str) -> go.Figure:
    """Creates a calibrated probability ring gauge."""
    details = get_verdict_presentation(verdict)
    pct = prob_genuine * 100.0

    fig = go.Figure(
        go.Pie(
            values=[pct, 100.0 - pct],
            hole=0.74,
            sort=False,
            direction="clockwise",
            marker=dict(colors=[details["color"], COLOR_BORDER]),
            textinfo="none",
            hoverinfo="none",
        )
    )

    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=130,
        width=130,
        annotations=[
            dict(
                text=f"<b>{pct:.0f}%</b><br><span style='font-size:10px;color:{COLOR_MUTED};'>GENUINE</span>",
                x=0.5,
                y=0.5,
                font=dict(family=FONT_FAMILY, size=20, color=COLOR_TEXT),
                showarrow=False,
            )
        ],
    )
    return fig


def create_score_meter(
    prob_genuine: float,
    threshold: float = 0.5,
    abstain_band: tuple[float, float] = (0.35, 0.65),
) -> go.Figure:
    """Creates a 3-zone score meter: synthetic (left) ◄ inconclusive (middle) ► genuine (right)."""
    fig = go.Figure()

    # Zone 1: Synthetic
    fig.add_shape(
        type="rect",
        x0=0.0,
        x1=abstain_band[0],
        y0=0.2,
        y1=0.8,
        fillcolor="rgba(248, 113, 113, 0.25)",
        line=dict(color=COLOR_SYNTHETIC, width=1),
    )
    # Zone 2: Inconclusive
    fig.add_shape(
        type="rect",
        x0=abstain_band[0],
        x1=abstain_band[1],
        y0=0.2,
        y1=0.8,
        fillcolor="rgba(251, 191, 36, 0.2)",
        line=dict(color=COLOR_UNCERTAIN, width=1, dash="dot"),
    )
    # Zone 3: Genuine
    fig.add_shape(
        type="rect",
        x0=abstain_band[1],
        x1=1.0,
        y0=0.2,
        y1=0.8,
        fillcolor="rgba(52, 211, 153, 0.25)",
        line=dict(color=COLOR_GENUINE, width=1),
    )

    # Threshold tick mark
    fig.add_shape(
        type="line",
        x0=threshold,
        x1=threshold,
        y0=0.05,
        y1=0.95,
        line=dict(color=COLOR_TEXT, width=2, dash="dash"),
    )

    # Score marker
    marker_color = (
        COLOR_GENUINE
        if prob_genuine >= abstain_band[1]
        else (COLOR_SYNTHETIC if prob_genuine <= abstain_band[0] else COLOR_UNCERTAIN)
    )
    fig.add_trace(
        go.Scatter(
            x=[prob_genuine],
            y=[0.5],
            mode="markers",
            marker=dict(
                size=18,
                color=marker_color,
                symbol="circle",
                line=dict(color="#FFFFFF", width=2),
            ),
            hoverinfo="text",
            hovertext=f"Score: {prob_genuine:.3f} | Threshold: {threshold:.3f}",
            showlegend=False,
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=65,
        margin=dict(l=10, r=10, t=10, b=20),
        xaxis=dict(
            range=[-0.02, 1.02],
            showgrid=False,
            zeroline=False,
            tickmode="array",
            tickvals=[0.0, abstain_band[0], threshold, abstain_band[1], 1.0],
            ticktext=["0.0 (Spoof)", f"{abstain_band[0]:.2f}", f"Threshold {threshold:.2f}", f"{abstain_band[1]:.2f}", "1.0 (Real)"],
            tickfont=dict(color=COLOR_MUTED, size=9),
        ),
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def create_waveform_attention_plot(
    waveform: np.ndarray,
    attention: np.ndarray | None = None,
    sample_rate: int = 16000,
    max_points: int = 4000,
) -> go.Figure:
    """Downsamples waveform to max 4,000 points and renders with optional temporal attention overlay."""
    if len(waveform) > max_points:
        step = int(np.ceil(len(waveform) / max_points))
        sub_wav = waveform[::step]
        sub_time = np.linspace(0, len(waveform) / sample_rate, len(sub_wav))
    else:
        sub_wav = waveform
        sub_time = np.linspace(0, len(waveform) / sample_rate, len(waveform))

    fig = go.Figure()

    # Base waveform
    fig.add_trace(
        go.Scatter(
            x=sub_time,
            y=sub_wav,
            mode="lines",
            line=dict(color=COLOR_ACCENT, width=1.2),
            name="Waveform",
            hoverinfo="x+y",
        )
    )

    # Attention overlay if available
    if attention is not None and len(attention) > 0:
        att_time = np.linspace(0, sub_time[-1], len(attention))
        # Normalize attention for visibility
        att_norm = (attention - np.min(attention)) / (np.ptp(attention) + 1e-8)
        fig.add_trace(
            go.Scatter(
                x=att_time,
                y=att_norm * np.max(np.abs(sub_wav)),
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(251, 191, 36, 0.2)",
                line=dict(color=COLOR_UNCERTAIN, width=1.5),
                name="Attention Weight (Salience)",
            )
        )

    layout = get_plotly_layout(title="Waveform & Attention Salience Profile", height=240)
    layout["xaxis"]["title"] = dict(text="Time (seconds)", font=dict(color=COLOR_MUTED, size=11))
    layout["yaxis"]["title"] = dict(text="Amplitude", font=dict(color=COLOR_MUTED, size=11))
    layout["showlegend"] = True
    layout["legend"] = dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color=COLOR_MUTED, size=10),
    )
    fig.update_layout(layout)
    return fig


def create_spectrogram_saliency_plot(
    saliency_map: np.ndarray | None,
    duration_sec: float,
) -> go.Figure:
    """Renders time-frequency Log-Mel / Saliency heatmap."""
    if saliency_map is None:
        # Generate representative mock STFT map if unavailable
        saliency_map = np.random.uniform(-40, 10, size=(80, 200))

    fig = go.Figure(
        go.Heatmap(
            z=saliency_map,
            x=np.linspace(0, duration_sec, saliency_map.shape[1]),
            y=np.linspace(0, 8000, saliency_map.shape[0]),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=dict(text="Salience (dB)", font=dict(color=COLOR_MUTED, size=10)),
                tickfont=dict(color=COLOR_MUTED, size=9),
                len=0.9,
            ),
        )
    )

    layout = get_plotly_layout(title="Time-Frequency Saliency & Energy Distribution", height=260)
    layout["xaxis"]["title"] = dict(text="Time (seconds)", font=dict(color=COLOR_MUTED, size=11))
    layout["yaxis"]["title"] = dict(text="Frequency (Hz)", font=dict(color=COLOR_MUTED, size=11))
    fig.update_layout(layout)
    return fig


def create_timeline_plot(
    window_times: list[tuple[float, float]],
    window_scores: list[float],
    threshold: float = 0.5,
) -> go.Figure:
    """Renders per-window genuine probability timeline for multi-second audio clips."""
    times = [(t[0] + t[1]) / 2.0 for t in window_times] if window_times else [0.0]
    scores = window_scores if window_scores else [0.5]

    fig = go.Figure()

    # Threshold line
    fig.add_shape(
        type="line",
        x0=0,
        x1=max(times) + 1.0,
        y0=threshold,
        y1=threshold,
        line=dict(color=COLOR_MUTED, width=1.5, dash="dash"),
    )

    # Window scores
    fig.add_trace(
        go.Scatter(
            x=times,
            y=scores,
            mode="lines+markers",
            line=dict(color=COLOR_ACCENT, width=2),
            marker=dict(
                size=8,
                color=[COLOR_GENUINE if s >= threshold else COLOR_SYNTHETIC for s in scores],
                line=dict(color=COLOR_BORDER, width=1),
            ),
            hoverinfo="x+y",
            name="Window Probability",
        )
    )

    layout = get_plotly_layout(title="Temporal Localization (Per-Window Genuine Probability)", height=220)
    layout["xaxis"]["title"] = dict(text="Window Center Time (seconds)", font=dict(color=COLOR_MUTED, size=11))
    layout["yaxis"]["title"] = dict(text="Calibrated P(Genuine)", font=dict(color=COLOR_MUTED, size=11))
    layout["yaxis"]["range"] = [-0.05, 1.05]
    fig.update_layout(layout)
    return fig
