# Acoustic Forensics: UI Design System & Dashboard Documentation

**Acoustic Forensics** features a product-grade, accessible, dark glassmorphic Streamlit dashboard tailored for high-stakes synthetic speech and voice deepfake screening.

---

## 🎨 Design System & Visual Tokens

The user interface adheres to strict typographic, tonal, and geometric constraints defined in `app/theme.py` and `app/style.css`:

### Color Palette (WCAG AA Compliant)
| Token | Hex Value | Semantic Usage |
|---|---|---|
| `COLOR_BG` | `#0B0F14` | Global application background |
| `COLOR_SURFACE` | `#121821` | Base container / card background |
| `COLOR_ELEVATED` | `#18202B` | Hover states, active tabs, modal callouts |
| `COLOR_BORDER` | `#1F2937` | 1px clean container dividers |
| `COLOR_TEXT` | `#E6EDF3` | Primary body and header text |
| `COLOR_MUTED` | `#8B98A9` | Secondary captions, axis labels, metadata |
| `COLOR_ACCENT` | `#38BDF8` | Interactive brand highlights, progress bars |
| `COLOR_GENUINE` | `#34D399` | Authentic human speech indicator (Emerald) |
| `COLOR_SYNTHETIC` | `#F87171` | Synthetic voice / deepfake indicator (Coral) |
| `COLOR_UNCERTAIN` | `#FBBF24` | Inconclusive / abstention band (Amber) |

### Typography & Spacing
- **Font Stack:** Inter, `-apple-system`, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif
- **Grid:** 8px base grid spacing
- **Borders:** 14px container radius (`border-radius: 14px`), 1px crisp borders
- **Color Independence:** All verdicts pair explicit icons (`✔`, `✖`, `?`) and descriptive labels ("GENUINE SPEECH", "SYNTHETIC SPEECH", "INCONCLUSIVE") alongside colors to guarantee accessibility for colorblind analysts.

---

## 🧭 Information Architecture (5 Core Tabs)

1. **Analyze (Single-Clip Screener):**
   - **Input Section:** Curated authentic/TTS/VC sample presets, local file uploader (WAV, FLAC, MP3, OGG, M4A), and live microphone recording.
   - **Interactive Controls:** Model architecture selector and operating threshold sensitivity slider (`0.10` to `0.90`).
   - **Hero Verdict Card:** Color-coded status badge, calibrated authenticity probability ring (0–100%), and 3-zone score spectrum meter with an abstention zone `[0.35, 0.65]`.
   - **Reliability Audit:** Deterministic pre-flight checks (duration, clipping ratio, frame-level SNR, spectral rolloff, silence ratio, and multi-model agreement).
   - **Evidence Panel:** Interactive Plotly tabs for downsampled waveform with self-attention salience overlay, STFT spectrogram saliency map, temporal multi-window localization, architectural model agreement table, and raw JSON technical schema.
   - **Export:** Instant downloads for complete forensic JSON reports and plaintext summary certificates.

2. **Batch (High-Throughput Screening):**
   - Multi-file or ZIP archive upload with automated extraction.
   - Live progress indicator with per-file status updates.
   - Score distribution histogram with threshold marker.
   - Sortable results table and one-click CSV export.

3. **Stress Lab (Robustness & Codec Vulnerability):**
   - Real-time lossy re-encoding (MP3, AAC, Opus at 16–64 kbps).
   - Additive Gaussian noise injection (5–30 dB SNR) and room impulse response reverberation.
   - Side-by-side clean vs degraded verdict comparison cards, dual audio players, and quantitative score impact metrics.

4. **Insights (Scientific Benchmarks & Diagnostics):**
   - Performance metrics HUD (Eval EER, ROC-AUC, Min-tDCF, Cross-Domain ITW EER).
   - Codec Robustness curves (EER vs bitrate across MP3, AAC, and Opus).
   - Architectural ablation horizontal bar charts.
   - Official ASVspoof 2019 LA evaluation benchmark table with 95% bootstrap confidence intervals.

5. **About (System Documentation & Ethics):**
   - Forensic problem formulation and signal processing pipeline overview.
   - Dual-branch CRNN model architecture breakdown.
   - Ethical use disclaimer, limitations on unseen generators, and VCTK heritage notes.
   - Academic team credits and primary research citations.

---

## 🚀 Running the App Locally

To start the Streamlit dashboard:

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Run via makefile
make demo

# Or directly with Streamlit:
streamlit run app/streamlit_app.py --server.port 8501
```

---

## ☁️ Deployment Guide

### Deploying to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Log into [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, branch (`main`), and set the main file path to:
   ```
   app/streamlit_app.py
   ```
4. In Advanced Settings, ensure Python 3.10+ is selected.
5. Click **Deploy**. Dependencies in `requirements.txt` will be installed automatically.

### Deploying to Hugging Face Spaces (Streamlit SDK)
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) and choose **Streamlit** as the SDK.
2. In the repository settings or `README.md` frontmatter, specify:
   ```yaml
   ---
   title: Acoustic Forensics Deepfake Screener
   emoji: 🎙️
   colorFrom: blue
   colorTo: gray
   sdk: streamlit
   app_file: app/streamlit_app.py
   pinned: false
   ---
   ```
3. Push the codebase. All checkpoints in `checkpoints/` and dependencies in `requirements.txt` are self-contained and will run on standard free CPU spaces.
