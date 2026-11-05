# VeriVox: Acoustic Forensics & Synthetic Speech Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2+-3178C6.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VeriVox** is a modular deepfake audio detection system and interactive forensic laboratory application designed to discriminate authentic human speech (*bona fide*) from synthetic speech and voice conversion deepfakes (*spoof*).

Developed for an advanced machine learning course project by:
- **Paras Bansal** (IIT2024221): Dataset curation, protocol parsing, audio I/O validation, feature extraction (LFCC, Log-Mel, Normalization).
- **Jashan Goyal** (IIT2024223): Classical ML baselines (LR, RBF-SVM, RF), dev-set hyperparameter tuning, statistical pooling.
- **Agam Ghotra** (IIT2024226): CRNN architecture (CNN + BiGRU + Self-Attention + Attention Pooling), regularized training engine, GMM baseline, score-level stacked ensemble, Platt calibration, FastAPI streaming service & React forensic instrument.
- **Chunit Bansal** (IIT2024233): Evaluation metrics (EER, min-tDCF, ROC-AUC), ffmpeg codec stress testing, In-The-Wild cross-domain testing, error audit.

---

## 1. Forensic Web Application ("Signal Intelligence Lab")

The application provides a scientific signal intelligence console rather than a generic SaaS landing page. It features dark obsidian chassis styling (`#0B0F12`), a signature Phosphor Lime CRT trace (`#22E575`), 1px graticule gridlines, monospace tabular figures (`JetBrains Mono`), and dual-encoded status badges for accessibility (contrast $\ge 4.5:1$).

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ [⌖] ACOUSTIC FORENSICS LAB   ::   SIGNAL INTELLIGENCE CONSOLE v2.4                         [● SYSTEM ONLINE]│
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ [ 1. ANALYSE ]      [ 2. STRESS TEST ]      [ 3. BENCHMARKS ]      [ 4. METHODOLOGY ]    DEV: MPS (Apple)    │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### The 4 Application Screens

1. **Screen 1: Analyse (Main Forensic Workstation)**
   - **Signal Dropzone**: Drag-and-drop audio upload, native file browser, microphone recording with live Web Audio API VU level meter, and instant reference sample selector.
   - **Real-Time SSE Pipeline Telemetry**: Streams actual backend progress events via Server-Sent Events with precise elapsed millisecond timings:
     $$\text{Validating} \rightarrow \text{Resampling} \rightarrow \text{Spectral Extraction} \rightarrow \text{Multi-Model Scoring} \rightarrow \text{L2 Meta-Fusion} \rightarrow \text{Synthesis}$$
   - **Multi-Lane Temporal Signal Inspector**:
     - *Lane 1 (Oscillogram)*: HTML5 Canvas rendering peak envelope min/max pairs with draggable playhead.
     - *Lane 2 (CRNN Attention Heatstrip)*: Aligned frame-level self-attention salience showing which temporal moments drove the neural decision. Hovering inspects local time and frame weight.
     - *Lane 3 (Dual Spectrogram Heatmap)*: Canvas 2D spectral energy map toggleable between 60-dim LFCC and 80-dim Log-Mel with synchronized playhead tracking.
   - **Calibrated Verdict Gauge**: Radial arc meter displaying calibrated probability $P(\text{Genuine})$, explicit uncertainty band $[40\%, 60\%]$, decision threshold ($0.1795$), and diagnostic summary.
   - **Multi-Model Decision Axis**: Comparative horizontal score axis plotting CRNN-LFCC, CRNN-LogMel, GMM, SVM, Random Forest, and Logistic Regression on a unified axis with stacking weights and agreement indicators.

2. **Screen 2: Stress Test (Lossy Codec Compression Harness)**
   - Re-encodes the loaded audio across **MP3**, **AAC**, and **Opus** at **16, 32, and 64 kbps** via native `ffmpeg`.
   - Generates a 9-cell experimental matrix comparing verdict stability, score shifts ($\Delta$), and high-frequency cutoff roll-off.

3. **Screen 3: Benchmarks & Empirical Validation**
   - **Master Results Table**: Dynamic evaluation metrics (Dev/Eval EER, ROC-AUC, min-tDCF, and operating thresholds) across all base models and fusion.
   - **Interactive Canvas ROC Curves**: Smooth multi-model ROC curves with interactive model toggles and diagonal random-guess baseline.
   - **Controlled Ablation Matrices**: Feature ablation (LFCC vs. Log-Mel vs. Both), Architecture ablation (CNN vs. BiGRU vs. Self-Attention), and Regularization ablation (Dropout/AdamW).
   - **Per-Attack Evaluation (A07–A19)**: Eval set breakdown across unseen speech synthesis and voice conversion attacks.
   - **Cross-Domain Transfer (In-the-Wild)**: Zero-shot performance on unconstrained YouTube deepfake speech.
   - **CSV Export**: One-click export for any benchmark matrix.

4. **Screen 4: Methodology & Scientific Rigor**
   - Interactive end-to-end signal flow breakdown.
   - Leakage-safe partitioning guarantees (Train $\rightarrow$ Dev $\rightarrow$ Eval).
   - Forensic limitations & known failure modes (lossy compression roll-off, short clips $<1.0\text{s}$, unseen diffusion vocoders).

5. **Printable Forensic Report**
   - Press <kbd>P</kbd> or click "Export Report" to generate an official one-page Forensic Examination Certificate via `@media print`.

---

## 2. Quickstart & Execution Commands

### Local Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/AgamGhotra1903/VeriVox.git
cd VeriVox

# 2. Initialize Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 3. Install frontend dependencies
cd frontend && npm install --legacy-peer-deps && cd ..
```

### Launching the Application
```bash
# Option A: Run the unified production app (FastAPI + React on port 8000)
make app
# Open: http://localhost:8000

# Option B: Run development mode (FastAPI API on 8000 + Vite HMR on 3000)
# Terminal 1 (Backend):
make api
# Terminal 2 (Frontend):
make web
# Open: http://localhost:3000
```

### Docker Deployment
```bash
# Build and launch multi-container stack (api on :8000, web on :3000)
make docker-up

# Stop containers
make docker-down
```

---

## 3. Automated Test Suite (86 / 86 Passing)

Run the full pytest suite across API, neural models, classical baselines, metrics, and leakage assertions:
```bash
# Run all 86 unit and integration tests
pytest

# Or via Makefile:
make test
```

Test coverage includes:
- `tests/test_api.py`: FastAPI endpoints, file validation, SSE order, stress-test execution, and benchmark loading.
- `tests/test_crnn.py`: Forward pass, LFCC & Log-Mel variants, attention pooling, Xavier weight initialization, shape assertions.
- `tests/test_classical.py`: GMM EM fitting, LLR scoring, train-only PCA ($95\%$ variance), and score sign conventions.
- `tests/test_stacking.py`: Zero-leakage assertions, StandardScaler, and L2 Logistic Regression meta-learner fit.
- `tests/test_metrics.py`: EER, min-tDCF (official ASVspoof cost parameters), and bootstrap confidence intervals.
- `tests/test_leakage.py`: Speaker disjointness, file ID disjointness, and content hash collision audits.

---

## 4. Measured Benchmark Results

All metrics are measured directly from the evaluation pipeline.

### Table 1: Base Models vs. Fusion Performance (ASVspoof 2019 LA)
| System | Type | Dev EER (%) | Eval EER (%) | Eval ROC-AUC | Eval min-tDCF | Operating Threshold |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Single: CRNN-LFCC** | Deep Neural Base | 0.00% | 0.00% | 1.0000 | 0.0000 | +19.2115 (from Dev) |
| **Single: CRNN-LogMel** | Deep Neural Base | 75.00% | 61.25% | 0.3012 | 1.0000 | +1.6352 (from Dev) |
| **Single: GMM** | Classical Spectral (64 comp) | 0.00% | 0.00% | 1.0000 | 0.0000 | +100.6544 (from Dev) |
| **Single: SVM (RBF + PCA)** | Classical Baseline | 0.00% | 0.00% | 1.0000 | 0.0000 | +0.6752 (from Dev) |
| **Single: Random Forest** | Classical Baseline | 0.00% | 0.00% | 1.0000 | 0.0000 | +0.3640 (from Dev) |
| **Single: Logistic Regression**| Classical Baseline | 0.00% | 0.00% | 1.0000 | 0.0000 | +1.0370 (from Dev) |
| **Baseline Mean Fusion** | Unweighted Z-Scores | 0.00% | 0.00% | 1.0000 | 0.0000 | +0.8767 (from Dev) |
| **Stacked Ensemble (Proposed)**| L2 Logistic Regression | **0.00%** | **0.00%** | **1.0000** | **0.0000** | **+0.1795 (from Dev)** |

### Table 2: Meta-Learner Base Model Contributions
| Base Model | Meta-Learner Weight | Absolute Contribution | Role in Ensemble |
|---|:---:|:---:|---|
| **CRNN-LFCC** | `+0.0521` | 0.0521 | Formant & sub-band spectral anomaly detection |
| **GMM** | `+0.0521` | 0.0521 | Frame-level acoustic generative density prior |
| **SVM (RBF + PCA)** | `+0.0514` | 0.0514 | Non-linear margin in low-rank PCA space |
| **Logistic Regression** | `+0.0512` | 0.0512 | Linear calibration anchor |
| **Random Forest** | `+0.0489` | 0.0489 | Ensemble partition tree regularization |
| **CRNN-LogMel** | `-0.0288` | 0.0288 | Down-weighted complementary spectrogram branch |

### Table 3: Feature Ablation (LFCC vs. Log-Mel vs. Both)
| Feature Representation | Input Dimension | Dev EER (%) | Eval EER (%) | Eval ROC-AUC | Eval min-tDCF |
|---|---|:---:|:---:|:---:|:---:|
| **LFCC (60-dim)** | 60 (20 stat + 20 delta + 20 delta-delta) | **0.00%** | **0.00%** | **1.0000** | **0.0000** |
| **Log-Mel (80-dim)** | 80 Mel filterbank energy bins | 67.50% | 60.00% | 0.4356 | 1.0000 |
| **Both (LFCC + Log-Mel Fusion)** | 140 combined spectral channels | 26.25% | 13.75% | 0.9625 | 0.2315 |

---

## 5. API Reference

The backend exposes an asynchronous REST and Server-Sent Events API:

- `POST /api/analyze`: Multipart audio upload (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`). Validates payload size ($\le 25\text{MB}$) and audio duration ($\le 60\text{s}$). Returns `{ "job_id", "stream_url", "result_url" }`.
- `GET /api/analyze/{id}/events`: Server-Sent Events (SSE) stream emitting pipeline stages:
  `validating` $\rightarrow$ `resampling` $\rightarrow$ `feature_extraction` $\rightarrow$ `model_scores` $\rightarrow$ `ensemble` $\rightarrow$ `done`.
- `GET /api/result/{id}`: Full forensic payload including verdict, calibrated $P(\text{Bona Fide})$, decision threshold, 6-model score breakdown, downsampled waveform envelope (800 min/max peaks), temporal attention weights, and downsampled LFCC/Log-Mel spectrogram matrices.
- `POST /api/stress-test`: Form parameter `job_id`. Re-encodes audio across MP3, AAC, and Opus at 16, 32, 64 kbps using `ffmpeg` and returns a comparative shift matrix.
- `GET /api/benchmarks`: Serves measured experimental results parsed dynamically from `results/tables/*.csv` and `results/metrics/*.json`.
- `GET /api/samples`: Lists bundled reference clips (`sample_bonafide_human_speech.wav`, `sample_spoof_neural_tts.wav`, `sample_spoof_vocoder_vc.wav`).
- `GET /api/health`: Operational device (`mps`/`cuda`/`cpu`), loaded models list, and memory consumption.

---

## 6. Keyboard Shortcuts

| Key | Action |
|:---:|---|
| <kbd>U</kbd> | Open file browser to upload an audio clip |
| <kbd>Space</kbd> | Toggle audio playback / pause with moving playhead |
| <kbd>1</kbd> | Switch to **Analyse** Workstation |
| <kbd>2</kbd> | Switch to **Stress Test** Console |
| <kbd>3</kbd> | Switch to **Benchmarks** Dashboard |
| <kbd>4</kbd> | Switch to **Methodology** Documentation |
| <kbd>P</kbd> | Export printable one-page Forensic Examination Certificate |

---

## 7. Forensic Limitations & Ethical Disclosures

1. **Lossy Codec Quantization**: Low-bitrate compression ($\le 16\text{ kbps}$ AAC/MP3) induces spectral roll-off and quantization smearing, shifting scores toward the uncertainty band.
2. **Short Utterance Context**: Speech clips shorter than 1.0s lack sufficient acoustic window context for recurrent GRU temporal modeling.
3. **Out-of-Domain Diffusion Vocoders**: Novel generative diffusion architectures unrepresented in ASVspoof 2019 training data may exhibit distinct phase signatures.
4. **Ethical Boundaries**: Designed for defensive media forensics and integrity verification. Predictions represent probabilistic likelihood estimates, not definitive legal proof.
