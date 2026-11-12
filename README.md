# VeriVox: Acoustic Forensics & Synthetic Speech Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2+-3178C6.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VeriVox** is a modular deepfake audio detection system and interactive forensic laboratory application designed to discriminate authentic human speech (*bona fide*) from synthetic speech and voice conversion deepfakes (*spoof*).

Developed by the **VeriVox Research Team**, Department of Information Technology, Indian Institute of Information Technology Allahabad.

---

## 1. Forensic Web Application ("Analysis Lab")

The application provides a scientific signal intelligence console rather than a generic SaaS landing page. It features dark obsidian chassis styling (`#0B0F12`), a signature Phosphor Lime CRT trace (`#22E575`), 1px graticule gridlines, monospace tabular figures (`JetBrains Mono`), and dual-encoded status badges for accessibility (contrast $\ge 4.5:1$).

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ [⌖] ACOUSTIC FORENSICS LAB   ::   SIGNAL INTELLIGENCE CONSOLE v2.4                         [● SYSTEM ONLINE]│
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ [ 1. ANALYSIS LAB ]                                                                      DEV: MPS (Apple)    │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Forensic Workstation Capabilities

- **Signal Ingestion Dropzone**: Drag-and-drop audio upload, native file browser, microphone recording with live Web Audio API VU level meter, and instant evaluation sample selector.
- **Real-Time SSE Pipeline Telemetry**: Streams actual backend progress events via Server-Sent Events with precise elapsed millisecond timings:
  $$\text{Validating} \rightarrow \text{Resampling} \rightarrow \text{Spectral Extraction} \rightarrow \text{Multi-Model Scoring} \rightarrow \text{L2 Meta-Fusion} \rightarrow \text{Synthesis}$$
- **Multi-Lane Temporal Signal Inspector**:
  - *Lane 1 (Oscillogram)*: HTML5 Canvas rendering peak envelope min/max pairs with draggable playhead.
  - *Lane 2 (CRNN Attention Heatstrip)*: Aligned frame-level self-attention salience showing which temporal moments drove the neural decision. Hovering inspects local time and frame weight.
  - *Lane 3 (Dual Spectrogram Heatmap)*: Canvas 2D spectral energy map toggleable between 60-dim LFCC and 80-dim Log-Mel with synchronized playhead tracking.
- **Calibrated Verdict Gauge**: Radial arc meter displaying calibrated probability $P(\text{Genuine})$, explicit uncertainty band $[40\%, 60\%]$, decision threshold ($0.1795$), and diagnostic summary.
- **Multi-Model Decision Axis**: Comparative horizontal score axis plotting CRNN-LFCC, CRNN-LogMel, GMM, SVM, Random Forest, and Logistic Regression on a unified axis with stacking weights and agreement indicators.
- **Printable Forensic Report**: Press <kbd>P</kbd> or click "Export Report" to generate an official one-page Forensic Examination Certificate via `@media print`.

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

## 3. Automated Test Suite (114 / 114 Passing)

Run the full pytest suite across API, neural models, classical baselines, metrics, leakage assertions, and provenance guards:
```bash
# Run all 114 unit and integration tests
pytest

# Or via Makefile:
make test
```

Test coverage includes:
- `tests/test_api.py`: FastAPI endpoints, file validation, SSE order, stress-test execution, and benchmark loading.
- `tests/test_api_crnn_parity.py`: Parity assertions between direct model inference and API endpoints (max abs diff < 1e-4).
- `tests/test_crnn.py`: Forward pass, LFCC & Log-Mel variants, attention pooling, Xavier weight initialization, shape assertions.
- `tests/test_classical.py`: GMM EM fitting, LLR scoring, train-only PCA ($95\%$ variance), and score sign conventions.
- `tests/test_stacking.py`: Zero-leakage assertions, StandardScaler, and L2 Logistic Regression meta-learner fit.
- `tests/test_metrics.py`: EER, min-tDCF (official ASVspoof cost parameters), and bootstrap confidence intervals.
- `tests/test_leakage.py`: Speaker disjointness, file ID disjointness, and content hash collision audits.
- `tests/test_provenance.py`: Rigorous provenance validation across all 34 metric JSON files, enforcing manifest hashes, git commit, and zero synthetic data.
- `tests/test_parity_check.py`: Offline cached vs online raw audio pipeline parity (0.00 pts ΔEER).

---

## 4. Empirical Benchmarks & Scientific Evaluation

> [!NOTE]
> **Status: FROZEN & VERIFIED (Nov 9, 2026)**  
> All reported numbers are computed by deterministic evaluation scripts on the full **ASVspoof 2019 Logical Access (LA)** evaluation partition ($N = 71{,}237$ trials) and unconstrained **In-the-Wild** dataset ($N = 25{,}589$ clips). All quantitative claims are audited and cryptographically tied to ground-truth JSON files in `results/metrics/` and verified via `scripts/verify_report_numbers.py` (139/139 claims verified). Zero synthetic or smoke-test data is present in active results.

### 4.1 Primary System: CRNN-LFCC Performance
The primary countermeasure is **CRNN-LFCC** (2D CNN + 2-layer BiGRU + Self-Attention pooling over 60-dimensional Linear Frequency Cepstral Coefficients). Multi-seed training stability was evaluated across three distinct random seeds:

| Architecture / Seed | Dev EER (%) | Eval EER (%) | Eval min-tDCF | Eval ROC-AUC | Parameters | Throughput (CPU p50) | Checkpoint Hash |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **CRNN-LFCC (Seed 42)** | 1.15% | **6.70%** | **0.2143** | **0.9800** | 1,439,778 | 10.28 ms (97.3 fps) | `f0da23...` |
| **CRNN-LFCC (Seed 43 - Dev Selected)** | **1.02%** | 7.61% | 0.2177 | 0.9784 | 1,439,778 | 10.28 ms (97.3 fps) | `e09c93...` |
| **CRNN-LFCC (Seed 44)** | 1.22% | 6.98% | 0.2178 | 0.9817 | 1,439,778 | 10.28 ms (97.3 fps) | `32b2e8...` |
| **CRNN-LFCC (3-Seed Mean ± Std)** | **1.13% ± 0.08%** | **7.10% ± 0.38%** | **0.2166 ± 0.0022** | **0.9800 ± 0.0017** | 1,439,778 | 10.28 ms (97.3 fps) | — |

*Evaluation Protocol: Full ASVspoof 2019 LA Eval partition ($N = 71{,}237$). Metrics: Equal Error Rate (EER), official minimum tandem Detection Cost Function (min-tDCF, $P_{spoof}=0.05, C_{miss}=1, C_{fa}=10$).*

### 4.2 Classical ML Baselines vs. Deep Neural System
Classical acoustic baselines benchmark conventional feature modeling against the deep CRNN:

| Model Architecture | Input Features | Dev EER (%) | Eval EER (%) | Eval ROC-AUC | Eval min-tDCF | Applied Threshold |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **CRNN-LFCC (Primary)** | LFCC (60-dim) | **1.15%** | **6.70%** | **0.9800** | **0.2143** | -1.0513 |
| **Classical GMM (64-comp)** | LFCC (60-dim unpooled) | 8.07% | 11.46% | 0.9544 | 0.3572 | 0.5711 |
| **Random Forest (200 trees)** | LFCC (120-dim pooled) | 10.09% | 17.52% | 0.8977 | 0.4695 | 0.1285 |
| **Support Vector Machine (RBF)** | LFCC (30-dim PCA) | 11.97% | 19.64% | 0.8932 | 0.5119 | 0.0504 |
| **Logistic Regression ($L_2$)** | LFCC (30-dim PCA) | 20.77% | 23.94% | 0.8459 | 0.6284 | 0.4507 |

### 4.3 Key Controlled Ablations
1. **Feature Ablation (Linear Filterbanks vs. Mel Scale)**:
   * **CRNN-LFCC**: **6.70%** Eval EER | 0.2143 min-tDCF
   * **CRNN-LogMel**: **10.28%** Eval EER | 0.2691 min-tDCF
   * *Acoustic Finding*: Linear frequency spacing preserves high-frequency spectral cues and phase artifacts in upper bands ($>4000\text{ Hz}$) that are compressed and obscured by logarithmic Mel-scale binning.
2. **Ensemble Ablation (Single Model vs. Score-Level Fusion)**:
   * **Stacked Ensemble** (Non-Negative $L_2$ Logistic Regression on Dev): **10.74%** Eval EER (ASVspoof) | **63.21%** EER (In-the-Wild).
   * *Acoustic Finding*: The meta-learner overfits to the clean Dev distribution and degrades generalization. CRNN-LFCC is deployed as the standalone primary system.

### 4.4 Forensic Audits & Critical Deployment Vulnerabilities
Comprehensive vulnerability testing informs our **"Pilot with Guardrails"** deployment recommendation:
- **The Silence Shortcut**: A trivial classifier trained on silence duration alone achieves **18.36% EER** ($AUC = 0.8872$). Trimming silence degrades CRNN EER by **+21.55 percentage points** (from $9.14\%$ to $30.69\%$), and swapping silence between pairs flips decisions in **34.85%** of trials.
  * *Deployment Mitigation*: Front-end Voice Activity Detection (VAD) is mandatory before inference.
- **Cross-Domain Domain Shift (In-the-Wild, $N = 25{,}589$)**:
  * Unconstrained real-world deepfakes yield an optimal EER of **44.73%** and ROC-AUC of **0.5638**. Polarity inversion diagnostic confirms standard polarity is mathematically sound ($AUC = 0.4362 < 0.50$). Performance drop is driven by genuine acoustic channel/codec divergence.
- **Acoustic Noise Sensitivity**: Additive Gaussian noise at 0 dB SNR increases EER to **48.60%**.

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
| <kbd>P</kbd> | Export printable one-page Forensic Examination Certificate |

---

## 7. Provenance, Code Attribution & Academic Integrity

For complete software provenance, external code attributions, mathematical lineage, and license compatibility matrices, see [SOURCES.md](SOURCES.md).

- **Original Implementations**: CRNN architecture with multi-head self-attention pooling, score-level stacking meta-learner, Platt scaling calibrator, FastAPI streaming service, and React/Canvas temporal forensic console.
- **Attributed Baselines & Standards**:
  - ASVspoof 2019 official evaluation metric toolkit (`compute_tDCF_legacy`, `compute_eer`): Sahidullah et al., Kinnunen et al. (BSD 3-Clause).
  - Linear Frequency Cepstral Coefficients (LFCC) filterbank & DCT-II formulation: Sahidullah et al. (2015).
  - 64-component Gaussian Mixture Model baseline: Todisco et al. (2019).
  - Stacking Generalization: Wolpert (1992).
- **License**: MIT License. All dependencies and external algorithms are permissively licensed (MIT, BSD-3-Clause, Apache-2.0) with zero copyleft (GPL/AGPL) conflicts.

---

## 8. Forensic Limitations & Ethical Disclosures

1. **Lossy Codec Quantization**: Low-bitrate compression ($\le 16\text{ kbps}$ AAC/MP3) induces spectral roll-off and quantization smearing, shifting scores toward the uncertainty band.
2. **Short Utterance Context**: Speech clips shorter than 1.0s lack sufficient acoustic window context for recurrent GRU temporal modeling.
3. **Out-of-Domain Diffusion Vocoders**: Novel generative diffusion architectures unrepresented in ASVspoof 2019 training data may exhibit distinct phase signatures.
4. **Ethical Boundaries**: Designed for defensive media forensics and integrity verification. Predictions represent probabilistic likelihood estimates, not definitive legal proof.
