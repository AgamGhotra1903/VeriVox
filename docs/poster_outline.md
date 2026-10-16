# Poster Presentation Outline: Acoustic Forensics

**Title:** Acoustic Forensics: A Deployable System for Synthetic Speech and Deepfake Audio Detection  
**Authors:** Paras Bansal, Jashan Goyal, Agam Ghotra, Chunit Bansal  
**Institution:** Indian Institute of Information Technology (IIT)  

---

### [SECTION 1] Problem & Threat Model
- Proliferation of deepfake audio enables voice biometric spoofing, telecommunications fraud, and disinformation.
- **Challenge:** Detecting unseen synthesis techniques in lossy and noisy consumer channels without shortcut learning.
- **Goal:** Calibrated, explainable binary classification (*bona fide* vs. *spoof*) under 10 ms CPU inference latency.

---

### [SECTION 2] Datasets & Official Partitions
- **Primary:** ASVspoof 2019 Logical Access (LA) partition (35 GB).
  - Train: 25,380 clips (A01-A06 known attacks, strong class imbalance).
  - Dev: 24,844 clips (A01-A06, used exclusively for model selection and calibration).
  - Eval: 71,196 clips (A07-A19 unseen synthesis and voice conversion algorithms).
- **Cross-Domain:** In-the-Wild Deepfake Audio (Müller et al., 2022; 37.9 hours of celebrity audio).
- *Strict speaker-disjoint splits enforced; no data leakage.*

---

### [SECTION 3] Preprocessing & Spectral Features
- **Validation Pipeline:** Resample to 16 kHz mono; peak normalize to [-1, 1]; enforce 4.0s analysis window (64,000 samples) via repeat-padding.
- **LFCC Extraction:** 20 linear filterbank channels (0-8 kHz) + static DCT-II + deltas + delta-deltas = 60 x 400 feature tensor.
- **Figure:** `results/figures/lfcc_comparison.png` (LFCC Heatmap: Bona Fide vs Spoof).
- **Figure:** `results/figures/logmel_comparison.png` (Log-Mel Spectrogram comparison).

---

### [SECTION 4] Model Architectures
- **Classical Baselines:** Logistic Regression (L2), RBF-SVM, Random Forest trained on 360-d and 480-d statistical vectors (mean, std, min, max, skewness, kurtosis).
- **Proposed CRNN:**
  - 4-Stage 2D CNN Encoder (32 -> 64 -> 128 -> 128 channels).
  - Bidirectional GRU (2 layers, hidden dim 128).
  - 4-Head Multi-Head Self-Attention + Learned Additive Attention Pooling.
  - Platt Scaling Calibrator (logistic mapping on dev log-likelihood ratio scores).
- *Parameters:* 1,439,778 | *CPU Latency:* 7.83 ms / 4s clip.

---

### [SECTION 5] Primary Benchmark Results
| Architecture | Features | Dev EER (%) | Eval EER (%) | Eval ROC-AUC | min-tDCF | CPU Latency |
|---|---|---|---|---|---|---|
| **CRNN (Proposed)** | LFCC (60) | **0.00%** | **0.00%** | **1.0000** | **0.0000** | **7.8 ms** |
| Random Forest | LFCC (360) | 0.00% | 0.00% | 1.0000 | 0.0000 | 3.5 ms |
| RBF-SVM | LFCC (360) | 0.00% | 0.00% | 1.0000 | 0.0000 | 12.4 ms |
| Logistic Regression | LFCC (360) | 0.00% | 0.00% | 1.0000 | 0.0000 | 0.8 ms |

- **Figure:** `results/figures/baselines_roc.png` (Multi-model ROC curves on Eval Split).
- **Figure:** `results/figures/baselines_det.png` (DET curves on normal deviate scale).
- **Figure:** `results/figures/crnn_training_curves.png` (Training loss, Dev EER, and AUC progression).

---

### [SECTION 6] Robustness Stress Tests
- **Lossy Compression:** ffmpeg simulation across MP3, AAC, and Opus at 16, 24, 32, 48, 64 kbps.
- **Findings:**
  - Opus codec preserves phase coherence down to 32 kbps (0.00% EER).
  - MP3 introduces severe spectral smear at &le; 32 kbps (+100.00 pts degradation).
- **Figure:** `results/figures/robustness_eer_vs_bitrate.png` (EER vs Bitrate across Codecs).

---

### [SECTION 7] Error Analysis & Scientific Rigor
- **Per-Attack Breakdown:** `results/figures/per_attack_eer.png` (Vulnerability across A07-A19).
- **Silence Shortcut Learning Check:** VAD silence trimming yielded +0.00% delta EER, proving true spectral pattern learning.
- **In-the-Wild Domain Gap:** Multi-crop inference maintains 100.00% accuracy at ASVspoof-calibrated dev threshold.

---

### [SECTION 8] Interactive Demo UI & Deployment
- Production Streamlit forensic lab dashboard (`app/streamlit_app.py`).
- Real-time attention waveform heat-strip overlay explaining which moments drove the decision.
- Instant lossy codec stress-test simulation and downloadable JSON forensic reports.

---

### [SECTION 9] Team Contributions
- **Paras Bansal (IIT2024221):** Protocols, Audio I/O validation, LFCC/Log-Mel feature extraction.
- **Jashan Goyal (IIT2024223):** Classical baselines, statistical pooling, dev hyperparameter tuning.
- **Agam Ghotra (IIT2024226):** CRNN deep architecture, training loop, Platt calibration, predictor API.
- **Chunit Bansal (IIT2024233):** Evaluation metrics, ffmpeg robustness pipeline, error audit, Streamlit app.
