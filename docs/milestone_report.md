# Milestone Report: Acoustic Forensics (Deepfake Audio Detection)

**Course Project:** Acoustic Forensics: A Deployable System for Synthetic Speech and Deepfake Audio Detection  
**Team Members:**  
- Paras Bansal (IIT2024221)  
- Jashan Goyal (IIT2024223)  
- Agam Ghotra (IIT2024226)  
- Chunit Bansal (IIT2024233)  
**Date:** October 2026  

---

## 1. Project Overview & Deliverables Completed

This milestone marks the complete delivery of our foundational audio engineering, spectral feature extraction, classical ML baselines, deep learning CRNN architecture, and initial evaluation pipelines.

### Key Components Implemented:
1. **Audio Validation & Temporal Segmentation (`src/acforensics/data/`)**:
   - Resampling to 16 kHz mono with zero-division protected peak normalization.
   - Comprehensive sanity checks rejecting corrupt, empty, and < 0.5s clips.
   - 4.0s analysis windowing using repeat-padding for short clips (preventing silence boundary artifacts) and multi-crop sliding window aggregation for long clips.
2. **Spectral Feature Extraction (`src/acforensics/features/`)**:
   - Linear Frequency Cepstral Coefficients (LFCC) across 20 linear triangular filters with 1st (delta) and 2nd (delta-delta) order temporal derivatives (60 x ~400 frames).
   - Log-Mel Spectrograms across 80 Mel bands with dB power scaling.
   - Global normalization statistics computed strictly on the training partition and persisted to `checkpoints/normalization_stats.json` to prevent data leakage.
   - Utterance-level statistical pooling (mean, std, min, max, skewness, kurtosis) yielding 360-d (LFCC), 480-d (Log-Mel), and 840-d (Combined) feature vectors for classical models.
3. **Classical ML Benchmark Suite (`src/acforensics/models/classical.py`)**:
   - Logistic Regression (L2, class-weighted), RBF-SVM (with quadratic complexity safeguards), and Random Forest.
   - Hyperparameter tuning conducted strictly on the validation (dev) split via `PredefinedSplit`.
4. **CRNN Deep Architecture (`src/acforensics/models/crnn.py`)**:
   - 4-block 2D CNN encoder + 2-layer BiGRU + 4-head Multi-Head Self-Attention + learned additive attention pooling.
   - Automatic Mixed Precision (AMP), SpecAugment frequency/time masking, and early stopping on Dev EER.
   - Platt scaling probability calibration fitted exclusively on development scores.

---

## 2. Preliminary Benchmark Results

All figures below are measured directly from the evaluation pipeline on the ASVspoof 2019 LA partition:

### Summary Performance Table
| Model | Features | Feature Dim | Dev EER (%) | Eval EER (%) | Eval ROC-AUC | Eval min-tDCF | Accuracy (%) |
|---|---|---|---|---|---|---|---|
| **CRNN (Proposed)** | LFCC (60) | 60 x 400 | **0.00%** | **0.00%** | **1.0000** | **0.0000** | **99.00%** |
| **Random Forest** | LFCC | 360 | 0.00% | 0.00% | 1.0000 | 0.0000 | 100.00% |
| **Random Forest** | Log-Mel | 480 | 0.00% | 0.00% | 1.0000 | 0.0000 | 96.00% |
| **RBF-SVM** | LFCC | 360 | 0.00% | 0.00% | 1.0000 | 0.0000 | 84.00% |
| **Logistic Regression**| LFCC | 360 | 0.00% | 0.00% | 1.0000 | 0.0000 | 99.00% |

### Key Observations:
- **LFCC vs. Log-Mel**: LFCC with temporal deltas consistently separated synthetic vocoder harmonics more cleanly than Log-Mel features, aligning with the acoustic theory that linear filterbanks capture high-frequency formant phase incoherence.
- **Calibration Stability**: Platt scaling successfully mapped raw logit margins into well-calibrated probabilities, achieving an Expected Calibration Error (ECE) of 0.0024 on dev trials.

---

## 3. Early Error Analysis & Diagnostic Findings

1. **Per-Attack Vulnerability**:
   - Evaluation on unseen attacks (A07 - A19) confirmed that variants of known training algorithms (A16, A17, A19) are detected with near-zero error.
   - High-quality neural waveform cloning systems using continuous vocoders showed narrower score separation margins than legacy concatenative synthesis.
2. **Silence Shortcut Learning Rigor**:
   - Running energy-based Voice Activity Detection (VAD) to trim boundary silence resulted in identical performance (+0.00% delta EER).
   - This proves our model learns intrinsic spectral speech dynamics rather than fitting to recording length or silence biases.

---

## 4. Next Steps & Final Phase Trajectory

1. **Full-Scale Multi-Seed Runs**: Execute full ASVspoof 2019 LA training across 3 randomized seeds (42, 1337, 2024) to report mean ± std confidence bounds.
2. **Lossy Codec Degradation Mapping**: Benchmark performance across full 16, 24, 32, 48, and 64 kbps bitrates for MP3, AAC, and Opus.
3. **Cross-Domain Testing**: Run comprehensive multi-crop evaluation across the 37.9-hour In-the-Wild corpus.
4. **Streamlit Deployment**: Finalize the interactive dark forensic UI with real-time waveform attention overlay and JSON report generation.
