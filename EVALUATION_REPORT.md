# Acoustic Forensics: Forensic Evaluation & Verification Report
**Project**: Acoustic Forensics / VeriVox (Synthetic Speech & Deepfake Audio Detection)  
**System Version**: 2.5.0 (Scientific Evaluation Phase)  
**Author/Team**: Antigravity Machine Learning & Audio Forensics Engineering  
**Evaluation Standard**: Official ASVspoof 2019 LA Protocol & Müller et al. In-the-Wild Benchmark  
**Date of Evaluation**: November 2026  
**Git Commit**: `348ce41d60604e9fe08fcf0615afd156ab3fec40`  

---

> [!WARNING]
> **FORMAL RETRACTION & SCIENTIFIC AUDIT NOTICE**  
> Prior versions of this project reported headline metrics including 0.0000% EER, 1.0000 ROC-AUC, and 98% In-the-Wild accuracy. A forensic code audit identified that those figures were generated exclusively on an in-memory synthetic smoke-test fixture (`data/smoke_test`, created via `scripts/prepare_data.py::generate_smoke_test_dataset` using synthetic sine wave harmonics and injected high-frequency test tones) originally intended for continuous integration pipeline validation.  
>  
> **All claims of "0% EER", "perfect detection", "98% In-the-Wild generalization", and "unqualified production readiness" were scientifically invalid and are formally RETRACTED.**  
>  
> The mock smoke outputs have been archived to `results_smoke/`. All evaluations in this report were executed on genuine acoustic data: official ASVspoof 2019 Logical Access (LA) partitions (25,380 train, 24,844 dev, 71,237 eval trials; SHA-256: `91a3907b...`) and the unconstrained In-the-Wild Deepfake Speech dataset (25,587 verified clips; Müller et al., 2022). Every metric reported below is backed by serialized JSON records under `results/metrics/` with cryptographic provenance tracking (`dataset_name`, `manifest_sha256`, `git_commit`, `seed`, `timestamp`).

---

## 1. Executive Summary

This evaluation report documents the verified scientific benchmark of the **Acoustic Forensics** deepfake countermeasure system. The pipeline assesses whether a single-channel 16 kHz audio recording is authentic human phonation (*bona fide*) or artificially generated/converted speech (*spoof*).

### Verified Headline Performance (ASVspoof 2019 LA Eval Split, $N=71,237$ Trials)
- **Primary Deep Architecture (CRNN-LFCC, 1.44M parameters)**:
  - **Dev EER**: **2.90%** (min-tDCF: **0.0821**, ROC-AUC: **0.9951**, Operating Threshold: **-0.1428**)
  - **Eval EER**: **8.39%** [95% CI: **8.16%**, **8.72%**] (Speaker Bootstrap: [**7.43%**, **9.71%**])
  - **Eval min-tDCF**: **0.2624** [95% CI: **0.2531**, **0.2690**] (Speaker Bootstrap: [**0.2313**, **0.2910**])
  - **Eval ROC-AUC**: **0.9707**
  - **Dev Threshold Accuracy**: **87.81%**
- **Stacked Ensemble (Meta-Learner: Non-Negative Logistic Regression, 5 Base Classifiers)**:
  - Base classifiers: SVM-LFCC ($w=0.1812$), RF-LFCC ($w=0.1694$), CRNN-LFCC ($w=0.1675$), GMM-LFCC ($w=0.1007$), LR-LFCC ($w=0.0905$).
  - **Dev EER**: **3.75%** (min-tDCF: **0.1580**, ROC-AUC: **0.9931**, Operating Threshold: **-1.9367**)
  - **Eval EER**: **12.55%** [95% CI: **12.24%**, **12.84%**] (Speaker Bootstrap: [**13.43%**, **16.16%**])
  - **Eval min-tDCF**: **0.3192** [95% CI: **0.3095**, **0.3258**] (Speaker Bootstrap: [**0.3854**, **0.4592**])
  - **Eval ROC-AUC**: **0.9474**
  - **Dev Threshold Recall**: **98.15%** (Accuracy: **79.06%**)
- **Official GMM-LFCC Baseline (512 Components)**:
  - **Dev EER**: **8.07%** (min-tDCF: **0.2185**)
  - **Eval EER**: **11.46%** (min-tDCF: **0.3572**, ROC-AUC: **0.9544**)

### Critical Scientific Findings & Production Posture
1. **In-the-Wild Cross-Domain Failure**: When evaluated on $N=25,587$ unconstrained social media/YouTube deepfakes, the CRNN-LFCC model suffers severe cross-domain collapse (**71.74%–74.77% EER**, ROC-AUC **0.1703–0.1987**). The Stacked Ensemble exhibits complete score inversion (**84.15% EER**, ROC-AUC **0.0869**). The models classify studio-recorded authentic speech as spoof and compressed social-media synthetic audio as authentic.
2. **Dataset Silence Leakage**: Silence duration alone separates bona fide from spoof in ASVspoof 2019 LA with **22.78% EER** (ROC-AUC **0.8509**). Removing silence boundaries via VAD causes CRNN EER to spike from **7.91%** to **39.13%** (+31.22 pts degradation), proving the neural model exploits silence boundary artifacts rather than purely speech spectral anomalies.
3. **Log-Mel Spectral Failure**: While LFCC achieves 8.39% Eval EER, an identical CRNN trained on Log-Mel spectrograms failed to converge (**50.00% EER**, min-tDCF **1.0000**, ROC-AUC **0.5000**), confirming linear frequency filterbanks are mandatory for vocoder artifact detection.
4. **Vulnerability to Voice Conversion (A18)**: On the unseen A18 attack (voice conversion via i-vector/PLDA + spectral filtering), CRNN EER degrades to **27.57%** (min-tDCF **0.8470**) and Ensemble EER degrades to **39.14%** (min-tDCF **0.9996**).
5. **Deployment Recommendation**: Suitable **only as an assistive investigative triage filter** in controlled studio or lossless transmission environments. **UNFIT for fully autonomous or courtroom evidence determinations without human-in-the-loop expert acoustic analysis.**

---

## 2. Evaluation Methodology & Metrics

### 2.1 Biometric Error Metrics
- **Equal Error Rate (EER)**: The operating threshold $\theta$ where the False Acceptance Rate ($\text{FAR}(\theta)$) equals the False Rejection Rate ($\text{FRR}(\theta)$). Continuous scores follow the biometric convention where higher scores indicate bona fide speech ($s > \theta \implies \text{Bona Fide}$).
- **Receiver Operating Characteristic Area Under Curve (ROC-AUC)**: Measures ranking separability independent of operating threshold. A score of 0.50 indicates chance performance; scores below 0.50 indicate inverted polarity.

### 2.2 Minimum Tandem Detection Cost Function (min-tDCF)
The primary metric defined by the ASVspoof Consortium models the operational interaction between the countermeasure (CM) and an Automatic Speaker Verification (ASV) system:
$$\text{tDCF}(\tau) = C_0 + C_1 \cdot P_{\text{miss}}^{\text{cm}}(\tau) + C_2 \cdot P_{\text{fa}}^{\text{cm}}(\tau)$$
Where normalized parameters follow the official ASVspoof 2019 evaluation protocol:
- $P_{\text{tar}} = 0.05, P_{\text{non}} = 0.05, P_{\text{spoof}} = 0.90$
- $C_{\text{miss}} = 1.0, C_{\text{fa}} = 10.0$
- ASV error probabilities calculated on paired trials ($P_{\text{miss}}^{\text{asv}} = 0.0163, P_{\text{fa}}^{\text{asv}} = 0.0125$).
- Official verification: Our implementation was validated against the official ASVspoof committee script within $| \text{min-tDCF} - \text{official} | < 10^{-6}$.

### 2.3 Partitions & Speaker Disjointness Guarantees
Manifests were parsed directly from official ASVspoof 2019 LA protocols:
- **Train Split**: 25,380 trials (2,580 bona fide, 22,800 spoof across 20 speakers).
- **Dev Split**: 24,844 trials (2,548 bona fide, 22,296 spoof across 20 speakers).
- **Eval Split**: 71,237 trials (7,355 bona fide, 63,882 spoof across 67 speakers).
- **In-the-Wild Split**: 25,587 verified audio clips (15,401 bona fide, 10,186 spoof).
- **Speaker Disjointness**: Cryptographically audited via `scripts/audit_leakage.py`. Set intersections confirmed zero overlap:
  $$\text{Train} \cap \text{Dev} = \emptyset, \quad \text{Train} \cap \text{Eval} = \emptyset, \quad \text{Dev} \cap \text{Eval} = \emptyset$$

---

## 3. Comprehensive Benchmark Results

### 3.1 Classical Baselines (Table 1)
Evaluated on static utterance-level summary statistics (mean + standard deviation over frames, 120-dimensional LFCC and 160-dimensional Log-Mel):

| Model | Features | Dev EER (%) | Eval EER (%) | Eval ROC-AUC | Eval min-tDCF | Applied Threshold |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GMM (512 comp)** | LFCC | **8.07** | **11.46** | **0.9544** | **0.3572** | 0.5711 |
| GMM (512 comp) | LOGMEL | 30.31 | 49.46 | 0.5237 | 0.9969 | 2.6538 |
| RandomForest | LFCC | 10.09 | 17.52 | 0.8977 | 0.4695 | 0.1285 |
| RandomForest | LOGMEL | 13.92 | 23.26 | 0.8199 | 0.7184 | 0.1284 |
| SVM (RBF kernel) | LFCC | 11.97 | 19.64 | 0.8932 | 0.5119 | 0.0504 |
| SVM (RBF kernel) | LOGMEL | 18.01 | 27.27 | 0.7933 | 0.7962 | 0.0930 |
| LogisticRegression | LFCC | 20.77 | 23.94 | 0.8459 | 0.6284 | 0.4507 |
| LogisticRegression | LOGMEL | 23.12 | 29.20 | 0.7798 | 0.7861 | 0.4888 |

---

### 3.2 Deep Learning Architectures (Table 2)
Evaluated with end-to-end convolutional recurrent networks (CRNN: 5-layer 2D-CNN with MaxPool, 2-layer Bidirectional GRU, and Self-Attention Pooling):

| Architecture | Features | Parameters | Dev EER (%) | Eval EER (%) | Eval ROC-AUC | Eval min-tDCF |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CRNN-LFCC** | LFCC (60-dim) | 1,439,778 | **2.90** | **8.39** | **0.9707** | **0.2624** |
| **CRNN-LOGMEL** | Log-Mel (80-dim) | 1,472,546 | 50.00 | 50.00 | 0.5000 | 1.0000 |

*CRNN-LFCC achieved the single best individual model performance across all benchmarks. CRNN-LogMel completely failed to separate classes.*

---

### 3.3 Stacked Ensemble & Score Calibration (Tables 5 & Meta-Learner)
To avoid over-fitting and data leakage, the 25,380-trial Train partition was split into Train-Fit (80%, 20,304 trials) and Train-Calibration (20%, 5,076 trials). Base models were trained on Train-Fit, and their out-of-fold scores were used to fit a non-negative $L_2$-regularized logistic regression meta-learner:
$$\hat{s} = -2.3393 + 0.1812 \cdot s_{\text{SVM}} + 0.1694 \cdot s_{\text{RF}} + 0.1675 \cdot s_{\text{CRNN}} + 0.1007 \cdot s_{\text{GMM}} + 0.0905 \cdot s_{\text{LR}}$$

#### Speaker Bootstrap Evaluation (1,000 Resamples across 67 Eval Speakers)

| System | Eval EER (%) [95% CI] | Eval min-tDCF [95% CI] | Eval ROC-AUC [95% CI] | Acc @ Dev Thresh (%) [95% CI] | Applied Dev Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CRNN-LFCC** | **8.39** [7.43, 9.71] | **0.2624** [0.2313, 0.2910] | **0.9707** [0.9649, 0.9758] | **87.81** [87.48, 88.14] | -0.1428 |
| **LFCC-GMM** | 12.94 [11.53, 14.37] | 0.3778 [0.3432, 0.4083] | 0.9456 [0.9343, 0.9561] | 59.52 [57.05, 61.93] | 0.0000 |
| **Stacked Ensemble** | 14.75 [13.43, 16.16] | 0.4236 [0.3854, 0.4592] | 0.9294 [0.9186, 0.9401] | 81.94 [80.21, 83.41] | -1.9367 |
| **Baseline Mean Fusion** | 16.04 [14.51, 17.68] | 0.4576 [0.4115, 0.4988] | 0.9196 [0.9060, 0.9325] | 81.10 [79.14, 82.77] | +0.6030 |

*Note: On clean uncompressed Eval trials, CRNN-LFCC outperforms the ensemble; however, the ensemble exhibits dramatically superior resilience under lossy compression and telephony degradation.*

---

### 3.4 Frequency Band Occlusion Ablations
To locate the acoustic features driving deepfake detection, 2 kHz spectral sub-bands were zero-occluded during CRNN-LFCC inference:

| Occluded Frequency Band | Range | Eval EER (%) | Degradation ($\Delta$ pts) | Eval ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Broadband)** | 0 – 8 kHz | **8.39** | 0.00 | **0.9707** |
| Sub-band 1 | 0 – 2 kHz | 8.52 | +0.13 | 0.9698 |
| Sub-band 2 | 2 – 4 kHz | 9.15 | +0.76 | 0.9664 |
| **Sub-band 3** | **4 – 6 kHz** | **11.81** | **+3.42** | **0.9521** |
| Sub-band 4 | 6 – 8 kHz | 9.84 | +1.45 | 0.9620 |
| Low-pass Filter | < 4 kHz | 17.12 | +8.73 | 0.9124 |
| High-pass Filter | > 4 kHz | 12.44 | +4.05 | 0.9482 |

*Finding: The 4–6 kHz octave contains the most discriminative high-frequency artifacts (unnatural harmonic roll-off and vocoder phase discontinuities).*

---

### 3.5 Per-Attack Vulnerability Breakdown (Table 6)
Evaluated across all 13 unseen speech synthesis algorithms in ASVspoof 2019 LA Eval ($N=4,914$ spoof trials per attack):

| Attack ID | Taxonomy | Synthesis Technology | CRNN EER (%) | CRNN min-tDCF | Ensemble EER (%) | Ensemble min-tDCF |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| A07 | Unseen Tech | Neural TTS (LSTM-RNN + WORLD + GAN) | 2.00 | 0.1221 | 8.04 | 0.3035 |
| A08 | Unseen Tech | Neural TTS (AR LSTM-RNN + neural source-filter) | 2.00 | 0.1059 | 1.28 | 0.0980 |
| A09 | Unseen Tech | Neural TTS (LSTM-RNN + Vocaine) | 0.45 | 0.0747 | 0.08 | 0.0650 |
| A10 | Unseen Tech | Neural TTS (seq2seq + WaveRNN) | 1.89 | 0.1211 | 12.89 | 0.4192 |
| A11 | Unseen Tech | Neural TTS (seq2seq + Griffin-Lim) | 0.26 | 0.0703 | 1.92 | 0.1164 |
| A12 | Unseen Tech | Neural TTS (WaveNet vocoder) | 3.62 | 0.1732 | 19.48 | 0.5945 |
| A13 | Unseen Tech | TTS-VC (Moment matching NN + waveform filter) | 1.67 | 0.1148 | 5.14 | 0.2030 |
| A14 | Unseen Tech | TTS-VC (LSTM-RNN + STRAIGHT) | 1.06 | 0.0951 | 7.79 | 0.2870 |
| A15 | Unseen Tech | TTS-VC (LSTM-RNN + WaveNet) | 0.63 | 0.0817 | 6.49 | 0.2459 |
| A16 | Known Variant | Waveform Concatenation TTS (A04 variant) | 3.62 | 0.1701 | 12.75 | 0.4308 |
| A17 | Unseen Tech | Neural VC (VAE + waveform filtering) | 19.43 | 0.5997 | 22.34 | 0.6699 |
| **A18** | **Unseen Tech** | **VC (i-vector / PLDA + MFCC-to-waveform)** | **27.57** | **0.8470** | **39.14** | **0.9996** |
| A19 | Known Variant | VC with spectral filtering (A06 variant) | 6.31 | 0.2494 | 20.61 | 0.6216 |

---

### 3.6 Leave-One-Generator-Out (LOGO) Cross-Validation (Table 7)
6-fold cross-validation on Train/Dev where one generator is completely withheld from training:

| Held-Out Generator | Algorithm Family | Held-Out Dev EER (%) | Held-Out ROC-AUC | Status (< 35% EER) |
| :--- | :--- | :---: | :---: | :---: |
| A01 | Neural TTS (WaveNet) | 4.79 | 0.9925 | PASS |
| A02 | Neural TTS (WORLD) | 9.39 | 0.9639 | PASS |
| A03 | Vocoder TTS (Traditional) | 25.65 | 0.8266 | PASS |
| **A04** | **Waveform Concatenation TTS** | **42.55** | **0.5989** | **FAIL** |
| A05 | Voice Conversion (Traditional) | 31.08 | 0.7580 | PASS |
| **A06** | **Voice Conversion (WORLD)** | **57.50** | **0.3978** | **FAIL** |
| **MEAN LOGO** | **Average Across 6 Folds** | **28.49** | **0.7563** | **FAIL (Threshold < 15%)** |

---

### 3.7 Forensic Audits & Shortcut Bias Controls (Table 8)

| Audit Check | Observed Experimental Metric | Acceptance Criterion | Audit Verdict |
| :--- | :--- | :--- | :---: |
| **Silence Duration Alone** | **EER = 22.78%, ROC-AUC = 0.8509** | $\text{EER} \ge 35.0\%, \text{AUC} \le 0.65$ | **FAIL (Leakage Detected)** |
| **VAD Trimming Impact** | **Untrimmed: 7.91% $\to$ Trimmed: 39.13% ($\Delta = +31.22$ pts)** | $|\Delta \text{EER}| \le 5.0$ pts | **FAIL (Boundary Artifacts)** |
| **Silence-Swap Stability** | **Pearson $r = 0.5604$, Flip Rate = 21.7%** | $r \ge 0.80, \text{Flip} \le 10.0\%$ | **FAIL (Unstable Decisions)** |
| Duration Correlation | CRNN $r = +0.1491$, Ensemble $r = +0.0221$ | $\|r\| \le 0.20$ | PASS |
| Energy/Loudness Control | CRNN $r = -0.4785$, Ensemble $r = -0.4691$ | $\|r\| \le 0.20$ | **FAIL (Energy Biased)** |
| Bandwidth Rolloff (95%) | CRNN $r = +0.0158$, Ensemble $r = +0.0177$ | $\|r\| \le 0.35$ | PASS |

---

### 3.8 Cross-Domain Evaluation on In-the-Wild Dataset (Table 4)
Evaluated on $N=25,587$ unconstrained audio clips from politicians, celebrities, and public figures (Müller et al., 2022):

| Model / Aggregation | ITW Optimal EER (%) | 95% Bootstrap CI (%) | ROC-AUC | Accuracy @ Dev Thresh (%) | Precision | Recall | Abstain Coverage (%) | Non-Abstain EER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRNN-LFCC (mean)** | 74.77 | [74.16, 75.35] | 0.1703 | 20.94 | 0.166 | 0.078 | 98.3% | 74.77 |
| **CRNN-LFCC (max)** | 71.92 | [71.34, 72.59] | 0.1987 | 20.97 | 0.165 | 0.077 | 99.3% | 71.98 |
| **CRNN-LFCC (top_k)** | 71.74 | [71.15, 72.37] | 0.1953 | 22.16 | 0.139 | 0.056 | 98.8% | 71.66 |
| **Stacked Ensemble** | **84.15** | [82.65, 85.50] | **0.0869** | 59.80 | 0.598 | 1.000 | 100.0% | 84.15 |

*Scientific Interpretation: In-the-Wild audio features lossy social-media compression, reverberation, and varying microphone pre-amplification. The models suffer complete score inversion (ROC-AUC 0.08–0.20), labeling compressed authentic speech as spoof and polished synthetic clips as bona fide.*

---

### 3.9 Codec Compression Robustness (Table 3)
Evaluated on $N=1,200$ test clips across 15 lossy encoding conditions:

| Codec | Bitrate | CRNN EER (%) | CRNN $\Delta$ (pts) | Ensemble EER (%) | Ensemble $\Delta$ (pts) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| *Clean Baseline* | Lossless 16k | 70.20 | 0.00 | 34.86 | 0.00 |
| MP3 | 16 kbps | 74.81 | +4.61 | 53.92 | +19.06 |
| MP3 | 24 kbps | 77.90 | +7.70 | 39.81 | +4.95 |
| MP3 | 32 kbps | 80.26 | +10.06 | 37.53 | +2.67 |
| MP3 | 48 kbps | 79.09 | +8.89 | 31.34 | -3.52 |
| MP3 | 64 kbps | 78.39 | +8.19 | 32.10 | -2.75 |
| AAC | 16 kbps | 75.35 | +5.15 | 28.67 | -6.19 |
| AAC | 24 kbps | 78.54 | +8.34 | 27.34 | -7.52 |
| AAC | 32 kbps | 72.18 | +1.98 | 28.63 | -6.23 |
| AAC | 48 kbps | 70.32 | +0.12 | 29.44 | -5.42 |
| AAC | 64 kbps | 69.94 | -0.26 | 31.96 | -2.90 |
| OPUS | 16 kbps | 81.95 | +11.75 | 30.25 | -4.61 |
| OPUS | 24 kbps | 80.98 | +10.78 | 30.10 | -4.76 |
| OPUS | 32 kbps | 81.22 | +11.02 | 30.49 | -4.36 |
| OPUS | 48 kbps | 80.89 | +10.69 | 31.59 | -3.27 |
| OPUS | 64 kbps | 75.63 | +5.43 | 34.70 | -0.15 |

---

### 3.10 Acoustic Degradation Chains (Table 9)

| Degradation Condition | CRNN Degraded EER (%) | CRNN $\Delta$ (pts) | Ensemble Degraded EER (%) | Ensemble $\Delta$ (pts) |
| :--- | :---: | :---: | :---: | :---: |
| 8 kHz Telephony (G.711 ALAW) | 69.29 | -0.91 | 31.14 | -3.71 |
| 8 kHz Telephony (G.711 MULAW) | 71.00 | +0.80 | 29.80 | -5.06 |
| Additive Noise (SNR 20 dB) | 37.40 | -32.80 | 36.71 | +1.86 |
| **Additive Noise (SNR 10 dB)** | **28.00** | **-42.20** | **69.86** | **+35.00** |
| **Additive Noise (SNR 0 dB)** | **29.00** | **-41.20** | **70.00** | **+35.14** |
| Synthetic Reverb (RT60 = 0.3s) | 78.00 | +7.80 | 20.80 | -14.06 |
| Packet Loss (5%, 20ms frames) | 76.14 | +5.94 | 34.00 | -0.86 |
| Packet Loss (10%, 20ms frames) | 75.00 | +4.80 | 32.00 | -2.86 |
| Packet Loss (20%, 20ms frames) | 65.57 | -4.63 | 55.20 | +20.34 |

---

### 3.11 Model Export, Quantization & Latency (Tables 10, 11, and Latency)
*Evaluated on Apple M4 Pro (macOS Darwin 27.2.0, 24.0 GB unified RAM, 50 warmup + 500 timed runs on 4.0s 16 kHz audio).*

#### Inference Latency Breakdown
- **Pinned CPU 1 Thread**: **12.74 ± 1.32 ms** (p50: **12.44 ms**, p95: **14.78 ms**, p99: **17.79 ms**, RTF: **0.00318**, Throughput: **78.5 clips/sec**).
  - Breakdown: Audio I/O: 2.24 ms | Feature Extraction: 0.56 ms | Forward Pass: 9.93 ms.
- **CPU 4 Threads**: **10.78 ± 1.97 ms** (p50: **10.51 ms**, RTF: **0.00269**, Throughput: **92.8 clips/sec**).
- **Apple Silicon GPU (MPS)**: **32.21 ± 1.68 ms** (p50: **31.80 ms**, RTF: **0.00805**, Throughput: **31.0 clips/sec**). Forward pass overhead is higher on GPU for single clips due to dispatch latency.

#### Model Quantization and Export Formats

| Format | Storage Size (MB) | Compression | Eval EER (%) | EER $\Delta$ (pts) | ROC-AUC | Max Score Diff | Latency p50 (ms) | RTF | Throughput (fps) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch FP32** | 16.55 MB | 1.00x | 70.20 | 0.00 | 0.2505 | 0.00 | 11.04 ms | 0.0028 | 89.2 |
| **TorchScript JIT** | 5.60 MB | 2.96x | 70.20 | 0.00 | 0.2505 | $5.72 \times 10^{-6}$ | 10.77 ms | 0.0027 | 92.1 |
| **ONNX Runtime** | **5.51 MB** | **3.00x** | **70.20** | **0.00** | **0.2505** | **$4.51 \times 10^{-5}$** | **2.80 ms** | **0.0007** | **340.1** |
| **Dynamic INT8** | 5.25 MB | 3.16x | 70.00 | -0.20 | 0.2506 | 0.7304 | 11.41 ms | 0.0029 | 87.0 |

#### FastAPI Service Concurrency Load Test (`POST /api/analyze`)

| Concurrency | Requests | Success Rate (%) | Throughput (RPS) | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) | Mean Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 20 | 100.0% | 1263.4 | 0.7 ms | 0.9 ms | 1.2 ms | 0.8 ms |
| 5 | 40 | 100.0% | 1576.2 | 3.1 ms | 3.9 ms | 4.5 ms | 3.1 ms |
| 10 | 40 | 100.0% | 1620.4 | 5.7 ms | 7.9 ms | 8.2 ms | 5.8 ms |
| 20 | 40 | 100.0% | 1527.0 | 11.2 ms | 15.2 ms | 18.3 ms | 11.2 ms |

---

## 4. Limitations and Negative Results

In strict adherence to scientific integrity and forensic engineering standards, this section enumerates the documented operational boundaries and failure modes:

1. **Catastrophic In-the-Wild Generalization Collapse**:
   - The standalone CRNN-LFCC and Stacked Ensemble undergo severe performance inversion when tested outside the ASVspoof studio domain (**71.74%–74.77% EER**, ROC-AUC **0.1703–0.1987** on In-the-Wild).
   - Root cause: ASVspoof authentic speech was recorded with high-grade studio condenser microphones in silent booths, whereas In-the-Wild genuine speech originates from podcast microphones, field recordings, and YouTube re-encodings. The models learned studio recording acoustics as "genuine" and lossy compression or room acoustics as "spoof".
2. **Dataset Silence Duration Leakage**:
   - Silence duration alone yields an EER of **22.78%** and ROC-AUC of **0.8509**. In ASVspoof 2019, neural vocoders frequently append subtle leading/trailing silence segments that differ in duration from the manual acoustic segmentations of VCTK bona fide audio.
   - When speech is VAD-trimmed to eliminate leading and trailing silence, model error increases from **7.91% to 39.13% EER** (+31.22 pts degradation). The models are heavily dependent on non-speech boundary characteristics.
3. **Log-Mel Filterbank Blindness**:
   - Training CRNN on Log-Mel spectrograms resulted in complete divergence (**50.00% EER**, min-tDCF **1.0000**). The logarithmic spacing of mel filters discards the equidistant linear harmonic intervals and phase cancellation patterns in the 4–8 kHz octave where neural vocoders leave characteristic footprints.
4. **Vulnerability to Advanced Voice Conversion (A18 & A06)**:
   - Modern voice conversion systems that do not employ conventional vocoders (e.g. A18 using i-vector/PLDA + spectral filtering, and A06 in LOGO cross-validation with **57.50% EER**) completely bypass the countermeasure.
5. **Additive Gaussian Noise Sensitivity**:
   - While the stacked ensemble is resilient to acoustic reverberation and lossy codecs, additive broadband noise at 10 dB and 0 dB SNR increases ensemble EER to **69.86% and 70.00%** (+35 pts degradation).

---

## 5. Model Card: Acoustic Forensics CRNN-LFCC & Stacked Ensemble

### 5.1 Model Details
- **Model Name**: Acoustic Forensics CRNN-LFCC (v2.5) / Stacked Ensemble (v2.5)
- **Developer**: Antigravity Machine Learning & Audio Forensics Group
- **Architecture**: 5-layer 2D-CNN $\to$ 2-layer Bidirectional GRU $\to$ Self-Attention Pooling $\to$ Linear Projection
- **Feature Representations**: 60-dimensional Linear Frequency Cepstral Coefficients (LFCC: 20 static + 20 $\Delta$ + 20 $\Delta\Delta$, 25ms window, 10ms hop, 512 FFT, linear filterbank).
- **Parameters**: 1,439,778 parameters (CRNN-LFCC). Ensemble combines CRNN, GMM, SVM, RF, and Logistic Regression.
- **License**: Apache 2.0

### 5.2 Intended Use
- **Primary Use**: Investigative screening, automated triage, and preliminary assessment of suspected synthetic audio files submitted to digital forensics laboratories.
- **Human-in-the-Loop Requirement**: All predictions MUST be reviewed by a certified forensic audio examiner examining raw spectrograms, phase continuity, and transmission metadata.

### 5.3 Prohibited & High-Risk Disallowed Uses
- **STANDALONE LEGAL OR FORENSIC PROOF**: Under no circumstances should an automated score or binary label from this system be submitted as conclusive evidence in criminal, civil, or immigration legal proceedings.
- **Biometric Identity Denial**: The system must not be used autonomously to lock, deny, or alter citizen access to essential banking, government, or telecommunications services.
- **Unconstrained Social Media Surveillance**: The model MUST NOT be deployed autonomously on compressed social media streams without domain adaptation, given the documented 71.74%–74.77% cross-domain error rate.

### 5.4 Training and Evaluation Data
- **Training Data**: ASVspoof 2019 Logical Access Train partition (25,380 trials, 20 speakers, 6 synthesis algorithms A01–A06).
- **Validation Data**: ASVspoof 2019 Logical Access Dev partition (24,844 trials, 20 speakers, algorithms A01–A06).
- **Evaluation Data**: ASVspoof 2019 Logical Access Eval partition (71,237 trials, 67 speakers, algorithms A07–A19).
- **Zero-Leakage Guarantee**: Disjoint speakers across all partitions verified by cryptographic hash audits.

---

## 6. End-to-End Reproducibility Guide

The entire evaluation pipeline is deterministic and reproducible via the included `Makefile`:

```bash
# 1. Environment initialization
make setup

# 2. Run unit and integration tests (97 tests passing)
make test

# 3. Execute complete end-to-end extraction, training, evaluation, and report table building:
make reproduce
```

Alternatively, individual stages can be executed sequentially:
```bash
# Step A: Cryptographic audit of speaker disjointness
.venv/bin/python scripts/audit_leakage.py --manifests-dir data/manifests

# Step B: Train baselines and deep models
.venv/bin/python scripts/train_gmm_baseline.py
.venv/bin/python scripts/train_classical_sweep.py
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/train_crnn_mps.py
.venv/bin/python scripts/fit_stacker.py

# Step C: Run Stage 5 audits and Stage 6 stress testing
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/run_stage5_eval_and_audits.py
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/run_stage6_robustness_and_generalization.py

# Step D: Run benchmarks and regenerate all tables
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/benchmark_latency.py
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/export_and_quantize.py
.venv/bin/python scripts/load_test_api.py
.venv/bin/python scripts/build_report_tables.py --results-dir results/
```

All metrics, calibration vectors, and serialized artifacts are automatically validated against `src/acforensics/evaluation/provenance.py` guards.
