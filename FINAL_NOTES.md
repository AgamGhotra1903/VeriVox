# Acoustic Forensics: Final Verification Notes & Engineering Audit Log
**Branch:** `fixes/final` (branched from `fixes/review-round-1`)  
**Authors:** Agam Ghotra (IIT2024226), Paras Bansal (IIT2024221), Chunit Bansal (IIT2024233), Jashan Goyal (IIT2024223)  
**Date:** November 2026  

---

## 1. Summary of Architectural Decisions & Ground Truth Findings

### 1.1 Single Headline Recommended System: CRNN-LFCC
- **Decision:** The CRNN-LFCC model is established as the single primary production model across the codebase, REST API, Streamlit UI, and evaluation reports.
- **Rationale:** 
  1. On ASVspoof 2019 LA Eval, CRNN-LFCC achieves **6.70%** EER, **0.2143** min-tDCF, and **0.9800** ROC-AUC (Seed 42), outperforming all four classical baselines (GMM: 11.46%, RF: 17.52%, SVM: 19.64%, LR: 23.94%).
  2. Across 3 distinct seeds (42, 43, 44), CRNN-LFCC yields **1.13% ± 0.08%** Dev EER, **7.10% ± 0.38%** Eval EER, and **0.2166 ± 0.0016** Eval min-tDCF.
  3. Model selection strictly on Dev set identifies **Seed 43** (Dev EER = **1.02%**, lowest among the three seeds) as the primary checkpoint, saved as `checkpoints/crnn_lfcc_best.pt`.
  4. On out-of-domain evaluation (In-the-Wild, $N=25{,}587$), CRNN-LFCC achieves **44.62%** EER [44.02%, 45.24%] and **0.5678** ROC-AUC with zero-shot multi-crop temporal window aggregation, outperforming the Stacked Ensemble (63.21% EER, 0.3700 ROC-AUC).
- **Ensemble Status:** The 6-model stacked ensemble is preserved strictly as an **EXPERIMENTAL ABLATION**. The meta-learner overfits to the ASVspoof Development distribution, exhibiting degraded generalization on out-of-domain In-the-Wild recordings. Consequently, CRNN-LFCC standalone is recommended for production.

---

## 2. Technical Findings, Root Causes & Verified Resolutions

### 2.1 Resolution of GMM vs Classical Baseline Performance Gap
- **Observed Behavior:** The 64-component diagonal GMM baseline achieves **11.46%** Eval EER, whereas Random Forest achieves **17.52%**, RBF-SVM achieves **19.64%**, and Logistic Regression achieves **23.94%**.
- **Root Cause:**
  - **GMM:** Operates directly on unpooled, frame-level 60-dimensional LFCC acoustic vectors ($N=250{,}000$ frames per class). The log-likelihood ratio scores each individual frame and averages across time, preserving frame-level density distributions.
  - **Classical Baselines:** Rely on utterance-level summary statistics (temporal mean $\mu$ and standard deviation $\sigma$ across frames, producing 120 dimensions, followed by PCA retaining 95% variance $\approx 66$ components). This collapses temporal sequential dynamics.
  - **CRNN-LFCC:** Retains the complete 2D spatio-temporal structure $(60 \times \sim400)$ through hierarchical 2D convolutions, bidirectional recurrent context (BiGRU), and self-attention pooling, outperforming GMM by 4.76 percentage points in Eval EER and 0.1429 in min-tDCF.

### 2.2 Numerical Parity Discrepancy between Offline Caching and Online REST API
- **Symptom:** Initial tests comparing the REST API output with offline cached evaluations showed a discrepancy on clips shorter than 4.0 seconds.
- **Root Causes:**
  1. **Peak Normalization:** `load_and_validate_audio` defaulted to `normalization_method="peak"`, whereas offline caching (`scripts/cache_lfcc_features.py`) used unnormalized floating-point audio directly from `soundfile.read`.
  2. **Padding Technique:** For audio clips shorter than 4.0 seconds (64,000 samples), `fixed_length_crop_or_pad` defaulted to `repeat_pad`, whereas offline caching used constant symmetric zero-padding.
- **Resolution:**
  - Standardized `EnsemblePredictor.predict_file`, `EnsemblePredictor.predict_waveform`, and `ForensicService` to use `normalization_method="none"` and constant symmetric zero-padding matching offline feature extraction.
  - Verified across 200 evaluation clips in `tests/test_ensemble_parity.py`: maximum absolute difference is **$3.03 \times 10^{-5}$** ($< 10^{-4}$ tolerance).
  - Verified across 50 audio files in `tests/test_api_crnn_parity.py`: maximum absolute difference between API `/api/result/{job_id}` and offline inference is **$2.58 \times 10^{-5}$** ($< 10^{-4}$ tolerance).

### 2.3 Cross-Domain In-the-Wild Polarity Guard
- **Symptom:** Inverted ROC-AUC on In-the-Wild corpus observed in earlier iterations.
- **Root Cause:** Mismatch in continuous score polarity conventions (biometric convention: higher score = more bona fide) when scoring In-the-Wild ground truth labels, coupled with an obsolete static global feature normalizer.
- **Resolution:**
  - Standardized all model outputs to output $s(x) = \text{logit}(\text{bonafide}) - \text{logit}(\text{spoof})$.
  - Standardized all evaluation scripts to use `per_utterance_cmvn`.
  - Verified with assertion test in `tests/test_protocols.py::test_in_the_wild_label_polarity_guard`.

### 2.4 MP3 Codec Delay Compensation Audit
- **Observation:** MP3 encoders (LAME) introduce an algorithmic encoding delay of 576 to 1,152 samples (~36 to 72 ms at 16 kHz), causing a slight temporal phase shift.
- **Audit Findings:** Evaluated on a stratified 500-clip evaluation subset comparing delay compensation ON vs OFF (`scripts/check_mp3_delay.py`).
  - MP3 16 kbps: Delay ON = 13.84%, Delay OFF = 13.84% ($\Delta = +0.00$ pts).
  - MP3 64 kbps: Delay ON = 21.15%, Delay OFF = 21.15% ($\Delta = +0.00$ pts).
  - **Conclusion:** Because CRNN-LFCC operates on spectral magnitude features via STFT (discarding phase) followed by temporal self-attention pooling, the temporal offset induces an EER difference of zero to two decimal places, confirming translation invariance. Serialized in `results/metrics/mp3_delay_check.json`.

### 2.5 Soundness Audits ($N \ge 1,000$ Trials)
To avoid misleading "PASS" claims, all soundness audits were evaluated on $N \ge 1,000$ trials:
1. **Silence Duration Classifier ($N=1,200$):** A logistic regression classifier on silence durations achieves EER of **21.55%** and AUC of **0.8579** (FAIL: showing that synthetic audio in this split contains non-trivial silence duration cues).
2. **VAD Trimming Audit ($N=1,000$ trials):** Untrimmed EER is **8.67%**, whereas trimmed EER is **16.00%**, resulting in $\Delta\text{EER} = \mathbf{7.33}$ percentage points (FAIL $> 5.0$ pts). The detector relies in part on natural onset/offset silence transitions.
3. **Silence-Swap Audit ($N=1,000$ paired trials, $2,000$ waveforms):** Splicing bona fide silence onto synthetic speech results in Pearson correlation $r = 0.7131$ and a decision flip rate of **36.5%** (FAIL $> 10.0\%$).
4. **Confounder Controls ($N=1,200$):** Duration $r = +0.1773$ (PASS), Bandwidth $r = +0.0444$ (PASS), Loudness $r = -0.4805$ (FAIL $> 0.20$, confirming score correlation with recording loudness).

---

## 3. Provenance & Artifact Checkpoint Hashes

All active checkpoints are version-controlled with immutable SHA-256 hashes:

| Checkpoint Path | Architecture / Purpose | Parameters | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| `checkpoints/crnn_lfcc_seed42.pt` | CRNN-LFCC (Seed 42 retrained) | 1,439,778 | `d215914bab832bb69124b9a3fe14b80bb469592ae8503444c6cbd972f650fa65` |
| `checkpoints/crnn_lfcc_seed43.pt` | CRNN-LFCC Primary (Lowest Dev EER) | 1,439,778 | `e09c93f4bdfa961cebf964ec6f40c26baf3dd05818b11315ff9a7786f006f565` |
| `checkpoints/crnn_lfcc_seed44.pt` | CRNN-LFCC Multi-Seed (Seed 44) | 1,439,778 | `561d9d3ed3966b5c7dce919e750f3655fad40678bab5cf4c483b63f862666996` |
| `checkpoints/crnn_lfcc_best.pt` | Production Checkpoint (Copy of Seed 43) | 1,439,778 | `e09c93f4bdfa961cebf964ec6f40c26baf3dd05818b11315ff9a7786f006f565` |
| `checkpoints/ensemble/classical_bundle.joblib` | Classical Models (GMM, SVM, RF, LR, PCA) | Multi-model | `4c4f09d8aa0f55cf08b76b4ef84c718a70baeb1918a2ea6898436ca142e053a4` |

---

## 4. Verification & Testing Evidence

1. **Automated Unit & Integration Test Suite:**
   - Command: `.venv/bin/pytest tests/`
   - Outcome: All tests passing across all test suites without regressions.
2. **Report Numbers Integrity Verifier:**
   - Command: `.venv/bin/python scripts/verify_report_numbers.py docs/REPORT.md`
   - Outcome: **54 claimed metrics scanned, 54 verified, 0 mismatches**.
3. **From-Scratch Algorithm Verification:**
   - Pure NumPy implementation of Newton-Raphson matches scikit-learn within 0.04 percentage points on Dev EER.
   - Pure NumPy SVD matches scikit-learn PCA explained variance within $10^{-7}$.
   - Purity and Adjusted Rand Index confirmed on $k$-Means ($k=2$).
