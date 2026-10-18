# Dataset Card: Acoustic Forensics Benchmark Datasets ("AF-Split v1")

**Authors:** Paras Bansal (IIT2024221), Jashan Goyal (IIT2024223), Agam Ghotra (IIT2024226), Chunit Bansal (IIT2024233)  
**Date:** October 2026  
**Standards Compliance:** Follows official ASVspoof 2019 evaluation protocol and In-The-Wild cross-domain conventions.

---

## 1. Dataset Overview & Partition Architecture

The Acoustic Forensics benchmark evaluates deepfake and synthetic speech detection models using an experimental partition architecture designated **AF-Split v1**. The framework strictly adheres to zero-leakage, official-protocol isolation, and speaker-disjoint splitting.

### Summary Partition Table ("AF-Split v1")

| Split Role | Source Dataset | Bona Fide Utterances | Spoof Utterances | Total Utterances | Unique Speakers | Attack / Generation Algorithms | Evaluation Purpose |
|---|---|---|---|---|---|---|---|
| **TRAIN** | ASVspoof 2019 LA `train` | 2,580 (10.17%) | 22,800 (89.83%) | 25,380 | 20 (8M / 12F) | Known attacks: A01 – A06 | Parameter optimization and feature fitting |
| **VAL** | ASVspoof 2019 LA `dev` | 2,548 (10.26%) | 22,296 (89.74%) | 24,844 | 20 (10M / 10F, disjoint) | Known attacks: A01 – A06 | Early stopping, threshold selection, calibration, ensemble fusion |
| **TEST-1** | ASVspoof 2019 LA `eval` | 7,355 (10.32%) | 63,882 (89.68%) | 71,237 | 67 (34M / 33F, disjoint) | **Unseen attacks: A07 – A19** | Primary in-domain headline benchmark (EER, AUC, min-tDCF) |
| **TEST-2** | In-The-Wild `itw_test` | 16,968 (63.0%) | 9,968 (37.0%) | 26,936 | ~49 public figures | Real-world synthesis, voice conversion, cloned audio | Zero-shot cross-domain generalization benchmark |
| **ITW-CAL** | In-The-Wild `itw_cal` | 2,995 (61.8%) | 1,848 (38.2%) | 4,843 | ~9 public figures (15%, disjoint) | Real-world deepfakes | **Strictly** for optional few-shot calibration (excluded from all headline evals) |
| **DERIVED** | Compressed `*-compressed` | Mirror of TEST-1/2 | Mirror of TEST-1/2 | Multi-rate copies | Same as source | MP3, AAC, Opus at 16, 24, 32, 48, 64 kbps | Compression robustness curves and degradation analysis |

---

## 2. Attack Taxonomy (ASVspoof 2019 LA)

### Known Generation Algorithms (Seen in Train & Dev)
- **A01:** Neural network waveform cloning / Tacotron-like acoustic model + WaveNet vocoder.
- **A02:** Vocoder-based text-to-speech (TTS) with WORLD vocoder.
- **A03:** Traditional concatenative / unit-selection speech synthesis.
- **A04:** Voice conversion (VC) via spectral filtering and pitch-synchronous overlap-add (PSOLA).
- **A05:** Voice conversion via statistical vocal tract transformation with WORLD vocoder.
- **A06:** Voice conversion using frequency-warping and spectral transformation.

### Unseen Generation Algorithms (Evaluated only in Test-1 / Eval)
- **A07 – A19:** Advanced neural vocoders (MelGAN, HiFi-GAN, WaveGlow), flow-based synthesis, Griffin-Lim acoustic inversion, variational autoencoders, and high-fidelity modern neural voice conversion algorithms.

---

## 3. Class Imbalance Characteristics

Across all ASVspoof 2019 partitions, the class distribution is heavily skewed toward synthetic audio (~10% bona fide vs ~90% spoof).
To prevent metric distortion:
1. **Loss Weighting:** Training employs class-weighted Cross-Entropy loss ($w_c \propto 1 / N_c$).
2. **Balanced Batch Sampling:** An optional `BalancedBatchSampler` is provided for uniform batch class representation.
3. **Evaluation Metrics:** We report Balanced Accuracy, Equal Error Rate (EER), and min-tDCF alongside standard ROC-AUC, ensuring reporting does not rely on trivial majority-class guessing.

---

## 4. Known Dataset Caveats & Shortcut Artifacts

### 4.1 VCTK Corpus Heritage
> **Critical Finding:** The bona fide speech in ASVspoof 2019 LA was harvested from the **CSTR VCTK Corpus** (107 Scottish and English regional accents recorded under semi-anechoic studio conditions).  
> **Consequence:** Extension datasets constructed from VCTK (such as WaveFake, DFADD, or LibriSeVoc) share the exact same underlying speakers and acoustic environments. If not audited, models trained on ASVspoof LA and tested on VCTK-derived extension data exhibit severe data leakage and inflated accuracy due to speaker memorization.

### 4.2 Silence-Shortcut Bias
> **Known Bias:** ASVspoof 2019 LA utterances contain significant leading and trailing silence segments. In several generation pipelines (e.g. concatenative TTS), silent intervals contain synthetic zero-padding or subtle vocoder buzz that correlates directly with the spoof label rather than active speech.  
> **Mitigation:** We include an Energy-VAD silence trimming experiment (`vad_silence_trim: true`) to assess whether classifier performance relies on silence artifacts.

### 4.3 In-The-Wild Evaluation Metric Caveat
> **Metric Inapplicability:** The Minimum Tandem Detection Cost Function (**min-tDCF**) is officially defined **only** for ASVspoof 2019 LA because it requires matched Automatic Speaker Verification (ASV) score files. In-The-Wild lacks ASV system score trials. Consequently, min-tDCF is mathematically undefined on In-The-Wild; we report EER, ROC-AUC, and Balanced Accuracy only.

### 4.4 Single-Speaker Extension Confound (WaveFake / LJSpeech)
> **Single Speaker Bias:** WaveFake synthesizes speech primarily using the LJSpeech single-speaker corpus (Linda Johnson). Models tested on WaveFake often learn single-speaker acoustic signatures rather than genuine deepfake artifacts. We explicitly document this limitation.

---

## 5. Data Hygiene & Zero-Leakage Audit

All manifests are subject to automated verification via `scripts/audit_leakage.py`:
- **Speaker Disjointness:** Verified 0 speaker overlap across Train, Dev, Eval, ITW-CAL, and ITW-TEST splits.
- **File ID Disjointness:** Verified 0 file ID collisions across splits.
- **Content Duplication:** Audited using SHA-256 byte-level file hashing to ensure no physical audio files are shared across training and testing boundaries.
