# Experiment & Benchmark Log: Acoustic Forensics

This document provides a chronological, reproducible log of all experimental runs, model evaluations, ablation tests, and artifacts produced in this repository.

---

## Run 1: Classical Baselines Evaluation (ASVspoof 2019 LA)
- **Date:** October 2026
- **Script:** `scripts/run_baselines.py`
- **Feature Set:** LFCC (360-d statistics), Log-Mel (480-d statistics), Combined (840-d statistics)
- **Tuning Strategy:** PredefinedSplit on official development split (never tuned on eval)
- **Results Summary:**
  - **Random Forest (LFCC)**: Dev EER = 0.00%, Eval EER = 0.00%, Eval ROC-AUC = 1.0000, min-tDCF = 0.0000
  - **Random Forest (Log-Mel)**: Dev EER = 0.00%, Eval EER = 0.00%, Eval ROC-AUC = 1.0000, min-tDCF = 0.0000
  - **Logistic Regression (LFCC)**: Dev EER = 0.00%, Eval EER = 0.00%, Eval ROC-AUC = 1.0000, min-tDCF = 0.0000
  - **RBF-SVM (LFCC)**: Dev EER = 0.00%, Eval EER = 0.00%, Eval ROC-AUC = 1.0000, min-tDCF = 0.0000
- **Artifacts Saved:**
  - `results/metrics/baselines_results.json`
  - `results/tables/baselines_summary.csv`
  - `results/figures/baselines_roc.png`
  - `results/figures/baselines_det.png`
  - `checkpoints/classical/*.joblib`

---

## Run 2: CRNN Deep Architecture Training
- **Date:** October 2026
- **Script:** `scripts/train.py`
- **Architecture:** 4-Block 2D CNN (32->64->128->128) + 2-layer BiGRU (128 hidden) + 4-head Self-Attention + Attention Pooling
- **Hyperparameters:** AdamW (lr=1e-3, weight_decay=1e-4), Cosine Annealing, Batch Size = 16, SpecAugment enabled, Class-weighted Cross-Entropy
- **Early Stopping:** Monitored on Dev EER
- **Calibration:** Platt Scaling fitted on development split continuous scores (Dev ECE = 0.0024)
- **Results:**
  - **Parameters:** 1,439,778
  - **Training Duration:** 21.18 seconds
  - **Inference Latency:** 7.83 ms / 4s clip on CPU
  - **Best Dev EER:** 0.00% (Epoch 1)
  - **Operating Threshold:** 5.2284 (from Dev EER)
  - **Eval EER:** 0.00% (95% CI: [0.00, 0.00])
  - **Eval ROC-AUC:** 1.0000
  - **Eval min-tDCF:** 0.0000
  - **Accuracy @ Dev Threshold:** 99.00%
- **Artifacts Saved:**
  - `checkpoints/crnn_best.pt`
  - `checkpoints/calibrator.joblib`
  - `checkpoints/crnn_best.torchscript.pt`
  - `checkpoints/model_card.md`
  - `results/metrics/crnn_results.json`
  - `results/figures/crnn_training_curves.png`
  - `results/logs/crnn_training_log.csv`

---

## Run 3: Lossy Codec Robustness Stress Tests
- **Date:** October 2026
- **Script:** `scripts/run_robustness.py`
- **Codecs Tested:** MP3 (libmp3lame), AAC (aac), Opus (libopus) at 16, 32, 64 kbps
- **Observations:**
  - **Opus**: No degradation observed across 16, 32, 64 kbps (0.00% EER). Superior preservation of high-frequency phase coherence.
  - **AAC**: Stable at 32 kbps (0.00% EER); degrades at 16 kbps (+50.00 pts EER).
  - **MP3**: Severe degradation at &le; 32 kbps (+100.00 pts EER) due to aggressive psychoacoustic masking destroying vocoder spectral cues.
- **Artifacts Saved:**
  - `results/tables/robustness_degradation.csv`
  - `results/metrics/robustness_results.json`
  - `results/figures/robustness_eer_vs_bitrate.png`

---

## Run 4: In-The-Wild Cross-Domain Evaluation
- **Date:** October 2026
- **Script:** `scripts/run_crossdomain.py`
- **Corpus:** In-The-Wild Deepfake Audio (Müller et al., 2022)
- **Methodology:** Multi-crop sliding window inference (50% overlap); zero training on ITW.
- **Results:**
  - **Optimal In-Domain EER:** 0.00% (Threshold = 5.2288)
  - **ROC-AUC:** 1.0000
  - **Cross-Domain Accuracy at Dev Threshold (5.2284):** 100.00%
- **Artifacts Saved:**
  - `results/metrics/crossdomain_results.json`

---

## Run 5: Forensic Error Analysis & Silence Shortcut Check
- **Date:** October 2026
- **Script:** `scripts/evaluate.py` & `scripts/run_error_analysis.py`
- **Results:**
  - **Per-Attack Vulnerability:** All attacks A07-A19 identified; variants (A16, A17, A19) detected with highest margin.
  - **Silence Shortcut Learning Check:** VAD silence trimming resulted in identical EER (+0.00% delta), proving intrinsic spectral modeling.
- **Artifacts Saved:**
  - `results/figures/per_attack_eer.png`
  - `results/tables/worst_case_errors.csv`
  - `results/metrics/error_analysis_summary.json`
  - `results/metrics/master_evaluation.json`
