# Reproduction Manual: Acoustic Forensics Project

This document provides step-by-step instructions to reproduce all models, evaluation metrics, robustness experiments, cross-domain tests, forensic audits, and report numbers from scratch.

---

## 1. System Requirements & Hardware Environment

- **Operating System:** macOS (Apple Silicon M-Series tested on M4 Pro, Darwin 27.2.0) or Linux (x86_64 / Ubuntu 22.04+).
- **CPU:** Minimum 8 cores recommended (benchmarked on 14-core Apple M4 Pro).
- **RAM:** Minimum 16 GB RAM (24 GB recommended).
- **Storage:** Minimum 80 GB free disk space for raw ASVspoof 2019 LA flac files, In-the-Wild wav files, and intermediate features.
- **Python Environment:** Python 3.11 - 3.13 (Python 3.13.7 used in reference runs).
- **GPU / Compute Acceleration:** Apple Silicon MPS or NVIDIA CUDA 12.x (CPU fallback supported).
- **External Binaries:** `ffmpeg` (version 7.0+ with libmp3lame, aac, and libopus enabled).

---

## 2. Environment Setup

```bash
# 1. Clone repository and checkout final release branch
git clone <repo-url> acoustic_forensics
cd acoustic_forensics
git checkout fixes/final

# 2. Create isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install package in editable mode with all development dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# 4. Verify environment and ffmpeg installation
python -c "import torch; print('PyTorch:', torch.__version__, 'MPS:', torch.backends.mps.is_available())"
ffmpeg -version | head -n 1
```

---

## 3. Dataset Verification & Manifest Preparation

```bash
# Verify disk space, checksums, and dataset availability
python scripts/verify_datasets.py

# Ensure manifests exist in data/manifests/
ls -lh data/manifests/*.parquet
# Expected:
# data/manifests/train.parquet
# data/manifests/dev.parquet
# data/manifests/eval.parquet
# data/manifests/in_the_wild.parquet
```

---

## 4. End-to-End Experiment Execution Pipeline

### Step 1: Preprocessing Parity Verification (Defect A2 Proof)
Verifies that offline cached features and online raw audio streaming produce mathematically identical LFCC representations:
```bash
python scripts/parity_check.py --samples 2000
pytest tests/test_parity_check.py -v
```
**Expected Output:** `results/metrics/parity_check.json` with `delta_eer: 0.0000 pts`, `parity_passed: true`.

---

### Step 2: Primary CRNN-LFCC Training (Multi-Seed 42, 43, 44)
Train the primary deepfake speech detection architecture across 3 independent seeds (25 epochs each, Adam lr=1e-3, CosineAnnealingLR, early stopping patience=8):
```bash
# Seed 42
python scripts/train_crnn_mps.py --features lfcc --seed 42 --epochs 25 --checkpoint checkpoints/crnn_lfcc_seed42.pt

# Seed 43 (Selected Primary Model)
python scripts/train_crnn_mps.py --features lfcc --seed 43 --epochs 25 --checkpoint checkpoints/crnn_lfcc_seed43.pt

# Seed 44
python scripts/train_crnn_mps.py --features lfcc --seed 44 --epochs 25 --checkpoint checkpoints/crnn_lfcc_seed44.pt

# Evaluate multi-seed summary and extract training dynamics
python scripts/plot_crnn_multiseed_curves.py
python scripts/save_crnn_training_histories.py
```
**Expected Outputs:**
- Checkpoints: `checkpoints/crnn_lfcc_seed{42,43,44}.pt`
- Training histories: `results/metrics/crnn_training_run_lfcc_seed{42,43,44}.json`
- Vector plots: `results/figures/crnn_training_curves.{pdf,png}`
- Expected 3-seed Eval EER: $7.10\% \pm 0.38\%$

---

### Step 3: Classical Baselines & Representation Sweep
Fit the 4 classical baselines (GMM, Random Forest, SVM, Logistic Regression) on pooled LFCC and Log-Mel representations:
```bash
# Fit GMM (64 diagonal components)
python scripts/train_gmm.py --features lfcc --components 64 --max-frames 250000

# Fit scikit-learn models (StandardScaler + PCA-30 + Classifier)
python scripts/train_classical.py --features lfcc
python scripts/train_classical.py --features logmel

# Syllabus from-scratch models (scratch Logistic Regression, scratch PCA + k-Means)
python scripts/train_syllabus_models.py
```
**Expected Outputs:**
- Baseline metrics: `results/metrics/baselines_results.json`
- GMM Eval EER: $11.46\%$, Random Forest: $17.52\%$, SVM: $19.64\%$, Logistic Regression: $23.94\%$

---

### Step 4: CRNN-LogMel Overfit & Representation Integrity Check
Verifies optimization dynamics on 64-sample batch of Log-Mel energies to rule out gradient vanishing or ReLU collapse:
```bash
python scripts/verify_logmel_overfit.py
```
**Expected Output:** `results/metrics/logmel_overfit_check.json` showing loss dropping from $0.86097 \rightarrow 0.09541$ with $98.44\%$ accuracy.

---

### Step 5: Stacked Ensemble Fusion & Calibration
Fit non-negative $L_2$-regularized Logistic Regression meta-learner strictly on Dev scores and evaluate on frozen Eval split:
```bash
python scripts/fit_stacker.py
```
**Expected Outputs:**
- Ensemble metrics: `results/metrics/ensemble_results.json`
- Checkpoint: `checkpoints/ensemble/stacked_ensemble.joblib`
- Stacked Ensemble Eval EER: $10.74\%$, Mean Fusion Eval EER: $11.47\%$

---

### Step 6: Full Acoustic Robustness Benchmark (Fixed Raw Audio Path)
Evaluate the primary model across 9 codec conditions (MP3, AAC, Opus at 16, 32, 64 kbps with delay compensation) and 8 physical degradation chains (telephony, additive noise, reverb, packet loss) on $N=5{,}000$ stratified eval clips:
```bash
python scripts/run_robustness_benchmark.py --samples 5000 --seeds 42 43 44
```
**Expected Outputs:**
- `results/metrics/codec_robustness_results.json`
- `results/metrics/degradation_chains_results.json`
- `results/figures/robustness_eer_vs_bitrate.{pdf,png}`
- `results/tables/table3_codec_robustness.md`, `table9_acoustic_degradations.md`

---

### Step 7: In-the-Wild Cross-Domain Generalization & Polarity Audit
Evaluate all 25,589 clips of the In-the-Wild dataset through the raw audio streaming path and execute the polarity inversion diagnostic:
```bash
python scripts/run_itw_audit.py --manifest data/manifests/in_the_wild.parquet
```
**Expected Outputs:**
- `results/metrics/in_the_wild_results.json`
- `results/tables/table4_crossdomain_in_the_wild.md`
- Optimal EER: $44.73\%$, ROC-AUC: $0.5638$ (Inverted diagnostic: ROC-AUC = $0.4362$)

---

### Step 8: Large-Scale Shortcut & Silence Vulnerability Audits
Run silence duration classifier, VAD-trimming sensitivity, silence swap perturbation, and acoustic property controls on $N=20{,}000$ stratified eval clips:
```bash
python scripts/run_large_scale_audits.py --samples 20000
```
**Expected Outputs:**
- `results/metrics/crnn_silence_audit.json`
- `results/tables/table7_silence_audit.md`
- Silence classifier EER: $18.36\%$, Trimmed EER: $30.69\%$ (Gap: $+21.55$ pts), Silence swap flip rate: $34.85\%$

---

### Step 9: Latency, Hardware Throughput & API Load Testing
Benchmark single CRNN and shipped ensemble across CPU (1-thread, 4-threads) and Apple Silicon MPS, followed by end-to-end API load testing (800 total requests across concurrency 1, 5, 10, 20):
```bash
# Hardware latency benchmark (50 warmup + 500 timed runs)
python scripts/benchmark_latency.py

# FastAPI end-to-end load test (200 requests per level)
python scripts/load_test_api.py --requests 200 --duration 4.0

# API-CRNN parity unit test (50 real audio files)
pytest tests/test_api_crnn_parity.py -v
```
**Expected Outputs:**
- `results/metrics/latency.json`
- `results/metrics/api_load_test.json`
- `results/tables/table_latency.md`, `table_api_load_test.md`
- Success Rate: $100.0\%$ across all 800 requests

---

### Step 10: Figure Generation & Master Tables
Generate publication-quality vector PDF and PNG figures:
```bash
python scripts/generate_dossier_figures.py
python scripts/build_master_baseline_table.py
python scripts/build_paper_facts.py
```
**Expected Outputs:**
- `results/figures/all_models_roc.{pdf,png}`
- `results/figures/all_models_det.{pdf,png}`
- `results/figures/eval_confusion_matrix.{pdf,png}`
- `results/figures/itw_confusion_matrix.{pdf,png}`
- `results/figures/silence_duration_vs_score.{pdf,png}`
- `results/metrics/paper_facts.json`

---

## 5. Automated Verification & Test Suite

Run the full automated test suite (all tests must pass):
```bash
pytest tests/ -v
```

Verify that 100% of reported numbers in `handover/final/REPORT.md` match ground-truth metrics JSON files within numerical tolerance:
```bash
python scripts/verify_report_numbers.py handover/final/REPORT.md
```

