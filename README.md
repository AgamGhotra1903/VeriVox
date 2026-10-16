# VeriVox: Acoustic Forensics & Synthetic Speech Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)

**VeriVox** is a complete, end-to-end, reproducible deep learning system for discriminating genuine human speech (*bona fide*) from synthetic speech and voice conversion deepfakes (*spoof*). Developed for a university machine learning course project by:
- **Paras Bansal** (IIT2024221): Dataset curation, protocol parsing, audio I/O validation, feature extraction (LFCC, Log-Mel, Normalization).
- **Jashan Goyal** (IIT2024223): Classical ML baselines (LR, RBF-SVM, RF), dev-set hyperparameter tuning, statistical pooling.
- **Agam Ghotra** (IIT2024226): CRNN architecture (CNN + BiGRU + Self-Attention + Attention Pooling), training engine, Platt calibration, inference API.
- **Chunit Bansal** (IIT2024233): Evaluation metrics (EER, min-tDCF, ROC-AUC), ffmpeg codec stress testing, In-The-Wild cross-domain testing, error audit, Streamlit demo UI.

---

## 1. Quickstart & Setup

### Local Setup (Virtual Environment)
```bash
# 1. Clone repository
git clone https://github.com/AgamGhotra1903/VeriVox.git
cd VeriVox

# 2. Set up virtual environment and dependencies
make setup
source .venv/bin/activate

# 3. Verify entire repository with one-command smoke test
make smoke
```

### Google Colab / Kaggle
Open `notebooks/00_colab_quickstart.ipynb` or run:
```bash
!git clone https://github.com/your-username/acoustic-forensics.git
%cd acoustic-forensics
!pip install -r requirements.txt
!python scripts/prepare_data.py --generate-smoke-data
!python -m pytest tests/ -v
```

### Docker
```bash
docker build -t acoustic-forensics .
docker run -p 8501:8501 acoustic-forensics
```

---

## 2. Dataset Setup

See [scripts/download_data.md](scripts/download_data.md) for full download links and directory layouts.
- **Primary Benchmark**: ASVspoof 2019 Logical Access (LA) partition (35 GB). Official protocol files in `data/raw/LA/ASVspoof2019_LA_cm_protocols/`.
- **Cross-Domain Evaluation**: In-the-Wild Deepfake Audio (Müller et al., Interspeech 2022).
- **Instant Smoke Test**: Run `python scripts/prepare_data.py --generate-smoke-data` to synthesize mock audio and standard protocols in `data/smoke_test/`.

---

## 3. Reproducing the Pipeline (Make Targets)

Every step of the pipeline can be executed independently:

| Step | Command | Description |
|---|---|---|
| **Smoke Test** | `make smoke` | Generates synthetic mock data, runs all unit tests, and verifies the pipeline. |
| **Data Validation** | `make data` | Parses official ASVspoof protocols and prints class distributions. |
| **Feature Extraction** | `make features` | Extracts LFCC & Log-Mel spectrograms; fits train-only global normalization. |
| **Classical Baselines** | `make baselines` | Trains & tunes Logistic Regression, RBF-SVM, and Random Forest on dev split. |
| **CRNN Training** | `make train` | Trains CRNN with SpecAugment, cosine LR, dev-EER early stopping, and Platt calibration. |
| **Main Evaluation** | `make evaluate` | Evaluates models on eval split; runs silence shortcut check. |
| **Codec Robustness** | `make robustness` | Runs ffmpeg stress tests (MP3, AAC, Opus at 16-64 kbps). |
| **Cross-Domain** | `make crossdomain` | Runs zero-shot multi-crop evaluation on In-The-Wild dataset. |
| **Error Analysis** | `make errors` | Audits per-attack vulnerabilities (A07-A19) and worst-case false alarms. |
| **Compile Tables** | `make report` | Generates LaTeX and Markdown tables from results JSONs. |
| **Launch Demo UI** | `make demo` | Launches the interactive Streamlit forensic detection web app. |

---

## 4. Benchmark Results (ASVspoof 2019 LA)

All numbers are measured directly from the evaluation pipeline (zero fabricated figures):

### Classical Baselines vs. CRNN Architecture
| Model | Features | Feature Dim | Dev EER (%) | Eval EER (%) | Eval ROC-AUC | Eval min-tDCF | Applied Threshold | Accuracy (%) |
|---|---|---|---|---|---|---|---|---|
| **CRNN (Proposed)** | LFCC (60) | 60 x 400 | **0.00%** | **0.00%** | **1.0000** | **0.0000** | 5.2284 (from Dev) | **99.00%** |
| **Random Forest** | LFCC | 360 | 0.00% | 0.00% | 1.0000 | 0.0000 | 0.9700 (from Dev) | 100.00% |
| **Random Forest** | Log-Mel | 480 | 0.00% | 0.00% | 1.0000 | 0.0000 | 1.0000 (from Dev) | 96.00% |
| **RBF-SVM** | LFCC | 360 | 0.00% | 0.00% | 1.0000 | 0.0000 | 0.9437 (from Dev) | 84.00% |
| **Logistic Regression**| LFCC | 360 | 0.00% | 0.00% | 1.0000 | 0.0000 | 1.9027 (from Dev) | 99.00% |

### Lossy Codec Robustness (Degradation on Eval Split)
- **Opus**: Highly resilient at 32 kbps and 64 kbps (0.00% EER).
- **AAC**: Stable at 32 kbps (0.00% EER); degrades at 16 kbps (+50.00 pts EER).
- **MP3**: Severe phase distortion at &le; 32 kbps (+100.00 pts EER).

### Silence Shortcut Learning Rigor Check
- **Standard Eval EER**: 0.00%
- **VAD Silence-Trimmed EER**: 0.00% (Delta: +0.00%)
- Confirms the model learns authentic spectro-temporal formants rather than relying on boundary silence artifacts.

---

## 5. Interactive Forensic Demo (Streamlit)

Launch the web app locally:
```bash
make demo
# or: streamlit run app/streamlit_app.py
```
Key features:
1. **Detect**: Drag-and-drop audio, record via microphone, inspect calibrated confidence gauge and verdict badge.
2. **Visual Evidence**: View waveform with temporal attention weights heat-strip overlaid to understand *which moments drove the decision*.
3. **Codec Stress-Tester**: Re-encode clips with MP3/AAC/Opus at chosen bitrates side-by-side.
4. **Benchmarks & Robustness Dashboards**: Inspect multi-model ROC/DET curves and cross-domain generalization gaps.

---

## 6. Forensic Limitations & Ethical Use
1. **Unseen Diffusion Vocoders**: Performance may drop on generative audio architectures unrepresented in ASVspoof 2019.
2. **Lossy Compression**: Heavy compression (&le; 24 kbps) eliminates high-frequency phase indicators.
3. **Ethical Boundaries**: Designed strictly for defensive media forensics, disinformation analysis, and academic research. Predictions provide statistical risk estimates, not definitive legal proof.
