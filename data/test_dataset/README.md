# Acoustic Forensics: Verification & Testing Audio Dataset

This directory contains curated, standalone audio benchmark clips stored directly on your computer to test the **Acoustic Forensics** deepfake audio screener across authentic human voices and various synthetic deepfake spoofing attacks.

---

## 📁 Directory Structure & File Locations

All files are located on your local filesystem under:
`/Users/agamghotra/Desktop/ML/data/test_dataset/`

```
data/test_dataset/
├── README.md                           <- This documentation file
├── ground_truth_labels.csv             <- Ground-truth labels and attack metadata for all 30 clips
├── test_audio_batch.zip                <- Ready-to-upload ZIP for the Streamlit "Batch" tab
│
├── normal_human_speech/                <- 10 Authentic Human Voice Recordings (Bonafide)
│   ├── human_01_authentic_speech.wav
│   ├── human_02_authentic_speech.wav
│   ├── ...
│   └── human_10_authentic_speech.wav
│
├── deepfake_synthetic_speech/          <- 10 Synthetic Voice Deepfakes (ASVspoof 2019 LA)
│   ├── deepfake_01_synthetic_A17.wav   <- Voice Conversion (VC)
│   ├── deepfake_02_synthetic_A14.wav   <- Neural Text-to-Speech (TTS)
│   ├── deepfake_03_synthetic_A11.wav   <- Neural Vocoder Clone
│   ├── ...
│   └── deepfake_10_synthetic_A18.wav   <- Deep Diffusion / Waveform model
│
├── in_the_wild_evaluation/             <- 10 In-The-Wild Cross-Domain Real-World Clips
│   ├── itw_normal_01.wav .. 05.wav     <- Authentic unconstrained speech
│   └── itw_deepfake_01.wav .. 05.wav   <- Deepfake speech collected from online media
│
└── all_mixed_for_batch_testing/        <- All 30 clips in one folder for blind screening
    ├── human_*.wav (15 files)
    └── deepfake_*.wav / itw_*.wav (15 files)
```

---

## 📊 Summary of Ground-Truth Labels

| Category | Split Origin | Total Count | Label | Expected Verdict | Expected Score |
|---|---|:---:|---|:---:|---|
| **Normal Human Speech** | ASVspoof 2019 LA Eval | 10 | `bonafide` (1) | `✔ GENUINE` | > 0.95 |
| **Normal In-The-Wild** | In-The-Wild Dataset | 5 | `bonafide` (1) | `✔ GENUINE` | > 0.95 |
| **Deepfake Speech** | ASVspoof 2019 LA Eval | 10 | `spoof` (0) | `✖ SYNTHETIC` | < 0.05 |
| **Deepfake In-The-Wild** | In-The-Wild Dataset | 5 | `spoof` (0) | `✖ SYNTHETIC` | < 0.05 |
| **Total Test Dataset** | Dual-Corpus Standard | **30 clips** | **Balanced 50/50** | **15 Genuine / 15 Synthetic** | **100% Calibrated** |

---

## 🚀 How to Test in the Streamlit Web Dashboard

### 1. Test Single Clips (Analyze Tab)
1. Open the dashboard at [http://localhost:8501](http://localhost:8501).
2. On the **Analyze** tab, select **Upload Audio File**.
3. Drag any `.wav` file from `data/test_dataset/normal_human_speech/` or `data/test_dataset/deepfake_synthetic_speech/`.
4. Inspect the **Calibrated Authenticity Ring**, **3-Zone Score Meter**, and the 5 **Forensic Evidence Tabs** (Waveform Attention, Saliency Heatmap, Timeline Localization).

### 2. Test Batch Screening (Batch Tab)
1. In the dashboard, switch to the **Batch** tab.
2. Drag and drop the ready-made archive:
   `/Users/agamghotra/Desktop/ML/data/test_dataset/test_audio_batch.zip`
3. Click **Run Batch Screening ▶**.
4. The system will process all 30 clips, render the score distribution histogram, display the sortable results table, and allow CSV export.

---

## 💻 CLI Verification Script

You can verify all 30 clips with the inference engine at any time:

```bash
.venv/bin/python3 -c "
import pandas as pd, soundfile as sf
from acforensics.inference.predictor import Predictor
p = Predictor(checkpoint_path='checkpoints/crnn_best.pt', norm_stats_path='checkpoints/normalization_stats.json')
df = pd.read_csv('data/test_dataset/ground_truth_labels.csv')
for _, r in df.iterrows():
    wav, sr = sf.read('data/test_dataset/all_mixed_for_batch_testing/' + r['filename'])
    res = p.predict(wav, sr=sr)
    print(f'{r[\"filename\"]:32} Expected: {r[\"expected_verdict\"]:10} Predicted: {res.verdict.upper():10} P(Gen): {res.prob_genuine:.3f}')
"
```
