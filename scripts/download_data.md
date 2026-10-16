# Dataset Acquisition Guide: ASVspoof 2019 LA & In-the-Wild

This document outlines the official sources, directory layout, and verification steps for downloading and setting up the primary and cross-domain datasets.

---

## 1. ASVspoof 2019 Logical Access (LA)

The primary dataset used for training, validation, and benchmarking is the **ASVspoof 2019 Logical Access (LA)** partition.

### Official Source
- **Host:** University of Edinburgh DataShare
- **DOI / URL:** [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)
- **Archive File:** `LA.zip` (~30 GB compressed, ~35 GB uncompressed)

### Directory Layout
Unpack `LA.zip` so that the directory tree under `data/raw/LA/` matches:

```
data/raw/LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_asv_scores/
│   ├── ASVspoof2019.LA.asv.dev.gi.trl.scores.txt
│   └── ASVspoof2019.LA.asv.eval.gi.trl.scores.txt
├── ASVspoof2019_LA_train/
│   └── flac/
│       ├── LA_T_1000137.flac
│       └── ... (25,380 FLAC files)
├── ASVspoof2019_LA_dev/
│   └── flac/
│       ├── LA_D_1047731.flac
│       └── ... (24,844 FLAC files)
└── ASVspoof2019_LA_eval/
    └── flac/
        ├── LA_E_1000147.flac
        └── ... (71,196 FLAC files)
```

### Command-line Quick Download (Linux / Mac / Colab)
```bash
mkdir -p data/raw/LA
cd data/raw/LA

# Download using curl or wget
curl -L -O https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip

# Unzip archive
unzip -q LA.zip
```

---

## 2. In-the-Wild Deepfake Audio (Cross-Domain Test)

Used strictly for zero-shot cross-domain generalization evaluation. **Never use for training.**

### Official Source
- **Publication:** Müller et al., *"Does Audio Deepfake Detection Generalize?"*, Interspeech 2022.
- **Repository / Release:** [https://github.com/piotrkawa/deepfake-whisper-features](https://github.com/piotrkawa/deepfake-whisper-features) or authors' repository.
- **Dataset Size:** ~37.9 hours, 31,779 clips (bona fide and deepfake speech from public figures).

### Directory Layout
Place the extracted files under `data/raw/in_the_wild/`:

```
data/raw/in_the_wild/
├── meta.csv           # Columns: file, speaker, label
└── audio/
    ├── clip_0001.wav
    └── ...
```

---

## 3. Synthetic Smoke-Test Dataset (Built-in)

For instant verification, unit testing, and continuous integration without downloading 35 GB of audio:
```bash
python scripts/prepare_data.py --generate-smoke-data
```
This generates synthetic bona fide and spoof audio with standard protocols under `data/smoke_test/`.
