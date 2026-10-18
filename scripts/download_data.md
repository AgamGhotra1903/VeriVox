# Dataset Acquisition & Setup Guide: Acoustic Forensics

This document outlines the official sources, directory layout, checksums, and verification steps for downloading and setting up the benchmark and extension datasets under **AF-Split v1**.

---

## 1. ASVspoof 2019 Logical Access (LA)

The primary dataset used for training, validation, and headline benchmarking is the **ASVspoof 2019 Logical Access (LA)** partition.

### Official Source
- **Host:** University of Edinburgh DataShare
- **DOI / URL:** [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)
- **Archive File:** `LA.zip` (~30 GB compressed, ~35 GB uncompressed)
- **MD5 Checksum:** `0a711ff63583688f8d956bf3a40c6c0b`

### Directory Layout
Unpack `LA.zip` so that the directory tree under `data/raw/LA/` matches:

```
data/raw/LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt     # 25,380 trials (A01-A06)
│   ├── ASVspoof2019.LA.cm.dev.trl.txt       # 24,844 trials (A01-A06)
│   └── ASVspoof2019.LA.cm.eval.trl.txt      # 71,237 trials (A07-A19 unseen)
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

# Verify MD5 (optional but recommended)
md5sum LA.zip

# Unzip archive
unzip -q LA.zip
```

---

## 2. In-the-Wild Deepfake Audio (Cross-Domain Test)

Used strictly for zero-shot cross-domain generalization evaluation. **Never used for training.**

### Official Source
- **Publication:** Müller et al., *"Does Audio Deepfake Detection Generalize?"*, Interspeech 2022.
- **Repository / Release:** [https://github.com/piotrkawa/deepfake-whisper-features](https://github.com/piotrkawa/deepfake-whisper-features) or authors' Zenodo release.
- **Dataset Size:** ~37.9 hours, 31,779 clips (bona fide and deepfake speech from public figures).

### Directory Layout
Place the extracted files under `data/raw/in_the_wild/`:

```
data/raw/in_the_wild/
├── meta.csv           # Required Columns: file, speaker, label
└── audio/
    ├── clip_0001.wav
    └── ...
```

---

## 3. Extension Datasets (Optional Multi-Domain, STRETCH)

### 3.1 DFADD (Diffusion & Flow-Matching TTS)
- **Source:** Hugging Face / Zenodo (`media-sec-lab/DFADD`)
- **Key feature:** Focuses on diffusion-based synthesis and flow-matching vocoders.
- **Target layout:** Place under `data/raw/DFADD/` with `meta.csv` containing columns `file, speaker, attack_id, label`.

### 3.2 WaveFake (Vocoder Deepfakes)
- **Source:** Frank & Schönherr, *WaveFake: A Data Set to Facilitate Audio Deepfake Detection*, ACM MM 2021.
- **Key feature:** 10 vocoders on LJSpeech (single speaker) and JSUT (Japanese).
- **Caveat:** Single-speaker confound for English subsets (see `docs/dataset_card.md`).

---

## 4. Synthetic Smoke-Test Dataset (Built-in)

For instant verification, unit testing, and CI without downloading 35 GB of audio:
```bash
python scripts/prepare_data.py --generate-smoke-data
```
This generates synthetic bona fide and spoof audio with standard protocols under `data/smoke_test/`.

---

## 5. Building Manifests and Leakage Auditing

Once datasets are in place (or using `--debug` mode):
```bash
# 1. Build standardized Parquet manifests
python scripts/prepare_manifests.py --debug

# 2. Run automated zero-leakage audit
python scripts/audit_leakage.py --manifests-dir data/manifests
```
