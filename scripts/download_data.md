# Dataset Acquisition & Verification Guide: Acoustic Forensics

This document outlines official download sources, high-speed mirrors, exact trial counts, directory layouts, checksums, and verification steps for **ASVspoof 2019 Logical Access (LA)** and the **In-the-Wild Deepfake Audio** dataset.

---

## 1. System Requirements & Disk Space Check

The raw audio archives require approximately **16 GB** download bandwidth and unpack to **~60–70 GB** of uncompressed 16 kHz FLAC/WAV audio.

Before beginning downloads, verify that your machine has at least **80 GB** of free disk space:

```bash
python scripts/verify_datasets.py --min-free-gb 80.0
```

---

## 2. ASVspoof 2019 Logical Access (LA)

The primary dataset used for training, validation, and headline countermeasure benchmarking is the **ASVspoof 2019 Logical Access (LA)** partition.

### Official Source
- **Host:** University of Edinburgh DataShare
- **DOI / URL:** [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)
- **Direct Bitstream URL:** `https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content`
- **Archive File:** `LA.zip` (7,640,952,520 bytes, ~7.64 GB compressed)
- **MD5 Checksum:** `0a711ff63583688f8d956bf3a40c6c0b`

### High-Speed CDN Mirror (Hugging Face)
If Edinburgh DataShare throughput is rate-limited in your region, use the verified mirror repository:
- **Mirror Repository:** `Nemez1z/asvspoof-2019-la` ([https://huggingface.co/datasets/Nemez1z/asvspoof-2019-la](https://huggingface.co/datasets/Nemez1z/asvspoof-2019-la))
- **Parquet Encoded Archive:** `Bisher/ASVspoof_2019_LA` ([https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA))

### Exact Partition Trial Counts
- **Train partition (`ASVspoof2019.LA.cm.train.trn.txt`):** exactly **25,380** trials (Attacks A01–A06, 20 speakers).
- **Development partition (`ASVspoof2019.LA.cm.dev.trl.txt`):** exactly **24,844** trials (Attacks A01–A06, 20 speakers).
- **Evaluation partition (`ASVspoof2019.LA.cm.eval.trl.txt`):** exactly **71,237** trials (Attacks A07–A19, 67 unseen speakers).

### Required Target Directory Layout
Unpack the dataset so that `data/raw/LA/` matches:

```
data/raw/LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt     # 25,380 trials
│   ├── ASVspoof2019.LA.cm.dev.trl.txt       # 24,844 trials
│   └── ASVspoof2019.LA.cm.eval.trl.txt      # 71,237 trials
├── ASVspoof2019_LA_asv_scores/
│   ├── ASVspoof2019.LA.asv.dev.gi.trl.scores.txt
│   └── ASVspoof2019.LA.asv.eval.gi.trl.scores.txt
├── ASVspoof2019_LA_train/
│   └── flac/
│       └── ... (25,380 FLAC files)
├── ASVspoof2019_LA_dev/
│   └── flac/
│       └── ... (24,844 FLAC files)
└── ASVspoof2019_LA_eval/
    └── flac/
        └── ... (71,237 FLAC files)
```

### Download Commands
```bash
mkdir -p data/raw/LA
cd data/raw/LA

# Option A: Direct from Edinburgh DataShare
curl -L -O "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content" -o LA.zip
unzip -q LA.zip

# Option B: High-Speed CLI / Python Sync
cd /Users/agamghotra/Desktop/ML
python scripts/download_data.py --dataset asvspoof2019_la --dest data/raw/LA
```

---

## 3. In-the-Wild Deepfake Audio (Zero-Shot Cross-Domain Benchmark)

Used strictly for zero-shot cross-domain generalization evaluation. **Never used for training.**

### Official Source
- **Publication:** Müller et al., *"Does Audio Deepfake Detection Generalize?"*, Interspeech 2022.
- **Fraunhofer AISEC Portal:** [https://deepfake-demo.aisec.fraunhofer.de/in_the_wild](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild)
- **Direct Hugging Face Download:** `https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip` (8,161,489,023 bytes, ~8.16 GB)
- **Dataset Size:** 31,775 total audio clips across authentic speech and online deepfakes.

### Required Directory Layout
Place the extracted files under `data/raw/in_the_wild/`:

```
data/raw/in_the_wild/
├── meta.csv           # Required Columns: file, speaker, label
└── audio/
    ├── *.wav          # 31,775 audio files
```

### Download Commands
```bash
mkdir -p data/raw/in_the_wild
cd data/raw/in_the_wild

curl -L -O "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
unzip -q release_in_the_wild.zip
```

---

## 4. Verification & Validation Protocol

After downloading and extracting, run the automated verification script:

```bash
python scripts/verify_datasets.py --raw-dir data/raw --out-report results/audit/dataset_verification.json
```

The script audits:
1. Disk space availability ($\ge 80$ GB).
2. Protocol line count conformance (25,380 / 24,844 / 71,237).
3. Availability of the official ASV evaluation scores (`ASVspoof2019.LA.asv.eval.gi.trl.scores.txt`).
4. Presence of all FLAC/WAV audio files without truncation.
5. In-the-Wild speaker metadata formatting.
