"""Project-wide constants, labels, audio parameters, attack taxonomy, and schemas.

Team:
- Paras Bansal (IIT2024221): Data, normalization, features
- Jashan Goyal (IIT2024223): Classical baselines and benchmarking
- Agam Ghotra (IIT2024226): CRNN/SSL models, training, inference
- Chunit Bansal (IIT2024233): Robustness, metrics, error analysis, UI/demo
"""

from pathlib import Path

# ==============================================================================
# Binary Classification Labels
# CRITICAL CONVENTION: BONAFIDE = 1, SPOOF = 0 across all models, metrics, and protocols.
# ==============================================================================
BONAFIDE: int = 1
SPOOF: int = 0

LABEL_STR_MAP: dict[str, int] = {
    "bonafide": BONAFIDE,
    "bona-fide": BONAFIDE,
    "genuine": BONAFIDE,
    "real": BONAFIDE,
    "human": BONAFIDE,
    "spoof": SPOOF,
    "fake": SPOOF,
    "synthetic": SPOOF,
    "deepfake": SPOOF,
}

INV_LABEL_STR_MAP: dict[int, str] = {
    BONAFIDE: "bonafide",
    SPOOF: "spoof",
}

# ==============================================================================
# Audio Processing Specifications
# Standardized across entire Acoustic Forensics pipeline: 16 kHz mono.
# ==============================================================================
SAMPLE_RATE: int = 16000
TARGET_DURATION_SEC: float = 4.0
TARGET_SAMPLES: int = int(SAMPLE_RATE * TARGET_DURATION_SEC)  # 64,000 samples
MIN_DURATION_SEC: float = 0.5
MIN_SAMPLES: int = int(SAMPLE_RATE * MIN_DURATION_SEC)        # 8,000 samples

# ==============================================================================
# Feature Extraction Default Dimensions
# LFCC: 20 static + 20 delta + 20 delta-delta = 60 coefficients
# Log-Mel: 80 filterbank channels (0-8000 Hz)
# Time frames for 4.0s (64,000 samples) with win_length=400, hop_length=160
# ==============================================================================
LFCC_DEFAULT_DIM: int = 60
LOGMEL_DEFAULT_DIM: int = 80
DEFAULT_TIME_FRAMES: int = 400

# ==============================================================================
# ASVspoof 2019 LA Attack Taxonomy
# Train & Dev attacks (A01 - A06):
# A01: Neural waveform cloning / TTS
# A02: Vocoder-based TTS
# A03: Waveform concatenation TTS
# A04: Waveform filtering VC
# A05: Vocoder-based VC
# A06: Waveform transform VC
# ==============================================================================
KNOWN_ATTACKS: list[str] = [f"A{i:02d}" for i in range(1, 7)]

# Eval attacks (A07 - A19) - unseen synthesis and voice conversion algorithms
# A07: Neural vocoder TTS; A08: Neural waveform VC; A09: Vocoder TTS
# A10: Neural waveform TTS; A11: Grifin-Lim TTS; A12: Neural waveform TTS
# A13: VC waveform filtering; A14: Waveform filtering VC; A15: Waveform transform VC
# A16: Waveform filtering VC (waveform splice); A17: Vocoder VC; A18: Vocoder VC; A19: Spectral VC
UNSEEN_ATTACKS: list[str] = [f"A{i:02d}" for i in range(7, 20)]

ALL_ATTACKS: list[str] = KNOWN_ATTACKS + UNSEEN_ATTACKS

# ==============================================================================
# Official Protocol Expected Counts ("AF-Split v1" Ground Truth)
# Used for verification, logging, and sanity acceptance checks.
# ==============================================================================
OFFICIAL_COUNTS: dict[str, dict[str, int]] = {
    "asvspoof_train": {
        "bonafide": 2580,
        "spoof": 22800,
        "total": 25380,
    },
    "asvspoof_dev": {
        "bonafide": 2548,
        "spoof": 22296,
        "total": 24844,
    },
    "asvspoof_eval": {
        "bonafide": 7355,
        "spoof": 63882,
        "total": 71237,
    },
    "in_the_wild_total": {
        "bonafide": 19963,
        "spoof": 11816,
        "total": 31779,
    },
}

# ==============================================================================
# Unified Manifest Schema Specification (Parquet / Pandas)
# ==============================================================================
MANIFEST_COLUMNS: list[str] = [
    "path",
    "dataset",
    "split",
    "speaker",
    "label",
    "attack_id",
    "duration",
    "codec",
    "bitrate",
    "gender",
]

# ==============================================================================
# Default Directory Paths (relative to workspace root)
# ==============================================================================
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_DATA_CONFIG_PATH = Path("configs/data.yaml")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_CHECKPOINTS_DIR = Path("checkpoints")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MANIFESTS_DIR = Path("data/manifests")
DEFAULT_LOGS_DIR = Path("results/logs")
