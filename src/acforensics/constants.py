"""Project-wide constants and conventions.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

from pathlib import Path

# Binary Classification Labels
# CRITICAL CONVENTION: BONAFIDE = 1, SPOOF = 0 throughout all models, metrics, and protocols.
BONAFIDE: int = 1
SPOOF: int = 0

LABEL_STR_MAP: dict[str, int] = {
    "bonafide": BONAFIDE,
    "bona-fide": BONAFIDE,
    "genuine": BONAFIDE,
    "real": BONAFIDE,
    "spoof": SPOOF,
    "fake": SPOOF,
    "synthetic": SPOOF,
}

INV_LABEL_STR_MAP: dict[int, str] = {
    BONAFIDE: "bonafide",
    SPOOF: "spoof",
}

# Audio Parameters
SAMPLE_RATE: int = 16000
TARGET_DURATION_SEC: float = 4.0
TARGET_SAMPLES: int = int(SAMPLE_RATE * TARGET_DURATION_SEC)  # 64,000 samples
MIN_DURATION_SEC: float = 0.5
MIN_SAMPLES: int = int(SAMPLE_RATE * MIN_DURATION_SEC)        # 8,000 samples

# Feature Extraction Default Dimensions
# LFCC: 20 static + 20 delta + 20 delta-delta = 60 coefficients
LFCC_DEFAULT_DIM: int = 60
LOGMEL_DEFAULT_DIM: int = 80
# Frame count for 4.0s (64,000 samples) with win_length=400, hop_length=160
# (64000 - 400) // 160 + 1 = 398 frames (approx 400 frames)
DEFAULT_TIME_FRAMES: int = 400

# ASVspoof 2019 LA Attack Taxonomy
# Train & Dev attacks (A01 - A06):
# A01: Neural waveform cloning / TTS
# A02: Vocoder-based TTS
# A03: Waveform concatenation TTS
# A04: Waveform filtering VC
# A05: Vocoder-based VC
# A06: Waveform transform VC
KNOWN_ATTACKS: list[str] = [f"A{i:02d}" for i in range(1, 7)]

# Eval attacks (A07 - A19) - unseen synthesis and voice conversion algorithms
# A16 is a variant of A04; A17 and A19 relate to A06 variants
UNSEEN_ATTACKS: list[str] = [f"A{i:02d}" for i in range(7, 20)]

ALL_ATTACKS: list[str] = KNOWN_ATTACKS + UNSEEN_ATTACKS

# Default Directory Paths (relative to workspace root)
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_CHECKPOINTS_DIR = Path("checkpoints")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_LOGS_DIR = Path("results/logs")
