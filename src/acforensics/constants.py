"""Project-wide constants, labels, audio parameters, attack taxonomy, and schemas.

Team:
"""

from pathlib import Path

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

SAMPLE_RATE: int = 16000
TARGET_DURATION_SEC: float = 4.0
TARGET_SAMPLES: int = int(SAMPLE_RATE * TARGET_DURATION_SEC)
MIN_DURATION_SEC: float = 0.5
MIN_SAMPLES: int = int(SAMPLE_RATE * MIN_DURATION_SEC)

LFCC_DEFAULT_DIM: int = 60
LOGMEL_DEFAULT_DIM: int = 80
DEFAULT_TIME_FRAMES: int = 400

# ASVspoof 2019 LA Attack Taxonomy
KNOWN_ATTACKS: list[str] = [f"A{i:02d}" for i in range(1, 7)]

UNSEEN_ATTACKS: list[str] = [f"A{i:02d}" for i in range(7, 20)]

ALL_ATTACKS: list[str] = KNOWN_ATTACKS + UNSEEN_ATTACKS

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

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_DATA_CONFIG_PATH = Path("configs/data.yaml")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_CHECKPOINTS_DIR = Path("checkpoints")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MANIFESTS_DIR = Path("data/manifests")
DEFAULT_LOGS_DIR = Path("results/logs")
