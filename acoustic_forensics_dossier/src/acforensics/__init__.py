"""Acoustic Forensics: Deployable Synthetic Speech & Deepfake Audio Detection.

Academic ML Project by:
"""

import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

__version__ = "1.0.0"

from acforensics.constants import BONAFIDE, SPOOF, SAMPLE_RATE, TARGET_SAMPLES

__all__ = [
    "BONAFIDE",
    "SPOOF",
    "SAMPLE_RATE",
    "TARGET_SAMPLES",
]
