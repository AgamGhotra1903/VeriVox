"""Acoustic Forensics: Deployable Synthetic Speech & Deepfake Audio Detection.

Academic ML Project by:
- Paras Bansal (IIT2024221)
- Jashan Goyal (IIT2024223)
- Agam Ghotra (IIT2024226)
- Chunit Bansal (IIT2024233)
"""

import warnings

# Suppress spurious macOS Accelerate BLAS matmul runtime warnings on Apple Silicon Python 3.13
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

__version__ = "1.0.0"
__author__ = "Paras Bansal, Jashan Goyal, Agam Ghotra, Chunit Bansal"

from acforensics.constants import BONAFIDE, SPOOF, SAMPLE_RATE, TARGET_SAMPLES

__all__ = [
    "BONAFIDE",
    "SPOOF",
    "SAMPLE_RATE",
    "TARGET_SAMPLES",
]
