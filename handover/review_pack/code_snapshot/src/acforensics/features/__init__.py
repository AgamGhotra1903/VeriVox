"""Feature extraction, normalization, statistical pooling, and quality profile modules."""

from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.normalization import FeatureNormalizer, per_utterance_cmvn
from acforensics.features.statistics import extract_utterance_statistics
from acforensics.features.cache import FeatureCache
from acforensics.features.quality import (
    extract_audio_quality_profile,
    evaluate_reliability_checks,
    compute_clipping_ratio,
    estimate_snr_db,
    compute_spectral_bandwidth_hz,
    compute_silence_ratio,
)

__all__ = [
    "LFCCExtractor",
    "LogMelExtractor",
    "FeatureNormalizer",
    "per_utterance_cmvn",
    "extract_utterance_statistics",
    "FeatureCache",
    "extract_audio_quality_profile",
    "evaluate_reliability_checks",
    "compute_clipping_ratio",
    "estimate_snr_db",
    "compute_spectral_bandwidth_hz",
    "compute_silence_ratio",
]
