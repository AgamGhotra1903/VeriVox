"""Data loading, protocol parsing, audio validation, segmentation, manifests, and leakage auditing."""

from acforensics.data.protocols import (
    load_asvspoof_protocol,
    load_in_the_wild_protocol,
    get_protocol_statistics,
)
from acforensics.data.audio_io import (
    load_and_validate_audio,
    AudioValidationResult,
    resample_audio,
    normalize_audio,
)
from acforensics.data.segmentation import (
    fixed_length_crop_or_pad,
    generate_multi_crop_windows,
)
from acforensics.data.datasets import (
    ASVspoofDataset,
    InTheWildDataset,
    collate_fn,
    get_balanced_sampler,
)
from acforensics.data.splits import (
    split_in_the_wild_by_speaker,
    split_extension_by_generator,
    summarize_splits,
)
from acforensics.data.manifests import (
    build_manifest_from_records,
    save_manifest,
    load_manifest,
    get_manifest_stats,
)
from acforensics.data.leakage import (
    audit_speaker_disjointness,
    audit_file_id_disjointness,
    audit_audio_hash_collisions,
    run_comprehensive_leakage_audit,
)

__all__ = [
    "load_asvspoof_protocol",
    "load_in_the_wild_protocol",
    "get_protocol_statistics",
    "load_and_validate_audio",
    "AudioValidationResult",
    "resample_audio",
    "normalize_audio",
    "fixed_length_crop_or_pad",
    "generate_multi_crop_windows",
    "ASVspoofDataset",
    "InTheWildDataset",
    "collate_fn",
    "get_balanced_sampler",
    "split_in_the_wild_by_speaker",
    "split_extension_by_generator",
    "summarize_splits",
    "build_manifest_from_records",
    "save_manifest",
    "load_manifest",
    "get_manifest_stats",
    "audit_speaker_disjointness",
    "audit_file_id_disjointness",
    "audit_audio_hash_collisions",
    "run_comprehensive_leakage_audit",
]
