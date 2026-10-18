"""Configuration dataclasses, YAML loader, and validation.

Author: Agam Ghotra (IIT2024226)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml
import torch

from acforensics.constants import (
    DEFAULT_CONFIG_PATH,
    SAMPLE_RATE,
    TARGET_DURATION_SEC,
    TARGET_SAMPLES,
    MIN_DURATION_SEC,
)


@dataclass
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    target_length_sec: float = TARGET_DURATION_SEC
    target_length_samples: int = TARGET_SAMPLES
    min_duration_sec: float = MIN_DURATION_SEC
    normalization_method: str = "peak"  # 'peak', 'rms', 'none'
    vad_silence_trim: bool = False
    eval_crop_strategy: str = "center"  # 'center', 'multi_crop'
    multi_crop_overlap_ratio: float = 0.5


@dataclass
class LFCCConfig:
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_filter: int = 20
    n_lfcc: int = 20
    with_deltas: bool = True


@dataclass
class LogMelConfig:
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    power: float = 2.0
    top_db: float = 80.0


@dataclass
class FeaturesConfig:
    feature_type: str = "lfcc"  # 'lfcc', 'logmel', 'combined'
    normalization: str = "global_meanstd"  # 'global_meanstd', 'per_utt_cmvn'
    stats_file: str = "checkpoints/normalization_stats.json"
    lfcc: LFCCConfig = field(default_factory=LFCCConfig)
    logmel: LogMelConfig = field(default_factory=LogMelConfig)


@dataclass
class DataConfig:
    asvspoof_root: str = "data/raw/LA"
    protocols_dir: str = "data/raw/LA/ASVspoof2019_LA_cm_protocols"
    train_protocol: str = "ASVspoof2019.LA.cm.train.trn.txt"
    dev_protocol: str = "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_protocol: str = "ASVspoof2019.LA.cm.eval.trl.txt"
    train_audio_dir: str = "ASVspoof2019_LA_train/flac"
    dev_audio_dir: str = "ASVspoof2019_LA_dev/flac"
    eval_audio_dir: str = "ASVspoof2019_LA_eval/flac"
    asv_eval_score_file: str = "data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    asv_dev_score_file: str = "data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
    in_the_wild_root: str = "data/raw/in_the_wild"
    in_the_wild_meta: str = "data/raw/in_the_wild/meta.csv"
    itw_cal_ratio: float = 0.15
    manifests_dir: str = "data/manifests"
    compressed_data_dir: str = "data/compressed"
    feature_cache_dir: str = "data/cache"


@dataclass
class SmokeTestConfig:
    enabled: bool = False
    data_dir: str = "data/smoke_test"
    train_samples: int = 200
    dev_samples: int = 100
    eval_samples: int = 100
    epochs: int = 2
    batch_size: int = 16


@dataclass
class PathsConfig:
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"
    metrics_dir: str = "results/metrics"
    tables_dir: str = "results/tables"
    figures_dir: str = "results/figures"
    logs_dir: str = "results/logs"


@dataclass
class MainConfig:
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42, 1337, 2024])
    device: str = "auto"
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    data: DataConfig = field(default_factory=DataConfig)
    smoke_test: SmokeTestConfig = field(default_factory=SmokeTestConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    raw_dict: dict[str, Any] = field(default_factory=dict)

    def get_resolved_device(self) -> torch.device:
        """Resolves target PyTorch device respecting hardware availability."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.device in ("mps", "auto") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        elif self.device == "cpu":
            return torch.device("cpu")
        else:
            return torch.device("cpu")


def _dict_to_dataclass(cls, data: dict[str, Any]) -> Any:
    """Helper to instantiate nested dataclasses from nested dictionaries."""
    if not isinstance(data, dict):
        return data
    fields = cls.__dataclass_fields__
    kwargs = {}
    for f_name, f_def in fields.items():
        if f_name in data:
            val = data[f_name]
            if hasattr(f_def.type, "__dataclass_fields__"):
                kwargs[f_name] = _dict_to_dataclass(f_def.type, val)
            else:
                kwargs[f_name] = val
    return cls(**kwargs)


def load_config(config_path: Optional[Path | str] = None) -> MainConfig:
    """Loads a YAML configuration file and parses it into a strongly typed MainConfig.

    Args:
        config_path: Path to YAML config file. Defaults to 'configs/default.yaml'.

    Returns:
        MainConfig instance.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    audio_cfg = _dict_to_dataclass(AudioConfig, raw_dict.get("audio", {}))

    feat_raw = raw_dict.get("features", {})
    lfcc_cfg = _dict_to_dataclass(LFCCConfig, feat_raw.get("lfcc", {}))
    logmel_cfg = _dict_to_dataclass(LogMelConfig, feat_raw.get("logmel", {}))
    features_cfg = FeaturesConfig(
        feature_type=feat_raw.get("feature_type", "lfcc"),
        normalization=feat_raw.get("normalization", "global_meanstd"),
        stats_file=feat_raw.get("stats_file", "checkpoints/normalization_stats.json"),
        lfcc=lfcc_cfg,
        logmel=logmel_cfg,
    )

    data_cfg = _dict_to_dataclass(DataConfig, raw_dict.get("data", {}))
    smoke_cfg = _dict_to_dataclass(SmokeTestConfig, raw_dict.get("smoke_test", {}))
    paths_cfg = _dict_to_dataclass(PathsConfig, raw_dict.get("paths", {}))

    return MainConfig(
        seed=raw_dict.get("seed", 42),
        seeds=raw_dict.get("seeds", [42, 1337, 2024]),
        device=raw_dict.get("device", "auto"),
        audio=audio_cfg,
        features=features_cfg,
        data=data_cfg,
        smoke_test=smoke_cfg,
        paths=paths_cfg,
        raw_dict=raw_dict,
    )
