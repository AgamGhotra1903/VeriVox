"""Unit tests for configuration loading and validation.

Author: Agam Ghotra (IIT2024226)
"""

from pathlib import Path
import pytest
from acforensics.config import load_config, MainConfig


def test_load_default_config():
    config_path = Path("configs/default.yaml")
    if not config_path.exists():
        pytest.skip("configs/default.yaml not present")

    cfg = load_config(config_path)
    assert isinstance(cfg, MainConfig)
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.target_length_sec == 4.0
    assert cfg.features.feature_type in ("lfcc", "logmel", "combined")
    assert cfg.seed == 42


def test_load_baselines_config():
    config_path = Path("configs/baselines.yaml")
    if not config_path.exists():
        pytest.skip("configs/baselines.yaml not present")

    cfg = load_config(config_path)
    assert isinstance(cfg, MainConfig)
    assert hasattr(cfg, "audio")
    assert cfg.audio.sample_rate == 16000
    assert "models" in cfg.raw_dict


def test_load_crnn_config():
    config_path = Path("configs/crnn.yaml")
    if not config_path.exists():
        pytest.skip("configs/crnn.yaml not present")

    cfg = load_config(config_path)
    assert isinstance(cfg, MainConfig)
    assert "model" in cfg.raw_dict
    assert cfg.raw_dict["model"]["cnn"]["channels"] == [32, 64, 128, 128]
    assert cfg.raw_dict["model"]["gru"]["hidden_size"] == 128


def test_load_robustness_config():
    config_path = Path("configs/robustness.yaml")
    if not config_path.exists():
        pytest.skip("configs/robustness.yaml not present")

    cfg = load_config(config_path)
    assert isinstance(cfg, MainConfig)
    assert cfg.audio.sample_rate == 16000


def test_load_nonexistent_config_falls_back_or_raises():
    with pytest.raises((FileNotFoundError, Exception)):
        load_config("configs/non_existent_file_xyz.yaml")
