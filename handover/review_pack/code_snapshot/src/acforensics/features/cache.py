"""On-disk feature caching utility with config hashing.

Author: Paras Bansal (IIT2024221)
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch

from acforensics.constants import DEFAULT_DATA_DIR
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureCache:
    """Manages on-disk caching of extracted spectrograms and feature tensors.

    Computes a cryptographic hash of the feature extraction parameters and file path
    to ensure cache invalidation whenever parameters change.
    """

    def __init__(self, cache_dir: Optional[Path | str] = None, enabled: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_DATA_DIR / "cache"
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, audio_path: str, config_dict: dict[str, Any]) -> str:
        """Computes unique MD5 hash for the audio path and feature configuration."""
        config_str = json.dumps(config_dict, sort_keys=True)
        key_raw = f"{audio_path}_{config_str}"
        return hashlib.md5(key_raw.encode("utf-8")).hexdigest()

    def get(self, audio_path: str, config_dict: dict[str, Any]) -> Optional[torch.Tensor]:
        """Retrieves cached tensor if available."""
        if not self.enabled:
            return None
        cache_key = self._compute_key(audio_path, config_dict)
        cache_file = self.cache_dir / f"{cache_key}.pt"
        if cache_file.exists():
            try:
                return torch.load(str(cache_file), weights_only=True)
            except Exception as e:
                logger.warning("Failed to load cache file %s: %s", cache_file, e)
                return None
        return None

    def put(self, audio_path: str, config_dict: dict[str, Any], tensor: torch.Tensor) -> None:
        """Saves feature tensor to disk cache."""
        if not self.enabled:
            return
        cache_key = self._compute_key(audio_path, config_dict)
        cache_file = self.cache_dir / f"{cache_key}.pt"
        try:
            torch.save(tensor.detach().cpu(), str(cache_file))
        except Exception as e:
            logger.warning("Failed to write cache file %s: %s", cache_file, e)
