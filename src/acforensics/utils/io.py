"""Atomic and type-safe I/O utilities for saving/loading experiment artifacts.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

import json
from pathlib import Path
from typing import Any
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder converting NumPy types to standard Python primitives."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: dict[str, Any], path: Path | str, indent: int = 2) -> None:
    """Atomically serializes dictionary to JSON file using temporary file swap.

    Args:
        data: Python dictionary containing serializable values.
        path: Destination file path.
        indent: Indentation level for pretty-printing.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_file = dest.with_suffix(dest.suffix + ".tmp")

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)

    temp_file.replace(dest)


def load_json(path: Path | str) -> dict[str, Any]:
    """Loads and deserializes JSON file.

    Args:
        path: Path to target JSON file.

    Returns:
        Loaded dictionary.

    Raises:
        FileNotFoundError: If the target file does not exist.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"JSON artifact not found at: {src}")
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_write_text(content: str, path: Path | str) -> None:
    """Safely writes text to destination ensuring parent directories exist."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(content)
