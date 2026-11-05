"""Structured logging configuration for Acoustic Forensics.

Author: Agam Ghotra (IIT2024226)
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[Path | str] = None,
    level: int = logging.INFO,
    format_str: str = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configures the root logger with both console and optional file handlers.

    Args:
        log_file: Optional path to write output logs. Parent directory is created automatically.
        level: Logging verbosity level (default: logging.INFO).
        format_str: Format string for log messages.
        date_format: Timestamp formatting pattern.
    """
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Stream (console) handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor to obtain a named child logger.

    Args:
        name: Name of module / component requesting logger.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)
