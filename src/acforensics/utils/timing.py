"""Execution timing and benchmark utilities.

Author: Chunit Bansal (IIT2024233)
"""

import time
from typing import Optional


class Timer:
    """Context manager and stopwatch for measuring execution durations."""

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    def get_elapsed_ms(self) -> float:
        """Returns elapsed time in milliseconds."""
        return self.elapsed * 1000.0

    def get_elapsed_sec(self) -> float:
        """Returns elapsed time in seconds."""
        return self.elapsed
