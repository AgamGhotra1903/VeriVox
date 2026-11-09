import json
from pathlib import Path
import pytest


def test_preprocessing_parity_check_artifact():
    parity_file = Path("results/metrics/parity_check.json")
    assert parity_file.is_file(), "results/metrics/parity_check.json does not exist. Run scripts/parity_check.py first."

    with open(parity_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("parity_passed") is True, f"Parity check failed: {data}"
    assert data.get("n_samples", 0) >= 2000, f"Expected at least 2000 samples, got {data.get('n_samples')}"

    max_diff = data.get("max_absolute_score_difference", 1.0)
    assert max_diff < 0.05, f"Max score difference {max_diff} exceeded tolerance 0.05"

    eer_diff = data.get("eer_difference_pts", 100.0)
    assert eer_diff < 0.5, f"EER difference {eer_diff} exceeded tolerance 0.5 pts"

    cached_eer = data.get("cached_path_eer")
    raw_eer = data.get("raw_audio_path_eer")
    assert abs(cached_eer - raw_eer) < 0.1, f"Cached EER {cached_eer} and raw EER {raw_eer} differ"

    ci = data.get("cached_path_eer_ci_95", [0.0, 100.0])
    assert ci[0] <= raw_eer <= ci[1], f"Raw EER {raw_eer} not within cached 95% CI {ci}"
