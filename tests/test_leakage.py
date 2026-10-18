"""Unit tests for leakage detection, speaker disjointness, and content hashing.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

from pathlib import Path
import pandas as pd
import pytest

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.data.leakage import (
    audit_speaker_disjointness,
    audit_file_id_disjointness,
    audit_audio_hash_collisions,
    check_vctk_heritage_warnings,
    run_comprehensive_leakage_audit,
    compute_file_content_hash,
)


def test_audit_speaker_disjointness_clean():
    """Verifies that disjoint splits pass the audit with zero violations."""
    splits = {
        "train": pd.DataFrame({"speaker": ["spk_01", "spk_02", "spk_03"], "label": [1, 0, 0]}),
        "dev": pd.DataFrame({"speaker": ["spk_04", "spk_05"], "label": [1, 0]}),
        "eval": pd.DataFrame({"speaker": ["spk_06", "spk_07", "spk_08"], "label": [1, 0, 0]}),
    }

    res = audit_speaker_disjointness(splits)
    assert res["passed"] is True
    assert len(res["violations"]) == 0


def test_audit_speaker_disjointness_detects_leakage():
    """Verifies that speaker overlap between train and dev fails the audit."""
    splits = {
        "train": pd.DataFrame({"speaker": ["spk_01", "spk_02", "spk_LEAK"], "label": [1, 0, 0]}),
        "dev": pd.DataFrame({"speaker": ["spk_04", "spk_LEAK"], "label": [1, 0]}),
    }

    res = audit_speaker_disjointness(splits)
    assert res["passed"] is False
    assert len(res["violations"]) > 0
    assert "spk_LEAK" in str(res["violations"])


def test_audit_file_id_disjointness():
    """Verifies detection of file/utterance ID overlap across splits."""
    splits_clean = {
        "train": pd.DataFrame({"file_id": ["f1", "f2"], "speaker": ["s1", "s2"]}),
        "dev": pd.DataFrame({"file_id": ["f3", "f4"], "speaker": ["s3", "s4"]}),
    }
    assert audit_file_id_disjointness(splits_clean)["passed"] is True

    splits_leaked = {
        "train": pd.DataFrame({"file_id": ["f1", "f2_shared"], "speaker": ["s1", "s2"]}),
        "dev": pd.DataFrame({"file_id": ["f2_shared", "f4"], "speaker": ["s3", "s4"]}),
    }
    res_leak = audit_file_id_disjointness(splits_leaked)
    assert res_leak["passed"] is False
    assert len(res_leak["violations"]) > 0


def test_audit_content_hash_collision(tmp_path: Path):
    """Verifies that physical audio content duplicates across splits are caught."""
    file_a = tmp_path / "audio_a.wav"
    file_b = tmp_path / "audio_b.wav"
    file_c = tmp_path / "audio_c.wav"

    # Write identical bytes to file_a and file_b
    dummy_audio_bytes = b"RIFF____WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    file_a.write_bytes(dummy_audio_bytes)
    file_b.write_bytes(dummy_audio_bytes)
    file_c.write_bytes(dummy_audio_bytes + b"distinct_tail_data")

    splits = {
        "train": pd.DataFrame({"path": [str(file_a)], "speaker": ["s1"]}),
        "eval": pd.DataFrame({"path": [str(file_b)], "speaker": ["s2"]}),
        "dev": pd.DataFrame({"path": [str(file_c)], "speaker": ["s3"]}),
    }

    res = audit_audio_hash_collisions(splits, max_files_per_split=10)
    assert res["passed"] is False
    assert len(res["collisions"]) == 1
    assert set(res["collisions"][0]["splits"]) == {"train", "eval"}


def test_vctk_heritage_warnings():
    """Verifies that VCTK speaker naming triggers explicit documentation caveats."""
    splits = {
        "train": pd.DataFrame({"speaker": ["p225", "p226"], "label": [1, 1]}),
    }
    warnings = check_vctk_heritage_warnings(splits)
    assert any("VCTK" in w for w in warnings)


def test_run_comprehensive_leakage_audit_clean():
    """Integration test for full audit on clean partitions."""
    splits = {
        "train": pd.DataFrame({"speaker": ["s1", "s2"], "file_id": ["t1", "t2"], "label": [1, 0]}),
        "dev": pd.DataFrame({"speaker": ["s3", "s4"], "file_id": ["d1", "d2"], "label": [1, 0]}),
        "eval": pd.DataFrame({"speaker": ["s5", "s6"], "file_id": ["e1", "e2"], "label": [1, 0]}),
    }

    report = run_comprehensive_leakage_audit(splits, check_hashes=False)
    assert report["status"] == "PASS"
    assert report["overall_passed"] is True
