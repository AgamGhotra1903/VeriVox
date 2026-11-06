"""Unit tests for ASVspoof 2019 LA and In-The-Wild protocol parsing.

Author: Paras Bansal (IIT2024221)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.data.protocols import (
    parse_asvspoof_line,
    load_asvspoof_protocol,
    load_in_the_wild_protocol,
    get_protocol_statistics,
)
from acforensics.evaluation.metrics import compute_roc_auc, compute_eer


def test_parse_asvspoof_line_bonafide():
    line = "LA_0079 LA_D_1047731 - - bonafide\n"
    res = parse_asvspoof_line(line)
    assert res is not None
    assert res["speaker_id"] == "LA_0079"
    assert res["file_id"] == "LA_D_1047731"
    assert res["attack_id"] == "-"
    assert res["key"] == "bonafide"
    assert res["label"] == BONAFIDE


def test_parse_asvspoof_line_spoof():
    line = "LA_0079 LA_D_1105538 - A06 spoof\n"
    res = parse_asvspoof_line(line)
    assert res is not None
    assert res["speaker_id"] == "LA_0079"
    assert res["file_id"] == "LA_D_1105538"
    assert res["attack_id"] == "A06"
    assert res["key"] == "spoof"
    assert res["label"] == SPOOF


def test_parse_asvspoof_line_invalid():
    # Incomplete line
    assert parse_asvspoof_line("LA_0079 LA_D_1105538") is None
    # Empty string
    assert parse_asvspoof_line("   \n") is None
    # Unknown key
    assert parse_asvspoof_line("LA_0079 LA_D_1105538 - A06 unknown_key") is None


def test_load_asvspoof_protocol_from_file(tmp_path: Path):
    proto_file = tmp_path / "sample_protocol.txt"
    proto_file.write_text(
        "LA_0001 LA_T_0001 - - bonafide\n"
        "LA_0001 LA_T_0002 - A01 spoof\n"
        "LA_0002 LA_T_0003 - A02 spoof\n"
    )

    df = load_asvspoof_protocol(proto_file, split_name="train")
    assert len(df) == 3
    assert list(df["label"]) == [BONAFIDE, SPOOF, SPOOF]
    assert df["split"].iloc[0] == "train"

    stats = get_protocol_statistics(df)
    assert stats["total"] == 3
    assert stats["bonafide_count"] == 1
    assert stats["spoof_count"] == 2
    assert stats["unique_speakers"] == 2
    assert stats["attack_counts"] == {"A01": 1, "A02": 1}


def test_load_in_the_wild_protocol(tmp_path: Path):
    meta_file = tmp_path / "meta.csv"
    meta_file.write_text(
        "file,speaker,label\n"
        "clip1.wav,speaker_a,bona-fide\n"
        "clip2.wav,speaker_b,spoof\n"
    )

    df = load_in_the_wild_protocol(meta_file)
    assert len(df) == 2
    assert df["label"].iloc[0] == BONAFIDE
    assert df["label"].iloc[1] == SPOOF
    assert df["split"].iloc[0] == "cross_domain"


def test_in_the_wild_label_polarity_guard(tmp_path: Path):
    """Guarantees ITW labels and countermeasure scores adhere to positive-class=bonafide convention.

    Fails if:
    1. 'bona-fide' is not mapped to BONAFIDE (1).
    2. 'spoof' is not mapped to SPOOF (0).
    3. Scores where bona-fide > spoof fail to produce ROC-AUC > 0.5.
    """
    assert BONAFIDE == 1, "Convention violation: BONAFIDE must be 1"
    assert SPOOF == 0, "Convention violation: SPOOF must be 0"

    meta_file = tmp_path / "polarity_test_meta.csv"
    meta_file.write_text(
        "file,speaker,label\n"
        "1.wav,speaker1,bona-fide\n"
        "2.wav,speaker2,spoof\n"
        "3.wav,speaker1,bona-fide\n"
        "4.wav,speaker2,spoof\n"
    )

    df = load_in_the_wild_protocol(meta_file)
    labels = df["label"].to_numpy()

    assert labels[0] == 1, "First sample 'bona-fide' must be label 1"
    assert labels[1] == 0, "Second sample 'spoof' must be label 0"
    assert labels[2] == 1, "Third sample 'bona-fide' must be label 1"
    assert labels[3] == 0, "Fourth sample 'spoof' must be label 0"

    # Higher scores for bona-fide (e.g. +2.0 vs -2.0)
    ideal_scores = np.where(labels == BONAFIDE, 2.0, -2.0)
    auc = compute_roc_auc(labels, ideal_scores)
    assert auc == 1.0, f"Expected perfect AUC=1.0 for bona-fide positive convention, got {auc}"

    # Verify that inverted labels would flip AUC below 0.5
    inverted_labels = 1 - labels
    inverted_auc = compute_roc_auc(inverted_labels, ideal_scores)
    assert inverted_auc == 0.0, f"Expected inverted AUC=0.0 when polarity flipped, got {inverted_auc}"
