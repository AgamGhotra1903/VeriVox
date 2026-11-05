"""Unit tests for ASVspoof 2019 LA and In-The-Wild protocol parsing.

Author: Paras Bansal (IIT2024221)
"""

import pytest
import pandas as pd
from pathlib import Path

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.data.protocols import (
    parse_asvspoof_line,
    load_asvspoof_protocol,
    load_in_the_wild_protocol,
    get_protocol_statistics,
)


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
