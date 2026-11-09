import json
from pathlib import Path
from typing import Any
import pandas as pd
import pytest

from acforensics.data.leakage import audit_speaker_disjointness
from acforensics.ensemble.stacking import assert_zero_data_leakage
from acforensics.evaluation.provenance import (
    compute_file_sha256,
    get_git_commit,
    create_provenance_record,
    validate_provenance,
    guard_no_smoke_test,
    guard_manifest,
    guard_results_and_provenance,
)
from scripts.build_report_tables import build_tables


def test_provenance_compute_sha256_and_git(tmp_path: Path):
    f = tmp_path / "manifest.csv"
    f.write_text("dummy content")
    sha = compute_file_sha256(f)
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)

    commit = get_git_commit()
    assert len(commit) >= 7

    rec = create_provenance_record(
        dataset_name="asvspoof2019_la",
        manifest_path=f,
        seed=42,
    )
    assert rec["dataset_name"] == "asvspoof2019_la"
    assert rec["manifest_sha256"] == sha
    assert rec["git_commit"] == commit
    assert rec["seed"] == 42
    assert "timestamp" in rec

    validate_provenance(rec)


def test_validate_provenance_missing_field():
    rec = {
        "dataset_name": "asvspoof2019_la",
        "manifest_sha256": "a" * 64,
        "git_commit": "1234567890abcdef",
        "seed": 42,
    }
    with pytest.raises(ValueError, match="Missing required provenance field 'timestamp'"):
        validate_provenance(rec)


def test_validate_provenance_invalid_values():
    rec_bad_sha = {
        "dataset_name": "asvspoof2019_la",
        "manifest_sha256": "invalid_hash",
        "git_commit": "1234567890abcdef",
        "seed": 42,
        "timestamp": "2026-11-05T00:00:00Z",
    }
    with pytest.raises(ValueError, match="manifest_sha256"):
        validate_provenance(rec_bad_sha)

    rec_bad_seed = {
        "dataset_name": "asvspoof2019_la",
        "manifest_sha256": "a" * 64,
        "git_commit": "1234567890abcdef",
        "seed": "42",
        "timestamp": "2026-11-05T00:00:00Z",
    }
    with pytest.raises(ValueError, match="seed"):
        validate_provenance(rec_bad_seed)


def test_guard_no_smoke_test():
    guard_no_smoke_test({"valid_key": "valid_value", "nested": [1, 2, "clean_audio.wav"]})

    with pytest.raises(ValueError, match="Smoke test artifact rejected"):
        guard_no_smoke_test("data/smoke_test/eval.flac")

    with pytest.raises(ValueError, match="Smoke test artifact rejected"):
        guard_no_smoke_test({"metrics": {"path": "data/smoke_test/manifest.csv"}})

    with pytest.raises(ValueError, match="Smoke test artifact rejected"):
        guard_no_smoke_test(["clean.wav", "data/smoke_test/train.wav"])


def test_guard_manifest(tmp_path: Path):
    clean_manifest = tmp_path / "clean_manifest.csv"
    clean_df = pd.DataFrame({
        "path": ["data/raw/LA_0001.flac", "data/raw/LA_0002.flac"],
        "speaker": ["spk1", "spk2"],
    })
    clean_df.to_csv(clean_manifest, index=False)
    guard_manifest(clean_manifest)

    smoke_manifest = tmp_path / "smoke_manifest.csv"
    smoke_df = pd.DataFrame({
        "path": ["data/smoke_test/LA_T_1000.flac", "data/raw/LA_0002.flac"],
        "speaker": ["spk1", "spk2"],
    })
    smoke_df.to_csv(smoke_manifest, index=False)
    with pytest.raises(ValueError, match="Smoke test artifact rejected"):
        guard_manifest(smoke_manifest)


def test_build_report_tables_fails_on_smoke_manifest(tmp_path: Path):
    smoke_manifest = tmp_path / "smoke_manifest.csv"
    pd.DataFrame({"path": ["data/smoke_test/audio.flac"]}).to_csv(smoke_manifest, index=False)

    out_dir = tmp_path / "results"
    out_dir.mkdir()

    with pytest.raises(ValueError, match="Smoke test artifact rejected"):
        build_tables(results_dir=out_dir, manifest_path=smoke_manifest)


def test_build_report_tables_fails_on_smoke_results_dir():
    with pytest.raises(ValueError):
        build_tables(results_dir="results_smoke")


def test_audit_speaker_disjointness_requires_real_speaker_metadata():
    splits_missing = {
        "train": pd.DataFrame({"speaker": ["unknown", "unknown"], "label": [1, 0]}),
        "dev": pd.DataFrame({"speaker": ["unknown", "unknown"], "label": [1, 0]}),
    }
    res = audit_speaker_disjointness(splits_missing)
    assert res["passed"] is False
    assert any("missing real speaker metadata" in v.lower() for v in res["violations"])


def test_assert_zero_data_leakage_speaker_checks():
    train_ids = ["t1", "t2"]
    dev_ids = ["d1", "d2"]
    eval_ids = ["e1", "e2"]

    assert_zero_data_leakage(
        train_ids, dev_ids, eval_ids,
        train_speakers=["spk_t1", "spk_t2"],
        dev_speakers=["spk_d1", "spk_d2"],
        eval_speakers=["spk_e1", "spk_e2"],
    )

    with pytest.raises(AssertionError, match="speakers shared between TRAIN and DEV"):
        assert_zero_data_leakage(
            train_ids, dev_ids, eval_ids,
            train_speakers=["spk_shared", "spk_t2"],
            dev_speakers=["spk_shared", "spk_d2"],
            eval_speakers=["spk_e1", "spk_e2"],
        )

    with pytest.raises(AssertionError, match="Empty real speaker metadata"):
        assert_zero_data_leakage(
            train_ids, dev_ids, eval_ids,
            train_speakers=["unknown", ""],
            dev_speakers=["spk_d1", "spk_d2"],
            eval_speakers=["spk_e1", "spk_e2"],
        )


def test_all_results_metrics_files_have_provenance_and_sufficient_sample_size():
    metrics_dir = Path("results/metrics")
    if not metrics_dir.is_dir():
        pytest.skip("results/metrics directory not found")

    json_files = list(metrics_dir.glob("*.json"))
    assert len(json_files) > 0, "No metrics JSON files found in results/metrics"

    trial_keys = ("sample_size", "num_trials", "n_trials", "n_samples", "total_trials")

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        guard_results_and_provenance(data, context=str(jf))

        subsample_found = []

        def check_counts(obj: Any):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in trial_keys and isinstance(v, (int, float)):
                        if v < 1000:
                            subsample_found.append((k, v))
                    check_counts(v)

        check_counts(data)

        if subsample_found:
            has_flag = (
                data.get("subset") is True
                or data.get("subset_flag") is True
                or data.get("is_subset") is True
                or "subset" in data
            )
            assert has_flag, (
                f"File {jf.name} reports < 1000 trials ({subsample_found}) "
                f"without an explicit 'subset' flag."
            )
