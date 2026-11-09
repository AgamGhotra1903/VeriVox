from pathlib import Path
import pytest
from scripts.verify_report_numbers import parse_and_verify_report

def test_report_numbers_consistency():
    repo_root = Path(__file__).resolve().parent.parent
    report_candidates = [
        repo_root / "handover" / "final" / "REPORT.md",
        repo_root / "docs" / "REPORT.md",
        repo_root / "EVALUATION_REPORT.md",
    ]
    target = None
    for cand in report_candidates:
        if cand.exists():
            target = cand
            break

    if target is None:
        pytest.skip("Neither docs/REPORT.md nor EVALUATION_REPORT.md found for verification test.")

    mismatches = parse_and_verify_report(target, repo_root=repo_root)
    assert len(mismatches) == 0, f"Found {len(mismatches)} discrepancies between report and JSON metrics: {mismatches}"
