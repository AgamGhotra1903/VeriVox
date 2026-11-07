from pathlib import Path
import pytest
from scripts.verify_report_numbers import verify_all_report_numbers

def test_report_numbers_match_json_ground_truth():
    assert verify_all_report_numbers(), "Report numbers do not match underlying metrics JSON ground truth!"
