import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from acforensics.evaluation.provenance import (
    compute_file_sha256,
    get_git_commit,
    get_git_diff_hash,
    create_provenance_record,
    guard_no_smoke_test,
)
from acforensics.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

EXPECTED_TRIAL_COUNTS = {
    "train": 25380,
    "dev": 24844,
    "eval": 71237,
}


def check_free_disk_space_gb(target_path: Path | str = ".") -> float:
    total, used, free = shutil.disk_usage(str(target_path))
    return float(free / (1024 ** 3))


def verify_protocol_file(file_path: Path) -> dict[str, Any]:
    if not file_path.is_file():
        return {"exists": False, "trials": 0, "sha256": None}
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    sha = compute_file_sha256(file_path)
    return {"exists": True, "trials": count, "sha256": sha}


def verify_asv_score_file(file_path: Path) -> dict[str, Any]:
    if not file_path.is_file():
        return {"exists": False, "target": 0, "nontarget": 0, "spoof": 0}
    n_tar = 0
    n_non = 0
    n_sp = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                t_type = parts[1].lower() if len(parts) == 3 else parts[2].lower()
                if t_type == "target":
                    n_tar += 1
                elif t_type == "nontarget":
                    n_non += 1
                elif t_type == "spoof":
                    n_sp += 1
    return {
        "exists": True,
        "target": n_tar,
        "nontarget": n_non,
        "spoof": n_sp,
        "sha256": compute_file_sha256(file_path),
    }


def verify_datasets(
    raw_dir: Path | str = "data/raw",
    output_report_path: Optional[Path | str] = "results/audit/dataset_verification.json",
    min_free_gb: float = 25.0,
    seed: int = 42,
) -> dict[str, Any]:
    raw_p = Path(raw_dir)
    guard_no_smoke_test(str(raw_p), context="raw_dir")

    free_gb = check_free_disk_space_gb(raw_p.parent if raw_p.exists() else ".")
    disk_ok = free_gb >= min_free_gb

    la_dir = raw_p / "LA" if (raw_p / "LA").exists() else raw_p
    cm_dir = la_dir / "ASVspoof2019_LA_cm_protocols"
    asv_dir = la_dir / "ASVspoof2019_LA_asv_scores"

    train_proto = cm_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_proto = cm_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_proto = cm_dir / "ASVspoof2019.LA.cm.eval.trl.txt"

    proto_results = {
        "train": verify_protocol_file(train_proto),
        "dev": verify_protocol_file(dev_proto),
        "eval": verify_protocol_file(eval_proto),
    }

    asv_dev = asv_dir / "ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
    asv_eval = asv_dir / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    asv_results = {
        "dev": verify_asv_score_file(asv_dev),
        "eval": verify_asv_score_file(asv_eval),
    }

    train_flac = la_dir / "ASVspoof2019_LA_train" / "flac"
    dev_flac = la_dir / "ASVspoof2019_LA_dev" / "flac"
    eval_flac = la_dir / "ASVspoof2019_LA_eval" / "flac"

    audio_counts = {
        "train": len(list(train_flac.glob("*.flac"))) if train_flac.exists() else 0,
        "dev": len(list(dev_flac.glob("*.flac"))) if dev_flac.exists() else 0,
        "eval": len(list(eval_flac.glob("*.flac"))) if eval_flac.exists() else 0,
    }

    itw_dir = raw_p / "in_the_wild"
    itw_meta = itw_dir / "meta.csv"
    itw_audio = itw_dir / "audio"
    itw_exists = itw_meta.is_file() and itw_audio.is_dir()
    itw_count = 0
    itw_audio_count = 0
    if itw_exists:
        with open(itw_meta, "r", encoding="utf-8") as f:
            itw_count = max(0, sum(1 for _ in f) - 1)
        itw_audio_count = len(list(itw_audio.glob("*.wav")))

    protocols_match = all(
        proto_results[s]["trials"] == EXPECTED_TRIAL_COUNTS[s]
        for s in ("train", "dev", "eval")
    )
    asv_valid = asv_results["eval"]["target"] > 0 and asv_results["eval"]["nontarget"] > 0
    audio_present = all(
        audio_counts[s] == EXPECTED_TRIAL_COUNTS[s]
        for s in ("train", "dev", "eval")
    )

    all_passed = disk_ok and protocols_match and asv_valid and audio_present and itw_exists

    commit = get_git_commit()
    is_dirty, diff_h = get_git_diff_hash()
    ref_file = eval_proto if eval_proto.is_file() else (train_proto if train_proto.is_file() else Path("README.md"))
    manifest_sha = compute_file_sha256(ref_file)

    report = {
        "status": "PASS" if all_passed else "INCOMPLETE",
        "dataset_name": "asvspoof2019_la_and_itw",
        "manifest_sha256": manifest_sha,
        "git_commit": commit,
        "git_dirty": is_dirty,
        "diff_hash": diff_h,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "disk_free_gb": round(free_gb, 2),
        "disk_threshold_gb": float(min_free_gb),
        "disk_check_passed": disk_ok,
        "protocols": proto_results,
        "asv_scores": asv_results,
        "audio_files": audio_counts,
        "in_the_wild": {
            "meta_exists": itw_meta.is_file(),
            "meta_trials": itw_count,
            "audio_files": itw_audio_count,
        },
        "protocols_match_expected": protocols_match,
        "asv_scores_valid": asv_valid,
        "audio_present": audio_present,
    }

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Dataset verification report written to %s", out_p)

    return report


def main():
    parser = argparse.ArgumentParser(description="ASVspoof 2019 LA & In-the-Wild Dataset Verification")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw datasets directory")
    parser.add_argument("--out-report", type=str, default="results/audit/dataset_verification.json", help="Report destination")
    parser.add_argument("--min-free-gb", type=float, default=25.0, help="Required disk space threshold in GB")
    parser.add_argument("--seed", type=int, default=42, help="RNG Seed for reproducibility tracking")
    args = parser.parse_args()

    setup_logging()
    rep = verify_datasets(
        raw_dir=args.raw_dir,
        output_report_path=args.out_report,
        min_free_gb=args.min_free_gb,
        seed=args.seed,
    )
    print(f"Dataset verification status: {rep['status']}")


if __name__ == "__main__":
    main()
