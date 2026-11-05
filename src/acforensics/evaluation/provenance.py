import hashlib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

PROVENANCE_REQUIRED_FIELDS = (
    "dataset_name",
    "manifest_sha256",
    "git_commit",
    "seed",
    "timestamp",
)


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for sha256 computation: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit(repo_path: Union[str, Path] = ".") -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res.stdout.strip()
        if commit:
            return commit
    except Exception:
        pass
    env_commit = os.getenv("GIT_COMMIT", "").strip()
    if env_commit:
        return env_commit
    raise RuntimeError("Unable to determine git commit hash.")


def create_provenance_record(
    dataset_name: str,
    manifest_path: Union[str, Path],
    seed: int,
    repo_path: Union[str, Path] = ".",
) -> dict[str, Any]:
    if not dataset_name or not isinstance(dataset_name, str):
        raise ValueError("dataset_name must be a non-empty string.")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    manifest_p = Path(manifest_path)
    sha256_hash = compute_file_sha256(manifest_p)
    commit = get_git_commit(repo_path)
    ts = datetime.now(timezone.utc).isoformat()
    return {
        "dataset_name": dataset_name,
        "manifest_sha256": sha256_hash,
        "git_commit": commit,
        "seed": seed,
        "timestamp": ts,
    }


def validate_provenance(record: Mapping[str, Any], context: str = "") -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"Provenance record must be a mapping ({context}).")
    for field in PROVENANCE_REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"Missing required provenance field '{field}' ({context}).")
        val = record[field]
        if val is None or val == "":
            raise ValueError(f"Provenance field '{field}' cannot be empty ({context}).")
    if not isinstance(record["seed"], int):
        raise ValueError(f"Provenance field 'seed' must be an int, got {type(record['seed'])} ({context}).")
    sha = str(record["manifest_sha256"]).strip()
    if len(sha) != 64 or not all(c in "0123456789abcdefABCDEF" for c in sha):
        raise ValueError(f"Provenance field 'manifest_sha256' must be 64-char hex string ({context}).")
    commit = str(record["git_commit"]).strip()
    if len(commit) < 7:
        raise ValueError(f"Provenance field 'git_commit' is invalid: {commit} ({context}).")


def guard_no_smoke_test(item: Any, context: str = "") -> None:
    if isinstance(item, str):
        if "smoke_test" in item:
            raise ValueError(f"Smoke test artifact rejected in scientific evaluation ({context}): '{item}'")
    elif isinstance(item, Path):
        if "smoke_test" in str(item):
            raise ValueError(f"Smoke test artifact rejected in scientific evaluation ({context}): '{item}'")
    elif isinstance(item, Mapping):
        for k, v in item.items():
            guard_no_smoke_test(k, context=f"{context}.key" if context else str(k))
            guard_no_smoke_test(v, context=f"{context}[{k}]" if context else str(k))
    elif isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
        for idx, elem in enumerate(item):
            guard_no_smoke_test(elem, context=f"{context}[{idx}]")


def guard_results_and_provenance(data: Mapping[str, Any], context: str = "") -> None:
    guard_no_smoke_test(data, context=context)
    if "provenance" in data and isinstance(data["provenance"], Mapping):
        validate_provenance(data["provenance"], context=f"{context}.provenance")
    else:
        validate_provenance(data, context=context)


def guard_manifest(manifest_path_or_df: Union[str, Path, Any], context: str = "") -> None:
    if isinstance(manifest_path_or_df, (str, Path)):
        p = Path(manifest_path_or_df)
        guard_no_smoke_test(str(p), context=f"manifest_path: {p}")
        if not p.is_file():
            raise FileNotFoundError(f"Manifest file not found: {p}")
        import pandas as pd
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
    else:
        df = manifest_path_or_df
    for col in ("path", "audio_path", "file_path", "filename"):
        if col in df.columns:
            for val in df[col].dropna():
                guard_no_smoke_test(str(val), context=f"manifest column '{col}'")

