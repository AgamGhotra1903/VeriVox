import json
from pathlib import Path
import re
import sys
from typing import Any

def get_nested_key(data: dict[str, Any], key_path: str) -> Any:
    if "/" in key_path:
        parts = key_path.split("/")
    else:
        parts = key_path.split(".")
    curr = data
    i = 0
    while i < len(parts):
        p = parts[i]
        if isinstance(curr, dict) and p in curr:
            curr = curr[p]
            i += 1
        elif isinstance(curr, list) and p.isdigit():
            idx = int(p)
            if idx < len(curr):
                curr = curr[idx]
                i += 1
            else:
                raise KeyError(f"Index {idx} out of bounds in list of length {len(curr)} for key path '{key_path}'")
        elif isinstance(curr, dict) and i + 1 < len(parts) and f"{p}.{parts[i+1]}" in curr:
            curr = curr[f"{p}.{parts[i+1]}"]
            i += 2
        else:
            raise KeyError(f"Key segment '{p}' not found in structure for key path '{key_path}'")
    return curr

def parse_and_verify_report(report_path: Path | str, repo_root: Path | str = ".") -> list[dict[str, Any]]:
    report_p = Path(report_path)
    root_p = Path(repo_root)

    if not report_p.exists():
        raise FileNotFoundError(f"Report file not found: {report_p}")

    text = report_p.read_text(encoding="utf-8")
    metrics_cache: dict[str, dict[str, Any]] = {}

    def load_json(rel_path: str) -> dict[str, Any]:
        if rel_path not in metrics_cache:
            p = root_p / rel_path
            if not p.exists():
                raise FileNotFoundError(f"Referenced metrics JSON does not exist: {p}")
            with open(p, "r", encoding="utf-8") as f:
                metrics_cache[rel_path] = json.load(f)
        return metrics_cache[rel_path]

    checks = []

    comment_pattern = re.compile(
        r'<!--\s*verify:\s*file="(?P<file>[^"]+)",\s*key="(?P<key>[^"]+)",\s*value="(?P<val>[^"]+)"(?:,\s*tol="(?P<tol>[^"]+)")?\s*-->'
    )
    for m in comment_pattern.finditer(text):
        checks.append({
            "source_type": "comment",
            "file": m.group("file").strip(),
            "key": m.group("key").strip(),
            "claimed_str": m.group("val").strip(),
            "tol_str": m.group("tol"),
        })

    inline_pattern = re.compile(
        r'\[metric:\s*(?P<file>[^\s\]]+)\s*->\s*(?P<key>[^\s=\]]+)\s*=\s*(?P<val>[0-9\.\-%]+)(?:,\s*tol=(?P<tol>[0-9\.]+))?\]'
    )
    for m in inline_pattern.finditer(text):
        checks.append({
            "source_type": "inline",
            "file": m.group("file").strip(),
            "key": m.group("key").strip(),
            "claimed_str": m.group("val").strip(),
            "tol_str": m.group("tol"),
        })

    table_pattern = re.compile(
        r'\|\s*`?(?P<file>[a-zA-Z0-9_\-\.\/]+\.json)#(?P<key>[a-zA-Z0-9_\-\.]+)`?\s*\|\s*(?P<val>[0-9\.\-%]+)\s*\|'
    )
    for m in table_pattern.finditer(text):
        checks.append({
            "source_type": "table",
            "file": m.group("file").strip(),
            "key": m.group("key").strip(),
            "claimed_str": m.group("val").strip(),
            "tol_str": None,
        })

    citation_pattern = re.compile(
        r'(?P<val>[0-9,\.\-%]+)\s*\[(?P<file>[a-zA-Z0-9_\-\.\/]+\.json):(?P<key>[a-zA-Z0-9_\-\.\/]+)\]'
    )
    for m in citation_pattern.finditer(text):
        checks.append({
            "source_type": "citation",
            "file": m.group("file").strip(),
            "key": m.group("key").strip(),
            "claimed_str": m.group("val").strip(),
            "tol_str": None,
        })

    mismatches = []
    verified_count = 0

    for item in checks:
        rel_file = item["file"]
        key = item["key"]
        claimed_raw = item["claimed_str"]

        claimed_clean = claimed_raw.rstrip("%").replace(",", "").strip()
        try:
            claimed_val = float(claimed_clean)
        except ValueError:
            mismatches.append({
                "file": rel_file,
                "key": key,
                "error": f"Could not parse claimed value '{claimed_raw}' as float.",
            })
            continue

        try:
            data = load_json(rel_file)
            truth_raw = get_nested_key(data, key)
            truth_val = float(truth_raw)
        except Exception as exc:
            mismatches.append({
                "file": rel_file,
                "key": key,
                "error": f"Failed reading ground truth: {exc}",
            })
            continue

        if item["tol_str"] is not None:
            tol = float(item["tol_str"])
        else:
            k_lower = key.lower()
            if "%" in claimed_raw or any(x in k_lower for x in ["eer", "far", "frr", "acc", "f1", "recall", "precision", "std", "rate"]):
                tol = 0.05
            elif any(x in k_lower for x in ["tdcf", "auc"]):
                tol = 0.005
            else:
                tol = 0.01

        diff = abs(claimed_val - truth_val)
        if diff > tol and abs(claimed_val - truth_val * 100.0) <= tol:
            diff = abs(claimed_val - truth_val * 100.0)
            truth_val = truth_val * 100.0

        if diff > tol:
            mismatches.append({
                "file": rel_file,
                "key": key,
                "claimed": claimed_val,
                "truth": truth_val,
                "diff": diff,
                "tolerance": tol,
                "error": f"Discrepancy: claimed {claimed_val} vs truth {truth_val} (|diff|={diff:.4f} > tol {tol})",
            })
        else:
            verified_count += 1

    print(f"Report verification scanned {len(checks)} claims: {verified_count} verified, {len(mismatches)} mismatches.")
    return mismatches

def verify_all_report_numbers(report_path: str = "docs/REPORT.md") -> bool:
    repo_root = Path(__file__).resolve().parent.parent
    p = repo_root / report_path
    if not p.exists():
        fallback = repo_root / "EVALUATION_REPORT.md"
        if fallback.exists():
            p = fallback
        else:
            print(f"Error: Neither {p} nor {fallback} exists.")
            return False

    mismatches = parse_and_verify_report(p, repo_root=repo_root)
    if mismatches:
        print(f"\nFAIL: Found {len(mismatches)} discrepancies in {p.name}:")
        for m in mismatches:
            print(f"  - {m}")
        return False
    print(f"\nSUCCESS: All claimed numbers in {p.name} match ground-truth metrics JSON files within tolerance!")
    return True

if __name__ == "__main__":
    rep = sys.argv[1] if len(sys.argv) > 1 else "docs/REPORT.md"
    ok = verify_all_report_numbers(rep)
    if not ok:
        sys.exit(1)
