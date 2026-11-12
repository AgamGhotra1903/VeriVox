#!/usr/bin/env python3
"""
Packages the submission files following the exact course naming format:
  Report: <GroupNumber>_report_<Title>.pdf
  Code:   <GroupNumber>_code_<Title>.zip

Usage:
  python3 scripts/package_submission.py [GroupNumber] [Title]
  
Example:
  python3 scripts/package_submission.py Group12 Acoustic_Forensics
"""

import sys
import os
import zipfile
import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def build_pdf(group_num: str, title: str, out_dir: Path) -> Path:
    from compile_submission_report import build_html_report, compile_pdf
    
    html_path = out_dir / f"{group_num}_report_{title}.html"
    pdf_path = out_dir / f"{group_num}_report_{title}.pdf"
    
    print(f"--> Building HTML report: {html_path}")
    build_html_report(html_path)
    print(f"--> Compiling PDF: {pdf_path}")
    compile_pdf(html_path, pdf_path)
    return pdf_path

def build_zip(group_num: str, title: str, out_dir: Path) -> Path:
    zip_path = out_dir / f"{group_num}_code_{title}.zip"
    print(f"--> Creating submission ZIP: {zip_path}")
    
    include_dirs = ['src', 'scripts', 'configs', 'tests', 'app', 'frontend', 'docs', 'results', 'checkpoints']
    include_files = ['README.md', 'requirements.txt', 'pyproject.toml', 'Makefile', 'Dockerfile', 'pytest.ini', '.gitignore', 'FIXES_LOG.md', 'FINAL_NOTES.md']

    def should_exclude(rel_path):
        parts = Path(rel_path).parts
        for p in parts:
            if p in ['node_modules', '.venv', '__pycache__', '.pytest_cache', '.git', '.vite', 'dist', 'submission']:
                return True
        if any(pattern in rel_path for pattern in ['data/raw', 'handover', 'review_pack']):
            return True
        if rel_path.endswith(('.pyc', '.DS_Store')):
            return True
        if 'checkpoints/' in rel_path:
            filename = Path(rel_path).name
            if '_epoch' in filename or '_seed' in filename or 'ablation_' in filename:
                return True
        return False

    files_to_zip = []
    for f in include_files:
        p = REPO_ROOT / f
        if p.exists():
            files_to_zip.append(p)

    for d in include_dirs:
        dir_path = REPO_ROOT / d
        if dir_path.exists():
            for root, _, files in os.walk(dir_path):
                for file in files:
                    p = Path(root) / file
                    rel = p.relative_to(REPO_ROOT).as_posix()
                    if not should_exclude(rel):
                        files_to_zip.append(p)

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files_to_zip:
            rel = p.relative_to(REPO_ROOT).as_posix()
            zf.write(p, arcname=f"{title}/{rel}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"--> ZIP created successfully ({len(files_to_zip)} files, {size_mb:.2f} MB)")
    return zip_path

def main():
    group_num = sys.argv[1] if len(sys.argv) > 1 else "GroupNumber"
    title = sys.argv[2] if len(sys.argv) > 2 else "Acoustic_Forensics"
    
    out_dir = REPO_ROOT / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Packaging Submission for Group: {group_num}, Title: {title} ===")
    pdf = build_pdf(group_num, title, out_dir)
    zip_f = build_zip(group_num, title, out_dir)
    
    # Calculate checksums
    h_pdf = hashlib.sha256(pdf.read_bytes()).hexdigest()
    h_zip = hashlib.sha256(zip_f.read_bytes()).hexdigest()
    
    manifest_path = out_dir / "SUBMISSION_MANIFEST.txt"
    manifest_content = f"""SUBMISSION ARTIFACTS MANIFEST
==============================
Group Number: {group_num}
Project Title: {title}
Generated At: {Path(__file__).stat().st_mtime}

1. Report PDF:
   File: {pdf.name}
   Path: {pdf.resolve()}
   Size: {pdf.stat().st_size / 1024:.1f} KB
   SHA-256: {h_pdf}

2. Code & Deliverables ZIP:
   File: {zip_f.name}
   Path: {zip_f.resolve()}
   Size: {zip_f.stat().st_size / (1024 * 1024):.2f} MB
   SHA-256: {h_zip}
"""
    manifest_path.write_text(manifest_content)
    print("\n" + manifest_content)
    print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    main()
