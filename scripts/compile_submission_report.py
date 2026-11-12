#!/usr/bin/env python3
"""
Compiles the comprehensive, audited project report into a publication-grade PDF
using headless Google Chrome.
"""
import os
import sys
import base64
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent

def img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = path.suffix.lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/svg+xml" if ext == "svg" else "image/jpeg"
    return f"data:{mime};base64,{data}"

def build_html_report(output_html: Path):
    fig_dir = REPO_ROOT / "results" / "figures"
    
    replacements = {
        "{{CRNN_CURVES}}": img_to_base64(fig_dir / "crnn_training_curves.png"),
        "{{ALL_ROC}}": img_to_base64(fig_dir / "all_models_roc.png"),
        "{{ALL_DET}}": img_to_base64(fig_dir / "all_models_det.png"),
        "{{CONF_MAT}}": img_to_base64(fig_dir / "confusion_matrices_eval_and_itw.png"),
        "{{PER_ATTACK}}": img_to_base64(fig_dir / "per_attack_eer.png"),
        "{{ROBUSTNESS}}": img_to_base64(fig_dir / "robustness_eer_vs_bitrate.png"),
        "{{BAND_ANALYSIS}}": img_to_base64(fig_dir / "band_analysis_degradation.png"),
        "{{SILENCE_AUDIT}}": img_to_base64(fig_dir / "silence_duration_vs_score.png"),
        "{{MULTISEED_DYN}}": img_to_base64(fig_dir / "crnn_multiseed_learning_dynamics.png"),
    }
    
    template_path = REPO_ROOT / "scripts" / "report_template.html"
    content = template_path.read_text(encoding="utf-8")
    for k, v in replacements.items():
        content = content.replace(k, v)
        
    output_html.write_text(content, encoding="utf-8")
    print(f"Generated HTML report at: {output_html} ({len(content)} bytes)")

def compile_pdf(html_path: Path, output_pdf: Path):
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome_bin):
        print(f"Error: Google Chrome not found at {chrome_bin}", file=sys.stderr)
        sys.exit(1)
        
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        chrome_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf}",
        f"file://{html_path.resolve()}"
    ]
    print(f"Compiling PDF via Google Chrome: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Chrome PDF generation failed with code {res.returncode}:\n{res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)
        
    if output_pdf.exists():
        size_kb = output_pdf.stat().st_size / 1024
        print(f"Successfully compiled report PDF: {output_pdf} ({size_kb:.1f} KB)")
    else:
        print("Error: Output PDF was not created.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    html_target = REPO_ROOT / "submission" / "report.html"
    pdf_target = REPO_ROOT / "submission" / "GroupNumber_report_Title.pdf"
    
    if len(sys.argv) > 1:
        pdf_target = Path(sys.argv[1]).resolve()

    build_html_report(html_target)
    compile_pdf(html_target, pdf_target)
