#!/usr/bin/env python3
"""Acoustic Forensics - Live Terminal Demonstration & Pipeline Verification.

Runs end-to-end inference across the 6-model stacked ensemble on real audio samples,
displays base model scores, fused meta-learner confidence, and outputs the official benchmark tables.

Author: Agam Ghotra (IIT2024226)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from acforensics.ensemble.predictor import EnsemblePredictor

def main():
    predictor = EnsemblePredictor()

    print("=" * 82)
    print("      🎙️  ACOUSTIC FORENSICS: 6-MODEL STACKED ENSEMBLE LIVE INFERENCE DEMO")
    print("=" * 82)

    samples = [
        ("data/smoke_test/ASVspoof2019_LA_eval/flac/LA_E_1000000.flac", "Authentic Human Speech (Bona Fide)"),
        ("data/smoke_test/ASVspoof2019_LA_eval/flac/LA_E_1000001.flac", "Deepfake Attack A10 (Neural TTS / Vocoder)"),
        ("data/smoke_test/ASVspoof2019_LA_eval/flac/LA_E_1000002.flac", "Deepfake Attack A12 (Neural WaveNet / VC)"),
        ("data/smoke_test/ASVspoof2019_LA_eval/flac/LA_E_1000003.flac", "Deepfake Attack A15 (Waveform Concatenation)"),
    ]

    for p, desc in samples:
        path = Path(p)
        if not path.exists():
            continue
        res = predictor.predict_file(path)
        verdict = res["verdict"]
        prob = res["probability_bonafide"] * 100.0
        fused = res["fused_score"]
        mean_sc = res["baseline_mean_score"]

        print(f"\n[SAMPLE]: {path.name} — {desc}")
        print(f"  ├── Final Screener Verdict : {verdict}")
        print(f"  ├── Calibrated P(Genuine)  : {prob:6.2f}%")
        print(f"  ├── Stacked Meta-Score     : {fused:+.4f} (Higher = More Bona Fide)")
        print(f"  ├── Baseline Mean Z-Score  : {mean_sc:+.4f}")
        print("  └── Base Model Breakdown:")
        for m_name, s_val in res["base_scores"].items():
            bar_len = min(20, max(1, int(abs(s_val) * 2)))
            bar_char = "█" if s_val >= 0 else "░"
            print(f"      • {m_name:<20s}: score = {s_val:+9.4f}  |{bar_char * bar_len}")

    print("\n" + "=" * 82)
    print("      📊 OFFICIAL ASVSPOOF 2019 LA EVALUATION BENCHMARK TABLE")
    print("=" * 82)
    p_eval = Path("results/tables/models_and_ensembles_eval.csv")
    if p_eval.exists():
        df_eval = pd.read_csv(p_eval)
        print(df_eval.to_string(index=False))

    print("\n" + "=" * 82)
    print("      ⚖️  META-LEARNER WEIGHTS (L2 LOGISTIC REGRESSION ON DEV SCORES)")
    print("=" * 82)
    p_w = Path("results/tables/meta_learner_weights.csv")
    if p_w.exists():
        df_w = pd.read_csv(p_w)
        print(df_w.to_string(index=False))

    print("\n" + "=" * 82)
    print("      🧪 CONTROLLED FEATURE & ARCHITECTURAL ABLATIONS")
    print("=" * 82)
    p_feat = Path("results/tables/feature_ablation.csv")
    if p_feat.exists():
        print("Feature Representation Ablation:")
        print(pd.read_csv(p_feat).to_string(index=False))

    p_arch = Path("results/tables/architecture_ablation.csv")
    if p_arch.exists():
        print("\nNeural Architecture Ablation:")
        print(pd.read_csv(p_arch).to_string(index=False))

    print("\n" + "=" * 82)
    print("      🌐 CROSS-DOMAIN ZERO-SHOT EVALUATION (IN-THE-WILD)")
    print("=" * 82)
    p_itw = Path("results/tables/crossdomain_in_the_wild.csv")
    if p_itw.exists():
        print(pd.read_csv(p_itw).to_string(index=False))
    print("=" * 82)

if __name__ == "__main__":
    main()

