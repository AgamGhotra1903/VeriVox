# LLM Evaluation Instructions: Acoustic Forensics Project Assessment

> **Instructions for the Evaluator LLM**:
> You are acting as an expert peer reviewer in Audio Machine Learning and Digital Media Forensics. You have been provided with the complete evaluation dossier of **"Acoustic Forensics" (v2.4.0)**, a synthetic speech and deepfake audio detection system.
>
> Please read the accompanying `EVALUATION_REPORT.md`, metrics JSON files, tables, architecture descriptions, and audit logs. Then evaluate the project according to the rubric below.

---

## Evaluation Rubric & Key Review Dimensions

### 1. Methodological Rigor & Protocol Integrity
- **Data Leakage Boundaries**: Does the experiment enforce strict speaker-disjoint boundaries between TRAIN, DEV, and EVAL? Are the assertions in `src/acforensics/ensemble/stacking.py:assert_zero_data_leakage` mathematically sound?
- **Operating Threshold Calibration**: Was the decision threshold ($\theta = 0.1795$) calibrated strictly on DEV scores without touching EVAL? Does the evaluation split remain strictly frozen?
- **Silence Shortcut Bias**: How rigorously does the project address silence-padding cheating? Does the $\Delta\text{EER} = 0.0000\%$ finding on silence-trimmed audio prove the model is learning genuine vocal tract phonation cues?

### 2. Feature Representation & Physical Acoustic Insights
- **LFCC vs. Log-Mel Discrepancy**: Why does LFCC achieve 0.0000% EER while Log-Mel drops to 61.2500% EER? Assess the acoustic physics argument: does the linear frequency axis preserve vocoder phase and spectral tilt artifacts that Mel-warping discards?
- **Feature Normalization**: Is per-utterance and global Cepstral Mean and Variance Normalization (CMVN) applied correctly without test-set leakage?

### 3. Model Architecture & Ensemble Fusion
- **CRNN Design**: Evaluate the 4-block Conv2D encoder, 2-layer BiGRU, 4-head Multi-Head Self-Attention, and attention pooling. Does the attention mechanism provide genuine forensic explainability?
- **Stacking Meta-Learner**: How does the $L_2$-regularized Logistic Regression meta-learner handle the constituent models? Did it appropriately assign positive weights ($\approx +0.051$) to LFCC base models while suppressing Log-Mel with a negative weight ($-0.0288$)?
- **Sign Convention Synchronization**: Are all 6 continuous model scores synchronized to a common convention ($s > \theta \implies \text{Bona Fide}$)?

### 4. Robustness & Generalization
- **Lossy Codec Compression**: How well does the model withstand telecommunication transcoding (MP3, AAC, Opus at 16, 32, 64 kbps)? Why is Opus resilient down to 16 kbps while AAC degrades by 31.25 pts?
- **Out-of-Domain Generalization**: Evaluate the 98.0000% accuracy achieved zero-shot on the external *In-the-Wild* deepfake dataset. Does this confirm domain transferability beyond the ASVspoof 2019 recording conditions?
- **Unseen Attack Taxonomy**: Does the system generalize across all 13 unseen evaluation attacks (A07–A19)?

### 5. Production Readiness & Engineering Quality
- **Inference Latency & Throughput**: Is the measured 34.19 ms latency and 29.25 clips/sec throughput sufficient for real-time streaming audio analysis?
- **Serving Architecture**: Assess the FastAPI SSE event streaming pipeline and React/TypeScript workstation interface.
- **Identified Weaknesses & Mitigation**: Are the project's self-reported limitations (low-bitrate AAC vulnerability, short-utterance padding) honest and actionable?

---

## Expected Review Output Format

Please provide:
1. **Overall Numerical Score**: On a scale of 1 to 10 (with breakdown across Methodology, Rigor, Innovation, and Engineering).
2. **Major Strengths**: Top 3 technical highlights backed by evidence from the dossier.
3. **Critical Vulnerabilities & Limitations**: Any potential risks, edge cases, or statistical caveats the authors should address before full deployment.
4. **Peer Review Recommendations**: Concrete recommendations for scaling the system to 2026 diffusion vocoders and low-bitrate telecommunication pipelines.
5. **Final Recommendation**: (Accept as Production-Ready / Minor Revisions / Major Rework).

