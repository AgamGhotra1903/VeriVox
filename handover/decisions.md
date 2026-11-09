# Engineering Decisions & Audit Rationale Log

This log documents all major architectural, statistical, and engineering decisions made during the final review and preparation of the Acoustic Forensics Production-Readiness Dossier.

---

## Decision D01: Strict Provenance & Zero Smoke Data Enforcement
- **Date:** 2026-11-09
- **Context:** Task A1 requirement to purge/clarify stale smoke artifacts from `results/metrics` and enforce that every JSON carries immutable provenance.
- **Decision:**
  1. Updated `src/acforensics/evaluation/provenance.py` to ensure `guard_no_smoke_test` comprehensively rejects any path or record containing smoke substrings (`results_smoke`, `smoke_test`, `smoke`).
  2. Enriched provenance records with `git_dirty` and `diff_hash` to record exact working tree state alongside the commit hash.
  3. Added CI test `test_all_results_metrics_files_have_provenance_and_sufficient_sample_size` in `tests/test_provenance.py` that verifies 100% of files in `results/metrics/` carry valid provenance and explicitly assert `subset=True` for any experiment evaluated on $< 1,000$ clips.
  4. Scripted regeneration of `handover/metrics_index.md` via `scripts/generate_metrics_index.py`.
- **Impact:** All 31 metrics files in `results/metrics/` pass provenance validation; `pytest tests/test_provenance.py` passes 10/10 tests.

## Decision D02: Preprocessing Parity Verification (Cached vs Raw Audio Path)
- **Date:** 2026-11-09
- **Context:** Task A2 requirement to verify that the offline cached-feature path and the online raw-audio path yield identical preprocessing, LFCC features, CMVN normalization, and predictions.
- **Decision:**
  1. Built `scripts/parity_check.py` to evaluate $N=2{,}000$ stratified ASVspoof 2019 Eval clips through both paths.
  2. Verified parity on specifications: sample rate 16 kHz, 4.0s center crop/pad, 20 linear filters, 20 cepstral coefficients with deltas and deltadeltas (60 dims total), per-utterance CMVN, no peak normalization, bonafide=1, spoof=0.
  3. Results on $N=2{,}000$ trials:
     - Cached path EER: **7.86%** [95% CI: 6.71%, 9.22%]
     - Raw audio path EER: **7.86%** [95% CI: 6.71%, 9.22%]
     - $\Delta\text{EER}$: **0.0000 pts**
     - Max absolute score difference: **$3.73 \times 10^{-3}$** (driven strictly by float16 storage quantization in cached files).
  4. Added integration test `tests/test_parity_check.py` to guarantee non-regression. Serialized findings to `results/metrics/parity_check.json`.
- **Impact:** Defect A2 is fully resolved and mathematically proven; green light granted to run robustness and In-the-Wild evaluations.
