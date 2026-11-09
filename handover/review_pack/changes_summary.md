# Changes Summary (Since Base Commit `f02eeda`)

## Git Commit History (`git log --oneline f02eeda..HEAD`)
```text
e6d0cde feat(audit): complete tasks A1-A11 and production-readiness dossier assembly
e003341 docs(phase6): finalize evaluation report with audited numbers and verify consistency
d5c0d1d fix(phase5): reconcile ensemble scoring, ensure speaker-disjoint stacker calibration, and align bootstrap CIs
7be55f7 fix(phase4): retrain CRNN multi-seed models and evaluate silence shortcut remediation
1d6bb6a fix(phase3): retrain CRNN-LogMel with per-utterance CMVN and update ablation metrics
e01252d fix(phase2): redo codec and degradation experiments on stratified 6k eval subset with speaker CIs
18e8c15 fix(phase1): resolve In-The-Wild feature normalization inversion, verify label polarity, and regenerate metrics
```

## File Modification Statistics (`git diff --stat f02eeda..HEAD`)
```text
.gitignore                                         |   2 +
 EVALUATION_REPORT.md                               | 217 ++---
 FINAL_NOTES.md                                     |  89 +++
 FIXES_LOG.md                                       | 326 ++++++++
 checkpoints/calibrator_crnn_lfcc_best.joblib       | Bin 957 -> 957 bytes
 checkpoints/calibrator_crnn_logmel_best.joblib     | Bin 957 -> 957 bytes
 .../ensemble/calibrator_stacked_ensemble.joblib    | Bin 977 -> 977 bytes
 checkpoints/ensemble/classical_bundle.joblib       | Bin 0 -> 30347013 bytes
 checkpoints/ensemble/stacked_ensemble.joblib       | Bin 1009 -> 993 bytes
 docs/FULL_PROJECT_REPORT.md                        | 874 +++++++++++++++++++++
 docs/REPORT.md                                     | 362 +++++++++
 results/metrics/codec_robustness_results.json      | 344 ++++++++
 results/metrics/crnn_logmel_results.json           | 230 ++++++
 results/metrics/degradation_chains_results.json    | 203 +++++
 results/metrics/in_the_wild_results.json           | 183 +++++
 results/tables/ensemble_comparison.md              |   2 +-
 results/tables/ensemble_comparison.tex             |   2 +-
 results/tables/master_baselines_table.md           |   9 +
 results/tables/master_baselines_table.tex          |  13 +
 results/tables/meta_learner_weights.md             |  10 +-
 results/tables/meta_learner_weights.tex            |  10 +-
 results/tables/table11_api_load_test.md            |   8 +-
 results/tables/table11_api_load_test.tex           |   8 +-
 results/tables/table2_main_model.md                |   4 +-
 results/tables/table2_main_model.tex               |   4 +-
 results/tables/table3_codec_robustness.md          |  28 +-
 results/tables/table3_codec_robustness.tex         |  28 +-
 results/tables/table3_robustness.md                |  30 +-
 results/tables/table3_robustness.tex               |  30 +-
 results/tables/table4_crossdomain.md               |  11 +-
 results/tables/table4_crossdomain.tex              |  11 +-
 results/tables/table4_crossdomain_in_the_wild.md   |  11 +-
 results/tables/table4_crossdomain_in_the_wild.tex  |  11 +-
 results/tables/table5_speaker_bootstrap_eval.md    |   8 +-
 results/tables/table5_speaker_bootstrap_eval.tex   |   8 +-
 results/tables/table6_per_attack_eer.md            |  26 +-
 results/tables/table6_per_attack_eer.tex           |  26 +-
 results/tables/table7_silence_audit.md             |   5 +
 results/tables/table7_silence_audit.tex            |   9 +
 results/tables/table8_audits_summary.md            |  16 +-
 results/tables/table8_audits_summary.tex           |   6 +-
 results/tables/table9_acoustic_degradations.md     |  21 +-
 results/tables/table9_acoustic_degradations.tex    |  21 +-
 results/tables/table_api_load_test.md              |   8 +-
 results/tables/table_api_load_test.tex             |   8 +-
 results/tables/table_itw_hand_checked_clips.md     |  22 +
 results/tables/table_latency.md                    |  12 +-
 results/tables/table_latency.tex                   |  12 +-
 results/tables/table_syllabus_models.md            |   6 +
 results/tables/table_syllabus_models.tex           |  10 +
 scripts/benchmark_latency.py                       | 301 ++++---
 scripts/build_master_baseline_table.py             | 120 +++
 scripts/build_paper_facts.py                       | 601 ++++++++++++++
 scripts/build_report_tables.py                     |  27 +-
 scripts/check_mp3_delay.py                         | 169 ++++
 scripts/from_scratch/logistic_regression.py        | 265 +++++++
 scripts/from_scratch/pca_kmeans.py                 | 168 ++++
 scripts/generate_dossier_figures.py                | 171 ++++
 scripts/generate_dossier_report.py                 | 401 ++++++++++
 scripts/generate_metrics_index.py                  | 153 ++++
 scripts/load_test_api.py                           | 186 +++--
 scripts/parity_check.py                            | 208 +++++
 scripts/plot_crnn_multiseed_curves.py              | 111 +++
 scripts/run_itw_audit.py                           | 302 +++++++
 scripts/run_itw_eval.py                            | 277 +++++++
 scripts/run_large_scale_audits.py                  | 385 +++++++++
 scripts/run_robustness_benchmark.py                | 425 ++++++++++
 scripts/run_stage5_eval_and_audits.py              | 198 +++--
 .../run_stage6_robustness_and_generalization.py    | 180 +++--
 scripts/save_crnn_training_histories.py            | 117 +++
 scripts/train_crnn_mps.py                          |  49 +-
 scripts/train_ensemble.py                          |  27 +-
 scripts/train_syllabus_models.py                   | 240 ++++++
 scripts/update_multiseed_records.py                |  94 +++
 scripts/verify_datasets.py                         |  16 +-
 scripts/verify_itw_scores.py                       |  99 +++
 scripts/verify_logmel_overfit.py                   | 126 +++
 scripts/verify_report_numbers.py                   | 189 +++++
 src/acforensics/api/models.py                      |  10 +
 src/acforensics/api/service.py                     | 141 ++--
 src/acforensics/data/audio_io.py                   |  13 +-
 src/acforensics/ensemble/predictor.py              | 207 ++---
 src/acforensics/evaluation/crossdomain.py          |   6 +-
 src/acforensics/evaluation/provenance.py           |  46 +-
 src/acforensics/models/gmm.py                      |   9 +-
 tests/test_api_crnn_parity.py                      |  89 +++
 tests/test_ensemble_fusion.py                      |  35 +
 tests/test_ensemble_parity.py                      |  69 ++
 tests/test_logmel.py                               |  44 ++
 tests/test_metrics.py                              |  20 +
 tests/test_parity_check.py                         |  27 +
 tests/test_protocols.py                            |  41 +
 tests/test_provenance.py                           |  43 +
 tests/test_report_verification.py                  |  22 +
 tests/test_robustness.py                           | 111 +++
 95 files changed, 8992 insertions(+), 830 deletions(-)
```
