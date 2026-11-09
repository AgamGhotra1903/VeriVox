# Metrics File Index

This index lists all ground-truth results files in `results/metrics/` with their provenance, dataset source, seed, model, key metrics, and timestamps.

| Result File Path                                          | Dataset Source | Seed | Model / System       | Key Metrics Exactly As Stored                             | Timestamp                        |
| --------------------------------------------------------- | -------------- | ---- | -------------------- | --------------------------------------------------------- | -------------------------------- |
| results/metrics/api_load_test.json                        | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-09T07:11:08.688490+00:00 |
| results/metrics/audits.json                               | REAL           | 42   | Multiple / Subsystem | Silence Classifier EER: 18.36%, VAD Trim Delta: 21.55 pts | 2026-11-09T06:50:10.924153+00:00 |
| results/metrics/band_analysis_results.json                | REAL           | 42   | CRNN-LFCC            | Evaluation metrics recorded                               | 2026-11-05T07:44:32.258506+00:00 |
| results/metrics/baselines_results.json                    | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-05T06:49:17.794497+00:00 |
| results/metrics/classical_sweep.json                      | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-05T06:49:17.794497+00:00 |
| results/metrics/codec_robustness_results.json             | REAL           | 42   | Multiple / Subsystem | Clean Baseline: 7.84%, 9 conditions evaluated             | 2026-11-09T06:13:01.575532+00:00 |
| results/metrics/crnn_lfcc_multiseed_summary.json          | REAL           | 42   | CRNN-LFCC            | Dev EER: 1.13%, Eval EER: 7.10%, min-tDCF: 0.2166         | 2026-11-08T10:17:18.449960+00:00 |
| results/metrics/crnn_lfcc_results.json                    | REAL           | 42   | CRNN-LFCC            | Dev EER: 1.15%, Eval EER: 6.70%, min-tDCF: 0.2143         | 2026-11-08T10:17:18.449960+00:00 |
| results/metrics/crnn_lfcc_results_seed42.json             | REAL           | 42   | CRNN-LFCC            | Dev EER: 1.15%, Eval EER: 6.70%, min-tDCF: 0.2143         | 2026-11-08T10:17:18.449960+00:00 |
| results/metrics/crnn_lfcc_results_seed43.json             | REAL           | 43   | CRNN-LFCC            | Dev EER: 1.02%, Eval EER: 7.61%, min-tDCF: 0.2177         | 2026-11-07T12:26:19.505245+00:00 |
| results/metrics/crnn_lfcc_results_seed44.json             | REAL           | 44   | CRNN-LFCC            | Dev EER: 1.22%, Eval EER: 6.98%, min-tDCF: 0.2178         | 2026-11-07T13:06:07.919922+00:00 |
| results/metrics/crnn_logmel_results.json                  | REAL           | 42   | CRNN-LOGMEL          | Dev EER: 2.37%, Eval EER: 10.28%, min-tDCF: 0.2691        | 2026-11-06T13:19:14.286391+00:00 |
| results/metrics/crnn_multiseed_results.json               | REAL           | 42   | CRNN-LFCC            | Dev EER: 1.13%, Eval EER: 7.10%, min-tDCF: 0.2166         | 2026-11-08T10:17:18.449960+00:00 |
| results/metrics/crnn_results.json                         | REAL           | 42   | CRNN-LFCC            | Dev EER: 1.15%, Eval EER: 6.70%, min-tDCF: 0.2143         | 2026-11-08T10:17:18.449960+00:00 |
| results/metrics/crnn_silence_audit.json                   | REAL           | 42   | Multiple / Subsystem | Silence Classifier EER: 18.36%, VAD Trim Delta: 21.55 pts | 2026-11-09T06:50:10.924153+00:00 |
| results/metrics/crnn_training_run_lfcc_seed42.json        | REAL           | 42   | CRNNForensics        | Evaluation metrics recorded                               | 2026-11-09T06:17:55.413647+00:00 |
| results/metrics/crnn_training_run_lfcc_seed43.json        | REAL           | 43   | CRNNForensics        | Evaluation metrics recorded                               | 2026-11-09T06:17:55.498358+00:00 |
| results/metrics/crnn_training_run_lfcc_seed44.json        | REAL           | 44   | CRNNForensics        | Evaluation metrics recorded                               | 2026-11-09T06:17:55.586505+00:00 |
| results/metrics/degradation_chains_results.json           | REAL           | 42   | Multiple / Subsystem | Clean Baseline: 7.84%, 8 conditions evaluated             | 2026-11-09T06:13:01.725049+00:00 |
| results/metrics/ensemble_results.json                     | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-07T13:18:09.301647+00:00 |
| results/metrics/error_audit.json                          | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-08T10:31:15.022184+00:00 |
| results/metrics/gmm_baseline.json                         | REAL           | 42   | LFCC-GMM             | Dev EER: 8.07%, Eval EER: 11.46%, min-tDCF: 0.3572        | 2026-11-05T06:30:47.420628+00:00 |
| results/metrics/in_the_wild_results.json                  | REAL           | 42   | Multiple / Subsystem | CRNN Mean EER: 44.73%, AUC: 0.5638                        | 2026-11-09T06:47:13.505534+00:00 |
| results/metrics/latency.json                              | REAL           | 42   | Multiple / Subsystem | Pipeline MPS Batch 1 p50: 0.0 ms                          | 2026-11-09T06:53:36.635361+00:00 |
| results/metrics/latency_benchmark.json                    | REAL           | 42   | Multiple / Subsystem | Pipeline MPS Batch 1 p50: 0.0 ms                          | 2026-11-05T18:06:01.758118+00:00 |
| results/metrics/logmel_overfit_check.json                 | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-09T06:21:10.037484+00:00 |
| results/metrics/logo_crossval.json                        | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-08T10:31:15.022184+00:00 |
| results/metrics/mp3_delay_check.json                      | REAL           | 42   | Multiple / Subsystem | MP3 16k delay delta: 0.00 pts, 64k delay delta: 0.00 pts  | 2026-11-08T10:31:15.022184+00:00 |
| results/metrics/paper_facts.json                          | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-09T07:13:05.479874+00:00 |
| results/metrics/parity_check.json                         | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-09T06:10:27.700788+00:00 |
| results/metrics/per_attack_evaluation.json                | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-08T10:31:15.022184+00:00 |
| results/metrics/quantization_benchmark.json               | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-05T18:06:43.707632+00:00 |
| results/metrics/speaker_bootstrap.json                    | REAL           | 42   | Multiple / Subsystem | CRNN Eval EER: 7.61%, Stacked Ensemble: 10.07%            | 2026-11-08T10:31:15.022184+00:00 |
| results/metrics/stage5_audits_results.json                | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-08T10:31:15.022184+00:00 |
| results/metrics/syllabus_models_results.json              | REAL           | 42   | Multiple / Subsystem | GaussianNB Eval EER: 33.68%, AdaBoost Eval EER: 22.10%    | 2026-11-08T10:56:21.736493+00:00 |
| results/metrics/syllabus_scratch_logistic_regression.json | REAL           | 42   | Multiple / Subsystem | Newton-Raphson Dev EER: 0.00%                             | 2026-11-08T09:17:10.552547+00:00 |
| results/metrics/syllabus_scratch_pca_kmeans.json          | REAL           | 42   | Multiple / Subsystem | Evaluation metrics recorded                               | 2026-11-08T09:17:46.228706+00:00 |
