# Figures & Plots Index

This index lists all visual diagrams, ROC/DET curves, training trajectories, and audit plots in `figures/` with their description, generation script, exact source results JSON, and status (REAL / SMOKE / STALE / SUPERSEDED).

| Figure File | Description | Generating Script | Source Results JSON | Status |
| :--- | :--- | :--- | :--- | :--- |
| `all_models_det.pdf` | Vector PDF of Detection Error Tradeoff (DET) curves for all 7 models | `scripts/generate_dossier_figures.py` | `results/metrics/crnn_lfcc_results_seed42.json, baselines_results.json, ensemble_results.json` | **REAL** |
| `all_models_det.png` | Detection Error Tradeoff (DET) curves (log-scale FAR vs FRR) for all 7 models | `scripts/generate_dossier_figures.py` | `results/metrics/crnn_lfcc_results_seed42.json, baselines_results.json, ensemble_results.json` | **REAL** |
| `all_models_roc.pdf` | Vector PDF of ROC curves on ASVspoof 2019 Eval (N=71,237) | `scripts/generate_dossier_figures.py` | `results/metrics/crnn_lfcc_results_seed42.json, baselines_results.json, ensemble_results.json` | **REAL** |
| `all_models_roc.png` | ROC curves on ASVspoof 2019 Eval (N=71,237) for CRNN-LFCC, Stacked Ensemble, GMM, RF, SVM, LR | `scripts/generate_dossier_figures.py` | `results/metrics/crnn_lfcc_results_seed42.json, baselines_results.json, ensemble_results.json` | **REAL** |
| `band_analysis_degradation.png` | EER progression as high-frequency bands are low-pass filtered (bandwidth truncation) | `scripts/run_band_analysis.py` | `results/metrics/band_analysis_results.json` | **REAL** |
| `confusion_matrices_eval_and_itw.pdf` | Vector PDF of side-by-side Eval and In-the-Wild confusion matrices | `scripts/generate_dossier_figures.py` | `results/metrics/crnn_lfcc_results_seed42.json, in_the_wild_results.json` | **REAL** |
| `confusion_matrices_eval_and_itw.png` | Normalized confusion matrices on ASVspoof 2019 Eval (N=71,237) and In-the-Wild (N=25,589) | `scripts/generate_dossier_figures.py` | `results/metrics/crnn_lfcc_results_seed42.json, in_the_wild_results.json` | **REAL** |
| `crnn_lfcc_training_curves.png` | CRNN-LFCC epoch progression curves for loss and Dev EER | `scripts/train_crnn_mps.py` | `results/metrics/crnn_training_run_lfcc_seed42.json` | **REAL** |
| `crnn_lfcc_training_curves_seed42.png` | Seed 42 CRNN-LFCC training dynamics | `scripts/train_crnn_mps.py` | `results/metrics/crnn_training_run_lfcc_seed42.json` | **REAL** |
| `crnn_logmel_training_curves.png` | CRNN-LogMel ablation training curves showing convergence with CMVN | `scripts/train_crnn_mps.py` | `results/metrics/crnn_logmel_results.json` | **REAL** |
| `crnn_multiseed_learning_dynamics.png` | Multi-seed (42, 43, 44) training dynamics and Dev EER progression showing seed convergence | `scripts/plot_crnn_multiseed_curves.py` | `results/metrics/crnn_training_run_lfcc_seed42.json, seed43.json, seed44.json` | **REAL** |
| `crnn_training_curves.pdf` | Vector PDF of CRNN-LFCC training curves | `scripts/save_crnn_training_histories.py` | `results/metrics/crnn_training_run_lfcc_seed42.json` | **REAL** |
| `crnn_training_curves.png` | Training/Validation loss and Dev EER trajectory over 25 epochs for CRNN-LFCC | `scripts/save_crnn_training_histories.py` | `results/metrics/crnn_training_run_lfcc_seed42.json` | **REAL** |
| `ensemble_reliability_diagram.png` | Calibration and reliability diagram for uncalibrated vs calibrated Stacked Ensemble | `scripts/train_ensemble.py` | `results/metrics/ensemble_results.json` | **REAL** |
| `ensemble_roc_curves.png` | ROC curves for Stacked Ensemble and constituent models | `scripts/train_ensemble.py` | `results/metrics/ensemble_results.json` | **REAL** |
| `eval_confusion_matrix.pdf` | Vector PDF of ASVspoof 2019 LA Eval confusion matrix | `scripts/generate_dossier_figures.py` | `results/metrics/crnn_lfcc_results_seed42.json` | **REAL** |
| `eval_confusion_matrix.png` | Individual normalized confusion matrix on ASVspoof 2019 LA Eval (N=71,237) | `scripts/generate_dossier_figures.py` | `results/metrics/crnn_lfcc_results_seed42.json` | **REAL** |
| `from_scratch/kmeans_objective_curve.png` | k-Means Lloyd's algorithm objective function (inertia) convergence over iterations | `scripts/from_scratch/pca_kmeans.py` | `results/metrics/syllabus_scratch_pca_kmeans.json` | **REAL** |
| `from_scratch/logistic_regression_loss_curves.png` | Loss curves for Newton-Raphson, SGD, and mini-batch GD optimizers | `scripts/from_scratch/logistic_regression.py` | `results/metrics/syllabus_scratch_logistic_regression.json` | **REAL** |
| `from_scratch/lr_ridge_vs_lasso.png` | Regularization path comparison between Ridge (L2) and Lasso (L1) on acoustic features | `scripts/from_scratch/logistic_regression.py` | `results/metrics/syllabus_scratch_logistic_regression.json` | **REAL** |
| `from_scratch/pca_2d_scatter.png` | 2D scatter plot of acoustic features projected onto first two principal components via SVD | `scripts/from_scratch/pca_kmeans.py` | `results/metrics/syllabus_scratch_pca_kmeans.json` | **REAL** |
| `itw_confusion_matrix.pdf` | Vector PDF of In-the-Wild confusion matrix | `scripts/generate_dossier_figures.py` | `results/metrics/in_the_wild_results.json` | **REAL** |
| `itw_confusion_matrix.png` | Individual normalized confusion matrix on In-the-Wild Speech (N=25,589) | `scripts/generate_dossier_figures.py` | `results/metrics/in_the_wild_results.json` | **REAL** |
| `per_attack_eer.png` | Breakdown of equal error rate across individual attack algorithms A07-A19 | `scripts/run_stage5_eval_and_audits.py` | `results/metrics/per_attack_evaluation.json` | **REAL** |
| `per_attack_eer.svg` | Vector SVG of per-attack EER breakdown across A07-A19 | `scripts/run_stage5_eval_and_audits.py` | `results/metrics/per_attack_evaluation.json` | **REAL** |
| `per_attack_eer_crnn.png` | CRNN-LFCC per-attack EER profile across A07-A19 | `scripts/run_stage5_eval_and_audits.py` | `results/metrics/per_attack_evaluation.json` | **REAL** |
| `per_attack_eer_crnn.svg` | Vector SVG of CRNN-LFCC per-attack EER profile | `scripts/run_stage5_eval_and_audits.py` | `results/metrics/per_attack_evaluation.json` | **REAL** |
| `per_attack_eer_ensemble.png` | Stacked Ensemble per-attack EER profile across A07-A19 | `scripts/run_stage5_eval_and_audits.py` | `results/metrics/per_attack_evaluation.json` | **REAL** |
| `per_attack_eer_ensemble.svg` | Vector SVG of Stacked Ensemble per-attack EER profile | `scripts/run_stage5_eval_and_audits.py` | `results/metrics/per_attack_evaluation.json` | **REAL** |
| `robustness_eer_vs_bitrate.pdf` | Vector PDF of codec robustness curves across bitrates | `scripts/run_robustness_benchmark.py` | `results/metrics/codec_robustness_results.json` | **REAL** |
| `robustness_eer_vs_bitrate.png` | EER degradation curves across bitrates (64, 32, 16 kbps) for MP3, AAC, and Opus on raw audio with delay compensation | `scripts/run_robustness_benchmark.py` | `results/metrics/codec_robustness_results.json` | **REAL** |
| `robustness_eer_vs_bitrate.svg` | Vector SVG of codec robustness curves across bitrates | `scripts/run_robustness_benchmark.py` | `results/metrics/codec_robustness_results.json` | **REAL** |
| `silence_duration_vs_score.pdf` | Vector PDF of silence swap perturbation audit | `scripts/generate_dossier_figures.py` | `results/metrics/crnn_silence_audit.json` | **REAL** |
| `silence_duration_vs_score.png` | Silence swap perturbation audit on N=2,000 pairs showing 34.85% decision flips and shortcut correlation (r=0.7036) | `scripts/generate_dossier_figures.py` | `results/metrics/crnn_silence_audit.json` | **REAL** |
| `syllabus/bias_variance_svm_c.png` | Bias-variance tradeoff analysis for SVM across regularization parameter C | `scripts/train_syllabus_models.py` | `results/metrics/syllabus_models_results.json` | **REAL** |
| `syllabus/bias_variance_tree_depth.png` | Bias-variance tradeoff analysis for Decision Tree / Random Forest across tree depth | `scripts/train_syllabus_models.py` | `results/metrics/syllabus_models_results.json` | **REAL** |
