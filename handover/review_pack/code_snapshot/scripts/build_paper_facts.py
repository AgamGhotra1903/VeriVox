import json
from pathlib import Path
from typing import Any

from acforensics.evaluation.provenance import (
    create_provenance_record,
    guard_results_and_provenance,
)


def load_json(p: Path | str) -> dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_paper_facts(repo_root: Path | str = ".") -> dict[str, Any]:
    root = Path(repo_root)
    m_dir = root / "results" / "metrics"

    crnn_multi = load_json(m_dir / "crnn_multiseed_results.json")
    crnn_primary = load_json(m_dir / "crnn_results.json")
    baselines = load_json(m_dir / "baselines_results.json")
    ensemble = load_json(m_dir / "ensemble_results.json")
    codec = load_json(m_dir / "codec_robustness_results.json")
    deg = load_json(m_dir / "degradation_chains_results.json")
    itw = load_json(m_dir / "in_the_wild_results.json")
    audits = load_json(m_dir / "crnn_silence_audit.json")
    parity = load_json(m_dir / "parity_check.json")
    latency = load_json(m_dir / "latency.json")
    spk_boot = load_json(m_dir / "speaker_bootstrap.json")
    logmel_chk = load_json(m_dir / "logmel_overfit_check.json")
    crnn_logmel = load_json(m_dir / "crnn_logmel_results.json")

    facts = {
        "primary_model_crnn_lfcc": {
            "parameters": {
                "val": crnn_primary["parameters"],
                "source": "results/metrics/crnn_results.json",
                "key": "parameters",
            },
            "dev_3seed_mean_eer": {
                "val": crnn_multi["metrics"]["dev_eer_mean"],
                "source": "results/metrics/crnn_multiseed_results.json",
                "key": "metrics.dev_eer_mean",
            },
            "dev_3seed_std_eer": {
                "val": crnn_multi["metrics"]["dev_eer_std"],
                "source": "results/metrics/crnn_multiseed_results.json",
                "key": "metrics.dev_eer_std",
            },
            "eval_3seed_mean_eer": {
                "val": crnn_multi["metrics"]["eval_eer_mean"],
                "source": "results/metrics/crnn_multiseed_results.json",
                "key": "metrics.eval_eer_mean",
                "n": 71237,
            },
            "eval_3seed_std_eer": {
                "val": crnn_multi["metrics"]["eval_eer_std"],
                "source": "results/metrics/crnn_multiseed_results.json",
                "key": "metrics.eval_eer_std",
                "n": 71237,
            },
            "seed_42": {
                "dev_eer": {
                    "val": crnn_multi["individual_runs"][0]["dev"]["eer"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.0.dev.eer",
                },
                "eval_eer": {
                    "val": crnn_multi["individual_runs"][0]["eval"]["eer"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.0.eval.eer",
                    "n": 71237,
                    "ci_95": crnn_multi["individual_runs"][0]["eval"]["eer_ci_95"],
                },
                "eval_roc_auc": {
                    "val": crnn_multi["individual_runs"][0]["eval"]["roc_auc"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.0.eval.roc_auc",
                    "n": 71237,
                },
                "eval_min_tdcf": {
                    "val": crnn_multi["individual_runs"][0]["eval"]["min_tdcf"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.0.eval.min_tdcf",
                    "n": 71237,
                    "ci_95": crnn_multi["individual_runs"][0]["eval"]["min_tdcf_ci_95"],
                },
            },
            "seed_43": {
                "dev_eer": {
                    "val": crnn_multi["individual_runs"][1]["dev"]["eer"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.1.dev.eer",
                },
                "eval_eer": {
                    "val": crnn_multi["individual_runs"][1]["eval"]["eer"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.1.eval.eer",
                    "n": 71237,
                    "ci_95": crnn_multi["individual_runs"][1]["eval"]["eer_ci_95"],
                },
                "eval_roc_auc": {
                    "val": crnn_multi["individual_runs"][1]["eval"]["roc_auc"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.1.eval.roc_auc",
                    "n": 71237,
                },
                "eval_min_tdcf": {
                    "val": crnn_multi["individual_runs"][1]["eval"]["min_tdcf"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.1.eval.min_tdcf",
                    "n": 71237,
                    "ci_95": crnn_multi["individual_runs"][1]["eval"]["min_tdcf_ci_95"],
                },
            },
            "seed_44": {
                "dev_eer": {
                    "val": crnn_multi["individual_runs"][2]["dev"]["eer"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.2.dev.eer",
                },
                "eval_eer": {
                    "val": crnn_multi["individual_runs"][2]["eval"]["eer"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.2.eval.eer",
                    "n": 71237,
                    "ci_95": crnn_multi["individual_runs"][2]["eval"]["eer_ci_95"],
                },
                "eval_roc_auc": {
                    "val": crnn_multi["individual_runs"][2]["eval"]["roc_auc"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.2.eval.roc_auc",
                    "n": 71237,
                },
                "eval_min_tdcf": {
                    "val": crnn_multi["individual_runs"][2]["eval"]["min_tdcf"],
                    "source": "results/metrics/crnn_multiseed_results.json",
                    "key": "individual_runs.2.eval.min_tdcf",
                    "n": 71237,
                    "ci_95": crnn_multi["individual_runs"][2]["eval"]["min_tdcf_ci_95"],
                },
            },
            "speaker_bootstrap": {
                "eer": {
                    "val": spk_boot["Best Single Model (CRNN-LFCC)"]["eer"],
                    "source": "results/metrics/speaker_bootstrap.json",
                    "key": "Best Single Model (CRNN-LFCC).eer",
                    "n": 71237,
                    "ci_95": spk_boot["Best Single Model (CRNN-LFCC)"]["eer_ci_95"],
                },
                "min_tdcf": {
                    "val": spk_boot["Best Single Model (CRNN-LFCC)"]["min_tdcf"],
                    "source": "results/metrics/speaker_bootstrap.json",
                    "key": "Best Single Model (CRNN-LFCC).min_tdcf",
                    "n": 71237,
                    "ci_95": spk_boot["Best Single Model (CRNN-LFCC)"]["min_tdcf_ci_95"],
                },
            },
            "eval_metrics_at_dev_threshold": {
                "accuracy": {
                    "val": crnn_primary["eval"]["metrics_at_dev_threshold"]["accuracy"],
                    "source": "results/metrics/crnn_results.json",
                    "key": "eval.metrics_at_dev_threshold.accuracy",
                    "n": 71237,
                },
                "precision": {
                    "val": crnn_primary["eval"]["metrics_at_dev_threshold"]["precision"],
                    "source": "results/metrics/crnn_results.json",
                    "key": "eval.metrics_at_dev_threshold.precision",
                    "n": 71237,
                },
                "recall": {
                    "val": crnn_primary["eval"]["metrics_at_dev_threshold"]["recall"],
                    "source": "results/metrics/crnn_results.json",
                    "key": "eval.metrics_at_dev_threshold.recall",
                    "n": 71237,
                },
                "f1": {
                    "val": crnn_primary["eval"]["metrics_at_dev_threshold"]["f1"],
                    "source": "results/metrics/crnn_results.json",
                    "key": "eval.metrics_at_dev_threshold.f1",
                    "n": 71237,
                },
                "tp": {
                    "val": crnn_primary["eval"]["metrics_at_dev_threshold"]["tp"],
                    "source": "results/metrics/crnn_results.json",
                    "key": "eval.metrics_at_dev_threshold.tp",
                    "n": 71237,
                },
                "fp": {
                    "val": crnn_primary["eval"]["metrics_at_dev_threshold"]["fp"],
                    "source": "results/metrics/crnn_results.json",
                    "key": "eval.metrics_at_dev_threshold.fp",
                    "n": 71237,
                },
                "tn": {
                    "val": crnn_primary["eval"]["metrics_at_dev_threshold"]["tn"],
                    "source": "results/metrics/crnn_results.json",
                    "key": "eval.metrics_at_dev_threshold.tn",
                    "n": 71237,
                },
                "fn": {
                    "val": crnn_primary["eval"]["metrics_at_dev_threshold"]["fn"],
                    "source": "results/metrics/crnn_results.json",
                    "key": "eval.metrics_at_dev_threshold.fn",
                    "n": 71237,
                },
            },
        },
        "baselines": {
            "gmm_lfcc": {
                "dev_eer": {
                    "val": baselines["GMM"]["lfcc"]["dev"]["eer"],
                    "source": "results/metrics/baselines_results.json",
                    "key": "GMM.lfcc.dev.eer",
                },
                "eval_eer": {
                    "val": baselines["GMM"]["lfcc"]["eval"]["eer"],
                    "source": "results/metrics/baselines_results.json",
                    "key": "GMM.lfcc.eval.eer",
                    "n": 71237,
                    "ci_95": baselines["GMM"]["lfcc"]["eval"]["eer_ci_95"],
                },
                "eval_min_tdcf": {
                    "val": baselines["GMM"]["lfcc"]["eval"]["min_tdcf"],
                    "source": "results/metrics/baselines_results.json",
                    "key": "GMM.lfcc.eval.min_tdcf",
                    "n": 71237,
                    "ci_95": baselines["GMM"]["lfcc"]["eval"]["min_tdcf_ci_95"],
                },
                "eval_roc_auc": {
                    "val": baselines["GMM"]["lfcc"]["eval"]["roc_auc"],
                    "source": "results/metrics/baselines_results.json",
                    "key": "GMM.lfcc.eval.roc_auc",
                    "n": 71237,
                },
            },
            "random_forest_lfcc": {
                "eval_eer": {
                    "val": baselines["RandomForest"]["lfcc"]["eval"]["eer"],
                    "source": "results/metrics/baselines_results.json",
                    "key": "RandomForest.lfcc.eval.eer",
                    "n": 71237,
                },
                "eval_roc_auc": {
                    "val": baselines["RandomForest"]["lfcc"]["eval"]["roc_auc"],
                    "source": "results/metrics/baselines_results.json",
                    "key": "RandomForest.lfcc.eval.roc_auc",
                    "n": 71237,
                },
            },
            "svm_lfcc": {
                "eval_eer": {
                    "val": baselines["SVM"]["lfcc"]["eval"]["eer"],
                    "source": "results/metrics/baselines_results.json",
                    "key": "SVM.lfcc.eval.eer",
                    "n": 71237,
                },
                "eval_roc_auc": {
                    "val": baselines["SVM"]["lfcc"]["eval"]["roc_auc"],
                    "source": "results/metrics/baselines_results.json",
                    "key": "SVM.lfcc.eval.roc_auc",
                    "n": 71237,
                },
            },
            "logistic_regression_lfcc": {
                "eval_eer": {
                    "val": baselines["LogisticRegression"]["lfcc"]["eval"]["eer"],
                    "source": "results/metrics/baselines_results.json",
                    "key": "LogisticRegression.lfcc.eval.eer",
                    "n": 71237,
                },
                "eval_roc_auc": {
                    "val": baselines["LogisticRegression"]["lfcc"]["eval"]["roc_auc"],
                    "source": "results/metrics/baselines_results.json",
                    "key": "LogisticRegression.lfcc.eval.roc_auc",
                    "n": 71237,
                },
            },
            "crnn_logmel": {
                "eval_eer": {
                    "val": crnn_logmel["eval"]["eer"],
                    "source": "results/metrics/crnn_logmel_results.json",
                    "key": "eval.eer",
                    "n": 71237,
                    "ci_95": crnn_logmel["eval"]["eer_ci_95"],
                },
                "overfit_check_loss": {
                    "val": logmel_chk["overfit_experiment"]["final_loss"],
                    "source": "results/metrics/logmel_overfit_check.json",
                    "key": "overfit_experiment.final_loss",
                    "n": 64,
                },
                "overfit_check_accuracy": {
                    "val": logmel_chk["overfit_experiment"]["final_accuracy_pct"],
                    "source": "results/metrics/logmel_overfit_check.json",
                    "key": "overfit_experiment.final_accuracy_pct",
                    "n": 64,
                },
            },
        },
        "stacked_ensemble_ablation": {
            "dev_eer": {
                "val": ensemble["stacked_ensemble"]["dev"]["eer"],
                "source": "results/metrics/ensemble_results.json",
                "key": "stacked_ensemble.dev.eer",
            },
            "eval_eer": {
                "val": ensemble["stacked_ensemble"]["eval"]["eer"],
                "source": "results/metrics/ensemble_results.json",
                "key": "stacked_ensemble.eval.eer",
                "n": 71237,
                "ci_95": ensemble["stacked_ensemble"]["eval"]["eer_ci_95"],
            },
            "eval_min_tdcf": {
                "val": ensemble["stacked_ensemble"]["eval"]["min_tdcf"],
                "source": "results/metrics/ensemble_results.json",
                "key": "stacked_ensemble.eval.min_tdcf",
                "n": 71237,
                "ci_95": ensemble["stacked_ensemble"]["eval"]["min_tdcf_ci_95"],
            },
            "eval_roc_auc": {
                "val": ensemble["stacked_ensemble"]["eval"]["roc_auc"],
                "source": "results/metrics/ensemble_results.json",
                "key": "stacked_ensemble.eval.roc_auc",
                "n": 71237,
            },
            "mean_fusion_eval_eer": {
                "val": ensemble["mean_baseline_fusion"]["eval"]["eer"],
                "source": "results/metrics/ensemble_results.json",
                "key": "mean_baseline_fusion.eval.eer",
                "n": 71237,
            },
        },
        "robustness_fixed_raw_audio": {
            "clean_baseline_eer": {
                "val": codec["clean_baseline"]["crnn_lfcc_eer"],
                "source": "results/metrics/codec_robustness_results.json",
                "key": "clean_baseline.crnn_lfcc_eer",
                "n": 5000,
                "ci_95": codec["clean_baseline"]["ci_95"],
            },
            "mp3_16k_eer": {
                "val": codec["conditions"]["mp3_16k"]["crnn"]["eer"],
                "source": "results/metrics/codec_robustness_results.json",
                "key": "conditions.mp3_16k.crnn.eer",
                "n": 5000,
            },
            "mp3_32k_eer": {
                "val": codec["conditions"]["mp3_32k"]["crnn"]["eer"],
                "source": "results/metrics/codec_robustness_results.json",
                "key": "conditions.mp3_32k.crnn.eer",
                "n": 5000,
            },
            "mp3_64k_eer": {
                "val": codec["conditions"]["mp3_64k"]["crnn"]["eer"],
                "source": "results/metrics/codec_robustness_results.json",
                "key": "conditions.mp3_64k.crnn.eer",
                "n": 5000,
            },
            "aac_16k_eer": {
                "val": codec["conditions"]["aac_16k"]["crnn"]["eer"],
                "source": "results/metrics/codec_robustness_results.json",
                "key": "conditions.aac_16k.crnn.eer",
                "n": 5000,
            },
            "aac_32k_eer": {
                "val": codec["conditions"]["aac_32k"]["crnn"]["eer"],
                "source": "results/metrics/codec_robustness_results.json",
                "key": "conditions.aac_32k.crnn.eer",
                "n": 5000,
            },
            "aac_64k_eer": {
                "val": codec["conditions"]["aac_64k"]["crnn"]["eer"],
                "source": "results/metrics/codec_robustness_results.json",
                "key": "conditions.aac_64k.crnn.eer",
                "n": 5000,
            },
            "opus_16k_eer": {
                "val": codec["conditions"]["opus_16k"]["crnn"]["eer"],
                "source": "results/metrics/codec_robustness_results.json",
                "key": "conditions.opus_16k.crnn.eer",
                "n": 5000,
            },
            "opus_32k_eer": {
                "val": codec["conditions"]["opus_32k"]["crnn"]["eer"],
                "source": "results/metrics/codec_robustness_results.json",
                "key": "conditions.opus_32k.crnn.eer",
                "n": 5000,
            },
            "opus_64k_eer": {
                "val": codec["conditions"]["opus_64k"]["crnn"]["eer"],
                "source": "results/metrics/codec_robustness_results.json",
                "key": "conditions.opus_64k.crnn.eer",
                "n": 5000,
            },
            "telephony_a_law_eer": {
                "val": deg["conditions"]["telephony_8k_alaw"]["crnn"]["eer"],
                "source": "results/metrics/degradation_chains_results.json",
                "key": "conditions.telephony_8k_alaw.crnn.eer",
                "n": 5000,
            },
            "telephony_mu_law_eer": {
                "val": deg["conditions"]["telephony_8k_mulaw"]["crnn"]["eer"],
                "source": "results/metrics/degradation_chains_results.json",
                "key": "conditions.telephony_8k_mulaw.crnn.eer",
                "n": 5000,
            },
            "noise_0db_eer": {
                "val": deg["conditions"]["noise_snr_0db"]["crnn"]["eer"],
                "source": "results/metrics/degradation_chains_results.json",
                "key": "conditions.noise_snr_0db.crnn.eer",
                "n": 5000,
            },
            "noise_10db_eer": {
                "val": deg["conditions"]["noise_snr_10db"]["crnn"]["eer"],
                "source": "results/metrics/degradation_chains_results.json",
                "key": "conditions.noise_snr_10db.crnn.eer",
                "n": 5000,
            },
            "noise_20db_eer": {
                "val": deg["conditions"]["noise_snr_20db"]["crnn"]["eer"],
                "source": "results/metrics/degradation_chains_results.json",
                "key": "conditions.noise_snr_20db.crnn.eer",
                "n": 5000,
            },
            "reverb_rt60_03s_eer": {
                "val": deg["conditions"]["reverb_rt60_0.3s"]["crnn"]["eer"],
                "source": "results/metrics/degradation_chains_results.json",
                "key": "conditions.reverb_rt60_0.3s.crnn.eer",
                "n": 5000,
            },
        },
        "in_the_wild_cross_domain": {
            "total_clips": {
                "val": itw["sample_size"],
                "source": "results/metrics/in_the_wild_results.json",
                "key": "sample_size",
                "n": 25589,
            },
            "bonafide_count": {
                "val": itw["bonafide_count"],
                "source": "results/metrics/in_the_wild_results.json",
                "key": "bonafide_count",
                "n": 25589,
            },
            "spoof_count": {
                "val": itw["spoof_count"],
                "source": "results/metrics/in_the_wild_results.json",
                "key": "spoof_count",
                "n": 25589,
            },
            "crnn_mean_optimal_eer": {
                "val": itw["aggregations"]["crnn_mean"]["optimal_threshold_metrics"]["itw_optimal_eer"],
                "source": "results/metrics/in_the_wild_results.json",
                "key": "aggregations.crnn_mean.optimal_threshold_metrics.itw_optimal_eer",
                "n": 25589,
                "ci_95": itw["aggregations"]["crnn_mean"]["optimal_threshold_metrics"]["itw_optimal_eer_ci_95"],
            },
            "crnn_mean_roc_auc": {
                "val": itw["aggregations"]["crnn_mean"]["optimal_threshold_metrics"]["roc_auc"],
                "source": "results/metrics/in_the_wild_results.json",
                "key": "aggregations.crnn_mean.optimal_threshold_metrics.roc_auc",
                "n": 25589,
            },
            "crnn_mean_accuracy_dev_th": {
                "val": itw["aggregations"]["crnn_mean"]["dev_threshold_metrics"]["accuracy"],
                "source": "results/metrics/in_the_wild_results.json",
                "key": "aggregations.crnn_mean.dev_threshold_metrics.accuracy",
                "n": 25589,
            },
            "polarity_inverted_diagnostic_auc": {
                "val": itw["aggregations"]["crnn_mean"]["polarity_inverted_diagnostic"]["inverted_roc_auc"],
                "source": "results/metrics/in_the_wild_results.json",
                "key": "aggregations.crnn_mean.polarity_inverted_diagnostic.inverted_roc_auc",
                "n": 25589,
            },
        },
        "forensic_audits": {
            "silence_duration_classifier_eer": {
                "val": audits["silence_duration_classifier"]["eer"],
                "source": "results/metrics/crnn_silence_audit.json",
                "key": "silence_duration_classifier.eer",
                "n": 20000,
                "ci_95": audits["silence_duration_classifier"]["ci_95"],
            },
            "silence_duration_classifier_auc": {
                "val": audits["silence_duration_classifier"]["roc_auc"],
                "source": "results/metrics/crnn_silence_audit.json",
                "key": "silence_duration_classifier.roc_auc",
                "n": 20000,
            },
            "vad_untrimmed_eer": {
                "val": audits["vad_trimming"]["untrimmed_eer"],
                "source": "results/metrics/crnn_silence_audit.json",
                "key": "vad_trimming.untrimmed_eer",
                "n": 20000,
            },
            "vad_trimmed_eer": {
                "val": audits["vad_trimming"]["trimmed_eer"],
                "source": "results/metrics/crnn_silence_audit.json",
                "key": "vad_trimming.trimmed_eer",
                "n": 20000,
                "ci_95": audits["vad_trimming"]["ci_95"],
            },
            "vad_delta_eer": {
                "val": audits["vad_trimming"]["delta_eer"],
                "source": "results/metrics/crnn_silence_audit.json",
                "key": "vad_trimming.delta_eer",
                "n": 20000,
            },
            "silence_swap_pearson_r": {
                "val": audits["silence_swap"]["pearson_r"],
                "source": "results/metrics/crnn_silence_audit.json",
                "key": "silence_swap.pearson_r",
                "n": 4000,
            },
            "silence_swap_flip_rate": {
                "val": audits["silence_swap"]["flip_rate_pct"],
                "source": "results/metrics/crnn_silence_audit.json",
                "key": "silence_swap.flip_rate_pct",
                "n": 4000,
            },
        },
        "preprocessing_parity": {
            "cached_eer": {
                "val": parity["cached_path_eer"],
                "source": "results/metrics/parity_check.json",
                "key": "cached_path_eer",
                "n": 2000,
                "ci_95": parity["cached_path_eer_ci_95"],
            },
            "raw_eer": {
                "val": parity["raw_audio_path_eer"],
                "source": "results/metrics/parity_check.json",
                "key": "raw_audio_path_eer",
                "n": 2000,
                "ci_95": parity["raw_audio_path_eer_ci_95"],
            },
            "delta_eer": {
                "val": parity["eer_difference_pts"],
                "source": "results/metrics/parity_check.json",
                "key": "eer_difference_pts",
                "n": 2000,
            },
            "max_abs_score_diff": {
                "val": parity["max_absolute_score_difference"],
                "source": "results/metrics/parity_check.json",
                "key": "max_absolute_score_difference",
                "n": 2000,
            },
        },
        "latency_and_throughput": {
            "crnn_cpu_1thread_ms": {
                "val": latency["benchmarks"]["crnn_cpu_single_thread"]["latency_ms"]["mean"],
                "source": "results/metrics/latency.json",
                "key": "benchmarks.crnn_cpu_single_thread.latency_ms.mean",
            },
            "crnn_cpu_4threads_ms": {
                "val": latency["benchmarks"]["crnn_cpu_multi_thread"]["latency_ms"]["mean"],
                "source": "results/metrics/latency.json",
                "key": "benchmarks.crnn_cpu_multi_thread.latency_ms.mean",
            },
            "crnn_mps_ms": {
                "val": latency["benchmarks"]["crnn_apple_silicon_mps"]["latency_ms"]["mean"],
                "source": "results/metrics/latency.json",
                "key": "benchmarks.crnn_apple_silicon_mps.latency_ms.mean",
            },
            "ensemble_cpu_4threads_ms": {
                "val": latency["benchmarks"]["ensemble_cpu_multi_thread"]["latency_ms"]["mean"],
                "source": "results/metrics/latency.json",
                "key": "benchmarks.ensemble_cpu_multi_thread.latency_ms.mean",
            },
            "ensemble_mps_ms": {
                "val": latency["benchmarks"]["ensemble_apple_silicon_mps"]["latency_ms"]["mean"],
                "source": "results/metrics/latency.json",
                "key": "benchmarks.ensemble_apple_silicon_mps.latency_ms.mean",
            },
        },
    }

    prov = create_provenance_record("frozen_paper_facts", "data/manifests/eval.parquet", seed=42)
    payload = {
        **prov,
        "n_facts": len(facts),
        "facts": facts,
    }

    guard_results_and_provenance(payload)

    out_file = root / "results" / "metrics" / "paper_facts.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Paper facts successfully compiled and verified at {out_file}!")
    return payload


if __name__ == "__main__":
    build_paper_facts()