import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.models.crnn import CRNNForensics
from acforensics.ensemble.stacking import NonNegativeStackedEnsemble
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_metrics_at_threshold,
    _load_cached_asv_scores,
)
from acforensics.evaluation.official_tdcf.eval_metrics import compute_det_curve
from acforensics.evaluation.error_analysis import run_error_audit, plot_per_attack_eer
from acforensics.evaluation.provenance import (
    compute_file_sha256,
    get_git_commit,
    guard_results_and_provenance,
    guard_no_smoke_test,
)

def score_crnn_tensor(model: nn.Module, feats: torch.Tensor, device: torch.device, batch_size: int = 128) -> np.ndarray:
    loader = DataLoader(TensorDataset(feats), batch_size=batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        for (b_x,) in loader:
            b_x = per_utterance_cmvn(b_x.to(device=device, dtype=torch.float32)).unsqueeze(1)
            logits, _ = model(b_x)
            s = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
            scores.append(s)
    return np.concatenate(scores, axis=0)

def compute_asv_constants(asv_file: str) -> tuple[float, float, float, float]:
    asv_tar, asv_non, asv_sp = _load_cached_asv_scores(asv_file)
    asv_frr, asv_far, asv_thresholds = compute_det_curve(asv_tar, asv_non)
    eer_idx = np.argmin(np.abs(asv_frr - asv_far))
    asv_th = asv_thresholds[eer_idx]
    p_miss = np.mean(asv_tar < asv_th)
    p_fa = np.mean(asv_non >= asv_th)
    p_fa_sp = np.mean(asv_sp >= asv_th)
    c0 = 0.9405 * 1.0 * p_miss + 0.0095 * 10.0 * p_fa
    c1 = 0.9405 * 1.0 - c0
    c2 = 0.05 * 10.0 * p_fa_sp
    tdcf_def = c0 + min(c1, c2)
    return float(c0), float(c1), float(c2), float(tdcf_def)

def fast_min_tdcf(bf_s: np.ndarray, sp_s: np.ndarray, c0: float, c1: float, c2: float, tdcf_def: float) -> float:
    if len(bf_s) == 0 or len(sp_s) == 0:
        return 1.0
    cm_frr, cm_far, _ = compute_det_curve(bf_s, sp_s)
    t_dcf = c0 + c1 * cm_frr + c2 * cm_far
    return float(np.min(t_dcf / (tdcf_def + 1e-12)))

def run_speaker_level_bootstrap(
    speakers: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    dev_threshold: float,
    asv_constants: tuple[float, float, float, float],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    c0, c1, c2, tdcf_def = asv_constants
    unique_spks = np.unique(speakers)
    n_spk = len(unique_spks)
    rng = np.random.RandomState(seed)
    spk_to_idx = {s: np.where(speakers == s)[0] for s in unique_spks}

    opt_eer, _ = compute_eer(labels, scores)
    opt_auc = compute_roc_auc(labels, scores)
    bf_s = scores[labels == BONAFIDE]
    sp_s = scores[labels == SPOOF]
    opt_tdcf = fast_min_tdcf(bf_s, sp_s, c0, c1, c2, tdcf_def)
    opt_acc = float(accuracy_score(labels, (scores >= dev_threshold).astype(int)) * 100.0)

    eers, tdcfs, aucs, accs = [], [], [], []
    for _ in range(n_bootstraps):
        sampled_spks = rng.choice(unique_spks, size=n_spk, replace=True)
        idx = np.concatenate([spk_to_idx[s] for s in sampled_spks])
        b_l = labels[idx]
        b_s = scores[idx]
        if len(np.unique(b_l)) < 2:
            continue
        b_eer, _ = compute_eer(b_l, b_s)
        b_auc = compute_roc_auc(b_l, b_s)
        b_acc = float(np.mean((b_s >= dev_threshold) == (b_l == 1)) * 100.0)
        b_bf = b_s[b_l == BONAFIDE]
        b_sp = b_s[b_l == SPOOF]
        b_tdcf = fast_min_tdcf(b_bf, b_sp, c0, c1, c2, tdcf_def)

        eers.append(b_eer)
        tdcfs.append(b_tdcf)
        aucs.append(b_auc)
        accs.append(b_acc)

    return {
        "eer": float(opt_eer),
        "eer_ci_95": [float(np.percentile(eers, 2.5)), float(np.percentile(eers, 97.5))],
        "eer_std": float(np.std(eers)),
        "min_tdcf": float(opt_tdcf),
        "min_tdcf_ci_95": [float(np.percentile(tdcfs, 2.5)), float(np.percentile(tdcfs, 97.5))],
        "min_tdcf_std": float(np.std(tdcfs)),
        "roc_auc": float(opt_auc),
        "roc_auc_ci_95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "roc_auc_std": float(np.std(aucs)),
        "accuracy_dev_th": float(opt_acc),
        "accuracy_dev_th_ci_95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "accuracy_dev_th_std": float(np.std(accs)),
        "applied_dev_threshold": float(dev_threshold),
        "n_speakers": int(n_spk),
        "n_bootstraps": int(n_bootstraps),
    }

def get_vad_boundaries(sig: np.ndarray, hop: int = 160, frame_len: int = 320) -> tuple[int, int, float, float, float]:
    n = len(sig) // hop
    if n < 2:
        return 0, len(sig), 0.0, 0.0, 0.0
    energies = np.array([np.mean(sig[i*hop : i*hop + frame_len] ** 2) for i in range(n - 1)])
    th = np.max(energies) * 0.01 if len(energies) > 0 and np.max(energies) > 0 else 1e-6
    active = np.where(energies > th)[0]
    if len(active) == 0:
        return 0, len(sig), 0.0, 0.0, 0.0
    start_samp = int(active[0] * hop)
    end_samp = int(min(len(sig), (active[-1] + 1) * hop))
    sr = 16000
    lead_sil = start_samp / sr
    trail_sil = (len(sig) - end_samp) / sr
    total_sil = lead_sil + trail_sil + float(np.sum(energies <= th) * (hop / sr))
    return start_samp, end_samp, lead_sil, trail_sil, total_sil

def compute_clip_acoustic_meta(sig: np.ndarray, sr: int = 16000) -> dict[str, float]:
    dur = float(len(sig) / sr)
    rms = float(np.sqrt(np.mean(sig ** 2) + 1e-12))
    rms_db = float(20.0 * np.log10(rms + 1e-12))
    fft_mag = np.abs(np.fft.rfft(sig[: min(len(sig), 64000)]))
    freqs = np.fft.rfftfreq(min(len(sig), 64000), d=1.0/sr)
    denom = np.sum(fft_mag) + 1e-12
    centroid = float(np.sum(freqs * fft_mag) / denom)
    cum_energy = np.cumsum(fft_mag ** 2)
    tot_energy = cum_energy[-1] + 1e-12
    idx_95 = np.searchsorted(cum_energy, 0.95 * tot_energy)
    rolloff_95 = float(freqs[min(idx_95, len(freqs) - 1)])
    return {
        "duration_sec": dur,
        "rms_dbfs": rms_db,
        "spectral_centroid_hz": centroid,
        "rolloff_95_hz": rolloff_95,
    }

def run_stage5_all(seed: int = 42, n_bootstraps: int = 1000):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    eval_parquet = "data/manifests/eval.parquet"
    train_parquet = "data/manifests/train.parquet"
    dev_parquet = "data/manifests/dev.parquet"
    eval_asv = "data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    dev_asv = "data/raw/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
    crnn_ckpt = "checkpoints/crnn_lfcc_best.pt"
    ens_ckpt = "checkpoints/ensemble/stacked_ensemble.joblib"

    for p in [eval_parquet, train_parquet, dev_parquet, eval_asv, dev_asv, crnn_ckpt, ens_ckpt]:
        guard_no_smoke_test(p)

    ev_df = pd.read_parquet(eval_parquet)
    tr_df = pd.read_parquet(train_parquet)
    dv_df = pd.read_parquet(dev_parquet)

    print("Loading precomputed LFCC features...")
    d_ev = np.load("data/cache/lfcc_eval.npz")
    ev_feats = torch.from_numpy(d_ev["features"])
    ev_pooled = d_ev["pooled"]
    ev_y = d_ev["labels"]
    ev_speakers = ev_df["speaker"].values
    ev_attacks = ev_df["attack_id"].values

    d_dv = np.load("data/cache/lfcc_dev.npz")
    dv_feats = torch.from_numpy(d_dv["features"])
    dv_pooled = d_dv["pooled"]
    dv_y = d_dv["labels"]

    print("Loading CRNN model...")
    ckpt = torch.load(crnn_ckpt, map_location=device, weights_only=False)
    crnn = CRNNForensics(in_channels=1, in_freq_dim=60)
    crnn.load_state_dict(ckpt["model_state_dict"])
    crnn.to(device)
    crnn.eval()
    dev_crnn_th = float(ckpt.get("dev_threshold", -0.1428))

    print("Scoring CRNN on Dev and Eval...")
    crnn_dev_scores = score_crnn_tensor(crnn, dv_feats, device)
    crnn_eval_scores = score_crnn_tensor(crnn, ev_feats, device)

    print("Loading Stacked Ensemble...")
    ensemble: NonNegativeStackedEnsemble = NonNegativeStackedEnsemble.load(ens_ckpt)
    with open("results/metrics/ensemble_results.json") as f:
        ens_meta = json.load(f)
    dev_stack_th = float(ens_meta["stacked_ensemble"]["dev"]["threshold"])
    dev_mean_th = float(ens_meta["mean_baseline_fusion"]["dev"]["threshold"])

    print("Loading Classical Base Models...")
    d_tr = np.load("data/cache/lfcc_train.npz")
    fit_idx = np.random.RandomState(seed).choice(len(d_tr["labels"]), int(len(d_tr["labels"]) * 0.8), replace=False)
    fit_pooled = d_tr["pooled"][fit_idx]
    fit_labels = d_tr["labels"][fit_idx]

    scaler_p = StandardScaler()
    fit_p_scaled = scaler_p.fit_transform(fit_pooled)
    ev_p_scaled = scaler_p.transform(ev_pooled)
    dv_p_scaled = scaler_p.transform(dv_pooled)

    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    pca = PCA(n_components=0.95, random_state=seed)
    fit_pca = pca.fit_transform(fit_p_scaled)
    ev_pca = pca.transform(ev_p_scaled)
    dv_pca = pca.transform(dv_p_scaled)

    lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=seed)
    lr.fit(fit_pca, fit_labels)
    lr_eval_scores = lr.decision_function(ev_pca)

    svm = SVC(C=1.0, kernel="rbf", probability=False, class_weight="balanced", random_state=seed)
    svm.fit(fit_pca, fit_labels)
    svm_eval_scores = svm.decision_function(ev_pca)

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    rf.fit(fit_p_scaled, fit_labels)
    rf_eval_scores = rf.predict_proba(ev_p_scaled)[:, BONAFIDE]

    fit_bf_frames = np.concatenate([f.T for f, l in zip(d_tr["features"][fit_idx], fit_labels) if l == BONAFIDE], axis=0)
    fit_sp_frames = np.concatenate([f.T for f, l in zip(d_tr["features"][fit_idx], fit_labels) if l == SPOOF], axis=0)
    sub_bf = fit_bf_frames[np.random.RandomState(seed).choice(len(fit_bf_frames), min(200000, len(fit_bf_frames)), replace=False)]
    sub_sp = fit_sp_frames[np.random.RandomState(seed + 1).choice(len(fit_sp_frames), min(200000, len(fit_sp_frames)), replace=False)]

    gmm_bf = GaussianMixture(n_components=64, covariance_type="diag", reg_covar=1e-3, max_iter=100, random_state=seed)
    gmm_sp = GaussianMixture(n_components=64, covariance_type="diag", reg_covar=1e-3, max_iter=100, random_state=seed + 1)
    gmm_bf.fit(sub_bf)
    gmm_sp.fit(sub_sp)

    def score_gmm_fast(feats_list):
        s = np.zeros(len(feats_list), dtype=np.float64)
        for i, f in enumerate(feats_list):
            t_f = f.T
            s[i] = np.mean(gmm_bf.score_samples(t_f)) - np.mean(gmm_sp.score_samples(t_f))
        return s

    gmm_eval_scores = score_gmm_fast(list(d_ev["features"]))

    eval_score_matrix = np.column_stack([crnn_eval_scores, gmm_eval_scores, rf_eval_scores, svm_eval_scores, lr_eval_scores])
    ensemble_eval_scores = ensemble.predict_score(eval_score_matrix)
    mean_fusion_eval_scores = ensemble.predict_baseline_mean(eval_score_matrix)

    print("Computing Official ASV Operating Constants for Eval min-tDCF...")
    asv_consts = compute_asv_constants(eval_asv)

    print("Running Speaker-Level Bootstrap Confidence Intervals (1000 resamples)...")
    boot_systems = {
        "Stacked Ensemble": (ensemble_eval_scores, dev_stack_th),
        "Best Single Model (CRNN-LFCC)": (crnn_eval_scores, dev_crnn_th),
        "Baseline Mean Fusion": (mean_fusion_eval_scores, dev_mean_th),
        "LFCC-GMM Baseline": (gmm_eval_scores, 0.0),
    }

    speaker_bootstrap_results = {}
    speaker_table_rows = []
    for s_name, (scores_arr, th_val) in boot_systems.items():
        res = run_speaker_level_bootstrap(
            speakers=ev_speakers,
            labels=ev_y,
            scores=scores_arr,
            dev_threshold=th_val,
            asv_constants=asv_consts,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
        speaker_bootstrap_results[s_name] = res
        speaker_table_rows.append({
            "System": s_name,
            "Eval EER (%) [95% CI]": f"{res['eer']:.2f} [{res['eer_ci_95'][0]:.2f}, {res['eer_ci_95'][1]:.2f}]",
            "Eval min-tDCF [95% CI]": f"{res['min_tdcf']:.4f} [{res['min_tdcf_ci_95'][0]:.4f}, {res['min_tdcf_ci_95'][1]:.4f}]",
            "Eval ROC-AUC [95% CI]": f"{res['roc_auc']:.4f} [{res['roc_auc_ci_95'][0]:.4f}, {res['roc_auc_ci_95'][1]:.4f}]",
            "Accuracy at Dev Thresh (%) [95% CI]": f"{res['accuracy_dev_th']:.2f} [{res['accuracy_dev_th_ci_95'][0]:.2f}, {res['accuracy_dev_th_ci_95'][1]:.2f}]",
            "Applied Dev Threshold": f"{th_val:+.4f}",
        })

    df_spk = pd.DataFrame(speaker_table_rows)
    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures")
    metrics_dir = Path("results/metrics")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(tables_dir / "table5_speaker_bootstrap_eval.md", "w") as f:
        f.write(df_spk.to_markdown(index=False) + "\n")
    with open(tables_dir / "table5_speaker_bootstrap_eval.tex", "w") as f:
        f.write(df_spk.to_latex(index=False) + "\n")

    print("Running Per-Attack EER Analysis on real A07-A19...")
    attack_taxonomy = {
        "A07": ("Neural TTS (LSTM-RNN + WORLD + GAN)", False),
        "A08": ("Neural TTS (AR LSTM-RNN + neural source-filter)", False),
        "A09": ("Neural TTS (LSTM-RNN + Vocaine)", False),
        "A10": ("Neural TTS (seq2seq + WaveRNN)", False),
        "A11": ("Neural TTS (seq2seq + Griffin-Lim)", False),
        "A12": ("Neural TTS (WaveNet vocoder)", False),
        "A13": ("TTS-VC (Moment matching NN + waveform filtering)", False),
        "A14": ("TTS-VC (LSTM-RNN + STRAIGHT)", False),
        "A15": ("TTS-VC (LSTM-RNN + WaveNet)", False),
        "A16": ("Waveform Concatenation TTS (A04 variant)", True),
        "A17": ("Neural VC (VAE + waveform filtering)", False),
        "A18": ("VC (i-vector / PLDA + MFCC-to-waveform)", False),
        "A19": ("VC with spectral filtering (A06 variant)", True),
    }

    bf_mask = ev_df["label"].values == BONAFIDE
    bf_crnn_scores = crnn_eval_scores[bf_mask]
    bf_ens_scores = ensemble_eval_scores[bf_mask]

    per_attack_results = {}
    attack_eers_crnn = {}
    attack_eers_ensemble = {}
    per_attack_rows = []

    for atk_id in sorted([a for a in np.unique(ev_attacks) if a != "-"]):
        desc, is_variant = attack_taxonomy.get(atk_id, ("Unseen Generator", False))
        atk_mask = ev_attacks == atk_id
        n_sp = int(np.sum(atk_mask))

        s_crnn_sp = crnn_eval_scores[atk_mask]
        comb_y = np.concatenate([np.ones_like(bf_crnn_scores), np.zeros_like(s_crnn_sp)])
        comb_s_crnn = np.concatenate([bf_crnn_scores, s_crnn_sp])
        eer_crnn, _ = compute_eer(comb_y, comb_s_crnn)
        auc_crnn = compute_roc_auc(comb_y, comb_s_crnn)
        tdcf_crnn = fast_min_tdcf(bf_crnn_scores, s_crnn_sp, asv_consts[0], asv_consts[1], asv_consts[2], asv_consts[3])
        acc_crnn = float(accuracy_score(comb_y, (comb_s_crnn >= dev_crnn_th).astype(int)) * 100.0)

        s_ens_sp = ensemble_eval_scores[atk_mask]
        comb_s_ens = np.concatenate([bf_ens_scores, s_ens_sp])
        eer_ens, _ = compute_eer(comb_y, comb_s_ens)
        auc_ens = compute_roc_auc(comb_y, comb_s_ens)
        tdcf_ens = fast_min_tdcf(bf_ens_scores, s_ens_sp, asv_consts[0], asv_consts[1], asv_consts[2], asv_consts[3])
        acc_ens = float(accuracy_score(comb_y, (comb_s_ens >= dev_stack_th).astype(int)) * 100.0)

        attack_eers_crnn[atk_id] = float(eer_crnn)
        attack_eers_ensemble[atk_id] = float(eer_ens)

        per_attack_results[atk_id] = {
            "description": desc,
            "is_known_variant": bool(is_variant),
            "spoof_trials": n_sp,
            "crnn_lfcc": {
                "eer": float(eer_crnn),
                "min_tdcf": float(tdcf_crnn),
                "roc_auc": float(auc_crnn),
                "accuracy_at_dev_th": float(acc_crnn),
            },
            "stacked_ensemble": {
                "eer": float(eer_ens),
                "min_tdcf": float(tdcf_ens),
                "roc_auc": float(auc_ens),
                "accuracy_at_dev_th": float(acc_ens),
            },
        }

        per_attack_rows.append({
            "Attack ID": atk_id,
            "Taxonomy": "Known Variant" if is_variant else "Unseen Tech",
            "Synthesis Description": desc,
            "Trials": n_sp,
            "CRNN EER (%)": f"{eer_crnn:.2f}",
            "CRNN min-tDCF": f"{tdcf_crnn:.4f}",
            "Ensemble EER (%)": f"{eer_ens:.2f}",
            "Ensemble min-tDCF": f"{tdcf_ens:.4f}",
        })

    df_atk = pd.DataFrame(per_attack_rows)
    with open(tables_dir / "table6_per_attack_eer.md", "w") as f:
        f.write(df_atk.to_markdown(index=False) + "\n")
    with open(tables_dir / "table6_per_attack_eer.tex", "w") as f:
        f.write(df_atk.to_latex(index=False) + "\n")

    plot_per_attack_eer(attack_eers_crnn, figures_dir / "per_attack_eer_crnn.png")
    plot_per_attack_eer(attack_eers_ensemble, figures_dir / "per_attack_eer_ensemble.png")

    print("Running Leave-One-Generator-Out (LOGO) cross-validation on Train/Dev...")
    logo_results = {}
    logo_rows = []
    generators = ["A01", "A02", "A03", "A04", "A05", "A06"]
    gen_names = {
        "A01": "Neural TTS (WaveNet)",
        "A02": "Neural TTS (WORLD)",
        "A03": "Vocoder TTS (Traditional)",
        "A04": "Waveform Concatenation TTS",
        "A05": "Voice Conversion (Traditional)",
        "A06": "Voice Conversion (WORLD)",
    }

    tr_attacks = tr_df["attack_id"].values
    tr_labels = tr_df["label"].values
    dv_attacks = dv_df["attack_id"].values
    dv_labels = dv_df["label"].values

    for held_out in generators:
        tr_mask = (tr_attacks != held_out)
        dv_mask = (dv_attacks == held_out) | (dv_labels == BONAFIDE)

        x_tr_sub = d_tr["pooled"][tr_mask]
        y_tr_sub = tr_labels[tr_mask]
        x_dv_sub = d_dv["pooled"][dv_mask]
        y_dv_sub = dv_labels[dv_mask]

        sc_logo = StandardScaler()
        x_tr_scaled = sc_logo.fit_transform(x_tr_sub)
        x_dv_scaled = sc_logo.transform(x_dv_sub)

        lr_logo = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=seed)
        lr_logo.fit(x_tr_scaled, y_tr_sub)
        pred_dv = lr_logo.decision_function(x_dv_scaled)

        eer_logo, th_logo = compute_eer(y_dv_sub, pred_dv)
        auc_logo = compute_roc_auc(y_dv_sub, pred_dv)
        status = "PASS" if eer_logo < 35.0 else "FAIL"

        logo_results[held_out] = {
            "name": gen_names[held_out],
            "eer": float(eer_logo),
            "roc_auc": float(auc_logo),
            "status": status,
        }

        logo_rows.append({
            "Held-Out Generator": held_out,
            "Algorithm Family": gen_names[held_out],
            "Held-Out Dev EER (%)": f"{eer_logo:.2f}",
            "Held-Out Dev ROC-AUC": f"{auc_logo:.4f}",
            "Pass/Fail (Threshold < 35%)": status,
        })

    mean_logo_eer = float(np.mean([r["eer"] for r in logo_results.values()]))
    logo_rows.append({
        "Held-Out Generator": "MEAN LOGO",
        "Algorithm Family": "Average Across 6 Held-Out Folds",
        "Held-Out Dev EER (%)": f"{mean_logo_eer:.2f}",
        "Held-Out Dev ROC-AUC": f"{np.mean([r['roc_auc'] for r in logo_results.values()]):.4f}",
        "Pass/Fail (Threshold < 35%)": "PASS" if mean_logo_eer < 15.0 else "FAIL",
    })
    df_logo = pd.DataFrame(logo_rows)
    with open(tables_dir / "table7_logo_crossval.md", "w") as f:
        f.write(df_logo.to_markdown(index=False) + "\n")
    with open(tables_dir / "table7_logo_crossval.tex", "w") as f:
        f.write(df_logo.to_latex(index=False) + "\n")

    print("Running Silence Audit and Shortcut Bias Controls...")
    audit_clips_sample = 1000
    sub_eval_idx = np.random.RandomState(seed).choice(len(ev_df), audit_clips_sample, replace=False)
    sub_eval_df = ev_df.iloc[sub_eval_idx].reset_index(drop=True)

    silence_features_list = []
    acoustic_meta_list = []
    valid_paths = []
    valid_indices = []

    print(f"Extracting acoustic & silence features for {audit_clips_sample} representative trials...")
    for idx, r in sub_eval_df.iterrows():
        p = r["path"]
        try:
            sig, sr = sf.read(p)
            s_samp, e_samp, lead_s, trail_s, tot_s = get_vad_boundaries(sig)
            ac_meta = compute_clip_acoustic_meta(sig, sr=sr)
            dur = ac_meta["duration_sec"]
            sil_ratio = float(tot_s / (dur + 1e-12))
            silence_features_list.append([lead_s, trail_s, tot_s, sil_ratio])
            acoustic_meta_list.append(ac_meta)
            valid_paths.append(p)
            valid_indices.append(sub_eval_idx[idx])
        except Exception:
            continue

    silence_feats_arr = np.array(silence_features_list, dtype=np.float64)
    sub_labels = ev_y[valid_indices]
    sub_crnn_scores = crnn_eval_scores[valid_indices]
    sub_ens_scores = ensemble_eval_scores[valid_indices]

    print("Audit 1: Silence-Duration-Only Classifier...")
    tr_silence_list = []
    tr_sample_idx = np.random.RandomState(seed).choice(len(tr_df), 2000, replace=False)
    for idx in tr_sample_idx:
        p = tr_df.iloc[idx]["path"]
        try:
            sig, sr = sf.read(p)
            _, _, lead_s, trail_s, tot_s = get_vad_boundaries(sig)
            dur = len(sig) / sr
            sil_ratio = float(tot_s / (dur + 1e-12))
            tr_silence_list.append([lead_s, trail_s, tot_s, sil_ratio])
        except Exception:
            continue

    tr_sil_arr = np.array(tr_silence_list, dtype=np.float64)
    tr_sil_y = tr_labels[tr_sample_idx[: len(tr_sil_arr)]]

    scaler_sil = StandardScaler()
    tr_sil_scaled = scaler_sil.fit_transform(tr_sil_arr)
    ev_sil_scaled = scaler_sil.transform(silence_feats_arr)

    sil_clf = LogisticRegression(class_weight="balanced", random_state=seed)
    sil_clf.fit(tr_sil_scaled, tr_sil_y)
    sil_preds = sil_clf.decision_function(ev_sil_scaled)
    sil_eer, _ = compute_eer(sub_labels, sil_preds)
    sil_auc = compute_roc_auc(sub_labels, sil_preds)
    sil_status = "PASS" if sil_eer >= 35.0 and sil_auc <= 0.65 else "FAIL"

    print("Audit 2: VAD-Trimmed Audio Evaluation...")
    extractor_lfcc = LFCCExtractor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    trimmed_scores_crnn = []
    for p in valid_paths[:200]:
        sig, sr = sf.read(p)
        s_samp, e_samp, _, _, _ = get_vad_boundaries(sig)
        trimmed = sig[s_samp:e_samp]
        if len(trimmed) < 1600:
            trimmed = sig
        w_tensor = torch.from_numpy(trimmed).float()
        w_tensor = w_tensor if w_tensor.ndim == 1 else w_tensor.mean(dim=0)
        from acforensics.data.segmentation import fixed_length_crop_or_pad
        w_tensor = fixed_length_crop_or_pad(w_tensor, target_samples=64000, mode="center")
        with torch.no_grad():
            feat = extractor_lfcc(w_tensor)
            feat_norm = per_utterance_cmvn(feat).unsqueeze(0).unsqueeze(0).to(device)
            logits, _ = crnn(feat_norm)
            score = (logits[:, BONAFIDE] - logits[:, SPOOF]).item()
            trimmed_scores_crnn.append(score)

    sub_200_labels = sub_labels[:200]
    sub_200_untrimmed_scores = sub_crnn_scores[:200]
    trimmed_scores_crnn = np.array(trimmed_scores_crnn)
    eer_untrimmed, _ = compute_eer(sub_200_labels, sub_200_untrimmed_scores)
    eer_trimmed, _ = compute_eer(sub_200_labels, trimmed_scores_crnn)
    delta_trim_eer = abs(eer_trimmed - eer_untrimmed)
    trim_status = "PASS" if delta_trim_eer <= 5.0 else "FAIL"

    print("Audit 3: Silence-Swap Test...")
    bf_v_idx = np.where(sub_labels[:200] == BONAFIDE)[0]
    sp_v_idx = np.where(sub_labels[:200] == SPOOF)[0]
    n_pairs = min(len(bf_v_idx), len(sp_v_idx))
    orig_pair_scores = []
    swapped_pair_scores = []

    for k in range(n_pairs):
        p_bf = valid_paths[bf_v_idx[k]]
        p_sp = valid_paths[sp_v_idx[k]]
        sig_bf, _ = sf.read(p_bf)
        sig_sp, _ = sf.read(p_sp)
        s_bf, e_bf, _, _, _ = get_vad_boundaries(sig_bf)
        s_sp, e_sp, _, _, _ = get_vad_boundaries(sig_sp)

        sp_swapped = np.concatenate([sig_bf[:s_bf], sig_sp[s_sp:e_sp], sig_bf[e_bf:]])
        w_tensor = torch.from_numpy(sp_swapped).float()
        w_tensor = fixed_length_crop_or_pad(w_tensor, target_samples=64000, mode="center")
        with torch.no_grad():
            feat = extractor_lfcc(w_tensor)
            feat_norm = per_utterance_cmvn(feat).unsqueeze(0).unsqueeze(0).to(device)
            logits, _ = crnn(feat_norm)
            sc_swapped = (logits[:, BONAFIDE] - logits[:, SPOOF]).item()

        orig_pair_scores.append(sub_crnn_scores[sp_v_idx[k]])
        swapped_pair_scores.append(sc_swapped)

    orig_pair_arr = np.array(orig_pair_scores)
    swapped_pair_arr = np.array(swapped_pair_scores)
    r_swap, _ = pearsonr(orig_pair_arr, swapped_pair_arr)
    flip_rate = float(np.mean((orig_pair_arr >= dev_crnn_th) != (swapped_pair_arr >= dev_crnn_th)) * 100.0)
    swap_status = "PASS" if r_swap >= 0.80 and flip_rate <= 10.0 else "FAIL"

    print("Audit 4, 5, 6: Duration, Loudness, Bandwidth Controls...")
    durations = np.array([m["duration_sec"] for m in acoustic_meta_list])
    loudness = np.array([m["rms_dbfs"] for m in acoustic_meta_list])
    rolloff = np.array([m["rolloff_95_hz"] for m in acoustic_meta_list])

    r_dur_crnn, _ = pearsonr(durations, sub_crnn_scores)
    r_dur_ens, _ = pearsonr(durations, sub_ens_scores)
    dur_status = "PASS" if abs(r_dur_crnn) <= 0.20 and abs(r_dur_ens) <= 0.20 else "FAIL"

    r_loud_crnn, _ = pearsonr(loudness, sub_crnn_scores)
    r_loud_ens, _ = pearsonr(loudness, sub_ens_scores)
    loud_status = "PASS" if abs(r_loud_crnn) <= 0.20 and abs(r_loud_ens) <= 0.20 else "FAIL"

    r_bw_crnn, _ = pearsonr(rolloff, sub_crnn_scores)
    r_bw_ens, _ = pearsonr(rolloff, sub_ens_scores)
    bw_status = "PASS" if abs(r_bw_crnn) <= 0.35 and abs(r_bw_ens) <= 0.35 else "FAIL"

    audit_summary_table = [
        {
            "Audit Check": "Silence Duration Classifier",
            "Observed Metric": f"EER = {sil_eer:.2f}%, AUC = {sil_auc:.4f}",
            "Criterion": "EER >= 35.0%, AUC <= 0.65",
            "Status": sil_status,
        },
        {
            "Audit Check": "VAD-Trimmed vs Untrimmed Audio",
            "Observed Metric": f"Untrimmed: {eer_untrimmed:.2f}%, Trimmed: {eer_trimmed:.2f}%, Delta: {delta_trim_eer:.2f} pts",
            "Criterion": "|Delta EER| <= 5.0 pts",
            "Status": trim_status,
        },
        {
            "Audit Check": "Silence-Swap Score Stability",
            "Observed Metric": f"Pearson r = {r_swap:.4f}, Prediction Flip Rate = {flip_rate:.1f}%",
            "Criterion": "r >= 0.80, Flip Rate <= 10.0%",
            "Status": swap_status,
        },
        {
            "Audit Check": "Duration Control",
            "Observed Metric": f"CRNN r = {r_dur_crnn:+.4f}, Ensemble r = {r_dur_ens:+.4f}",
            "Criterion": "|r| <= 0.20",
            "Status": dur_status,
        },
        {
            "Audit Check": "Loudness Control (RMS dBFS)",
            "Observed Metric": f"CRNN r = {r_loud_crnn:+.4f}, Ensemble r = {r_loud_ens:+.4f}",
            "Criterion": "|r| <= 0.20",
            "Status": loud_status,
        },
        {
            "Audit Check": "Bandwidth Control (95% Rolloff)",
            "Observed Metric": f"CRNN r = {r_bw_crnn:+.4f}, Ensemble r = {r_bw_ens:+.4f}",
            "Criterion": "|r| <= 0.35",
            "Status": bw_status,
        },
    ]

    df_aud = pd.DataFrame(audit_summary_table)
    with open(tables_dir / "table8_audits_summary.md", "w") as f:
        f.write(df_aud.to_markdown(index=False) + "\n")
    with open(tables_dir / "table8_audits_summary.tex", "w") as f:
        f.write(df_aud.to_latex(index=False) + "\n")

    print("Running Forensic Error Analysis Audit...")
    error_audit_summary = run_error_audit(
        eval_df=ev_df,
        scores=crnn_eval_scores,
        threshold=dev_crnn_th,
        output_dir="results",
    )

    manifest_sha = compute_file_sha256("data/manifests/eval.parquet")
    git_hash = get_git_commit()
    ts = datetime.now(timezone.utc).isoformat()

    full_stage5_payload = {
        "dataset_name": "ASVspoof 2019 LA",
        "manifest_sha256": manifest_sha,
        "git_commit": git_hash,
        "seed": seed,
        "timestamp": ts,
        "speaker_bootstrap_eval": speaker_bootstrap_results,
        "per_attack_evaluation": per_attack_results,
        "logo_crossval": {
            "mean_logo_eer": mean_logo_eer,
            "folds": logo_results,
        },
        "audits": {
            "silence_duration_classifier": {
                "eer": float(sil_eer),
                "roc_auc": float(sil_auc),
                "status": sil_status,
            },
            "vad_trimming": {
                "untrimmed_eer": float(eer_untrimmed),
                "trimmed_eer": float(eer_trimmed),
                "delta_eer": float(delta_trim_eer),
                "status": trim_status,
            },
            "silence_swap": {
                "pearson_r": float(r_swap),
                "flip_rate_pct": float(flip_rate),
                "status": swap_status,
            },
            "controls": {
                "duration_r_crnn": float(r_dur_crnn),
                "duration_r_ensemble": float(r_dur_ens),
                "loudness_r_crnn": float(r_loud_crnn),
                "loudness_r_ensemble": float(r_loud_ens),
                "bandwidth_r_crnn": float(r_bw_crnn),
                "bandwidth_r_ensemble": float(r_bw_ens),
                "duration_status": dur_status,
                "loudness_status": loud_status,
                "bandwidth_status": bw_status,
            },
        },
        "error_audit": error_audit_summary,
    }

    guard_results_and_provenance(full_stage5_payload)
    out_json = metrics_dir / "stage5_audits_results.json"
    with open(out_json, "w") as f:
        json.dump(full_stage5_payload, f, indent=2)

    with open(metrics_dir / "audits.json", "w") as f:
        json.dump(full_stage5_payload["audits"], f, indent=2)
    with open(metrics_dir / "speaker_bootstrap.json", "w") as f:
        json.dump(speaker_bootstrap_results, f, indent=2)
    with open(metrics_dir / "per_attack_evaluation.json", "w") as f:
        json.dump(per_attack_results, f, indent=2)
    with open(metrics_dir / "logo_crossval.json", "w") as f:
        json.dump(full_stage5_payload["logo_crossval"], f, indent=2)
    with open(metrics_dir / "error_audit.json", "w") as f:
        json.dump(error_audit_summary, f, indent=2)

    print("STAGE 5 completed successfully. All artifacts written to results/.")
    return full_stage5_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()

    run_stage5_all(seed=args.seed, n_bootstraps=args.bootstraps)
