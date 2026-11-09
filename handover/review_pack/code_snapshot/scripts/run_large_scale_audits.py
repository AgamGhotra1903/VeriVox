import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import scipy.stats
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from acforensics.constants import SAMPLE_RATE, TARGET_SAMPLES, BONAFIDE, SPOOF
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.evaluation.metrics import (
    compute_eer,
    compute_roc_auc,
    bootstrap_confidence_interval,
)
from acforensics.evaluation.provenance import (
    compute_file_sha256,
    create_provenance_record,
    guard_results_and_provenance,
)
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics


def get_vad_boundaries(sig: np.ndarray, hop: int = 160, frame_len: int = 320) -> tuple[int, int, float, float, float]:
    n = len(sig) // hop
    if n < 2:
        return 0, len(sig), 0.0, 0.0, 0.0
    energies = np.array([np.mean(sig[i * hop : i * hop + frame_len] ** 2) for i in range(n - 1)])
    th = np.max(energies) * 0.01 if len(energies) > 0 and np.max(energies) > 0 else 1e-6
    active = np.where(energies >= th)[0]
    if len(active) == 0:
        return 0, len(sig), 0.0, 0.0, 0.0
    s_frame = int(active[0])
    e_frame = int(active[-1])
    s_samp = max(0, s_frame * hop)
    e_samp = min(len(sig), (e_frame + 1) * hop)
    lead_s = float(s_samp / SAMPLE_RATE)
    trail_s = float(max(0, len(sig) - e_samp) / SAMPLE_RATE)
    inactive_frames = np.sum(energies < th)
    tot_s = float((inactive_frames * hop) / SAMPLE_RATE)
    return s_samp, e_samp, lead_s, trail_s, tot_s


def compute_spectral_centroid_and_rms(sig: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[float, float]:
    rms = float(np.sqrt(np.mean(sig ** 2) + 1e-12))
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / sr)
    sum_spec = np.sum(spec) + 1e-12
    centroid = float(np.sum(freqs * spec) / sum_spec)
    return rms, centroid


def score_feats_batch(
    feats_list: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(feats_list), batch_size):
            b = feats_list[i : i + batch_size]
            b_norm = [per_utterance_cmvn(f) for f in b]
            b_stack = torch.stack(b_norm, dim=0).unsqueeze(1).to(device=device, dtype=torch.float32)
            logits, _ = model(b_stack)
            sc = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
            all_scores.append(sc)
    return np.concatenate(all_scores, axis=0)


def run_large_scale_audits(
    eval_manifest: str = "data/manifests/eval.parquet",
    train_manifest: str = "data/manifests/train.parquet",
    checkpoint_path: str = "checkpoints/crnn_lfcc_best.pt",
    out_dir: str = "results/metrics",
    tables_dir: str = "results/tables",
    n_eval_samples: int = 20000,
    n_train_samples: int = 10000,
    seed: int = 42,
    device_name: str = "auto",
) -> dict[str, Any]:
    if device_name == "auto":
        device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    ev_df_full = pd.read_parquet(eval_manifest)
    weights_ev = ev_df_full.groupby(["attack_id", "label"])["path"].transform("count") / len(ev_df_full)
    ev_df = ev_df_full.sample(n=min(n_eval_samples, len(ev_df_full)), weights=weights_ev, random_state=seed).reset_index(drop=True)

    tr_df_full = pd.read_parquet(train_manifest)
    weights_tr = tr_df_full.groupby(["attack_id", "label"])["path"].transform("count") / len(tr_df_full)
    tr_df = tr_df_full.sample(n=min(n_train_samples, len(tr_df_full)), weights=weights_tr, random_state=seed).reset_index(drop=True)

    print(f"Loading and analyzing {len(ev_df)} eval clips for silence and acoustic confounders...")

    def process_eval_clip(item):
        idx, row = item
        p = row["path"]
        try:
            sig, sr = sf.read(p, dtype="float32")
            if sig.ndim > 1:
                sig = np.mean(sig, axis=1)
            s_samp, e_samp, lead_s, trail_s, tot_s = get_vad_boundaries(sig)
            dur = float(len(sig) / sr)
            sil_ratio = float(tot_s / (dur + 1e-12))
            rms, centroid = compute_spectral_centroid_and_rms(sig, sr)
            w_untrimmed = fixed_length_crop_or_pad(torch.from_numpy(sig), target_samples=TARGET_SAMPLES, mode="center")
            active_sig = sig[s_samp:e_samp]
            if len(active_sig) < 800:
                active_sig = sig
            w_trimmed = fixed_length_crop_or_pad(torch.from_numpy(active_sig), target_samples=TARGET_SAMPLES, mode="center")
            return idx, [lead_s, trail_s, tot_s, sil_ratio], rms, centroid, w_untrimmed, w_trimmed, p, int(row["label"])
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        ev_results = list(ex.map(process_eval_clip, ev_df.iterrows()))

    ev_valid = [r for r in ev_results if r is not None]
    ev_valid.sort(key=lambda x: x[0])

    ev_sil_feats = np.array([r[1] for r in ev_valid], dtype=np.float64)
    ev_rms = np.array([r[2] for r in ev_valid], dtype=np.float64)
    ev_centroid = np.array([r[3] for r in ev_valid], dtype=np.float64)
    ev_waves_untrimmed = [r[4] for r in ev_valid]
    ev_waves_trimmed = [r[5] for r in ev_valid]
    ev_paths = [r[6] for r in ev_valid]
    ev_labels = np.array([r[7] for r in ev_valid], dtype=np.int64)

    print(f"Loading and analyzing {len(tr_df)} train clips for silence classifier...")

    def process_train_clip(item):
        idx, row = item
        p = row["path"]
        try:
            sig, sr = sf.read(p, dtype="float32")
            if sig.ndim > 1:
                sig = np.mean(sig, axis=1)
            s_samp, e_samp, lead_s, trail_s, tot_s = get_vad_boundaries(sig)
            dur = float(len(sig) / sr)
            sil_ratio = float(tot_s / (dur + 1e-12))
            return idx, [lead_s, trail_s, tot_s, sil_ratio], int(row["label"])
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        tr_results = list(ex.map(process_train_clip, tr_df.iterrows()))

    tr_valid = [r for r in tr_results if r is not None]
    tr_valid.sort(key=lambda x: x[0])
    tr_sil_feats = np.array([r[1] for r in tr_valid], dtype=np.float64)
    tr_labels = np.array([r[2] for r in tr_valid], dtype=np.int64)

    print("Running Audit 1: Silence-Duration-Only Classifier...")
    scaler = StandardScaler()
    tr_sil_scaled = scaler.fit_transform(tr_sil_feats)
    ev_sil_scaled = scaler.transform(ev_sil_feats)

    sil_clf = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=500)
    sil_clf.fit(tr_sil_scaled, tr_labels)
    sil_scores = sil_clf.decision_function(ev_sil_scaled)

    sil_eer, sil_th = compute_eer(ev_labels, sil_scores)
    sil_auc = compute_roc_auc(ev_labels, sil_scores)
    sil_ci = bootstrap_confidence_interval(lambda l, s: compute_eer(l, s)[0], ev_labels, sil_scores, n_bootstraps=1000, seed=seed)
    sil_status = "PASS" if sil_eer >= 35.0 and sil_auc <= 0.65 else "FAIL"
    print(f"  -> Silence Classifier EER: {sil_eer:.2f}% (95% CI: [{sil_ci[0]:.2f}%, {sil_ci[1]:.2f}%]), AUC: {sil_auc:.4f} [Status: {sil_status}]")

    print("Running Audit 2: VAD-Trimmed vs Untrimmed Audio...")
    model = CRNNForensics(in_channels=1, in_freq_dim=60, features="lfcc")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()

    extractor = LFCCExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True).to(device)

    def extract_lfcc_waves(waves_list: list[torch.Tensor], batch_size: int = 128) -> list[torch.Tensor]:
        feats_all = []
        with torch.no_grad():
            for i in range(0, len(waves_list), batch_size):
                b = torch.stack(waves_list[i : i + batch_size], dim=0).to(device=device, dtype=torch.float32)
                f = extractor(b)
                feats_all.extend([f[j].cpu() for j in range(len(f))])
        return feats_all

    feats_untrimmed = extract_lfcc_waves(ev_waves_untrimmed)
    feats_trimmed = extract_lfcc_waves(ev_waves_trimmed)

    scores_untrimmed = score_feats_batch(feats_untrimmed, model, device)
    scores_trimmed = score_feats_batch(feats_trimmed, model, device)

    eer_untrimmed, th_untrimmed = compute_eer(ev_labels, scores_untrimmed)
    eer_trimmed, th_trimmed = compute_eer(ev_labels, scores_trimmed)
    delta_trim = float(abs(eer_trimmed - eer_untrimmed))
    trim_ci = bootstrap_confidence_interval(lambda l, s: compute_eer(l, s)[0], ev_labels, scores_trimmed, n_bootstraps=1000, seed=seed)
    trim_status = "PASS" if delta_trim <= 5.0 else "FAIL"
    print(f"  -> Untrimmed EER: {eer_untrimmed:.2f}%, Trimmed EER: {eer_trimmed:.2f}%, Delta: {delta_trim:.2f} pts [Status: {trim_status}]")

    print("Running Audit 3: Silence Swap Experiment...")
    n_pairs = min(2000, len(ev_valid) // 2)
    bf_indices = np.where(ev_labels == BONAFIDE)[0]
    sp_indices = np.where(ev_labels == SPOOF)[0]
    rng_p = np.random.RandomState(seed)
    bf_sel = rng_p.choice(bf_indices, size=n_pairs, replace=False)
    sp_sel = rng_p.choice(sp_indices, size=n_pairs, replace=False)

    def swap_pair(k):
        p_bf = ev_paths[bf_sel[k]]
        p_sp = ev_paths[sp_sel[k]]
        sig_bf, _ = sf.read(p_bf, dtype="float32")
        sig_sp, _ = sf.read(p_sp, dtype="float32")
        if sig_bf.ndim > 1:
            sig_bf = np.mean(sig_bf, axis=1)
        if sig_sp.ndim > 1:
            sig_sp = np.mean(sig_sp, axis=1)
        s_bf, e_bf, _, _, _ = get_vad_boundaries(sig_bf)
        s_sp, e_sp, _, _, _ = get_vad_boundaries(sig_sp)
        swapped = np.concatenate([sig_bf[:s_bf], sig_sp[s_sp:e_sp], sig_bf[e_bf:]])
        w_swapped = fixed_length_crop_or_pad(torch.from_numpy(swapped), target_samples=TARGET_SAMPLES, mode="center")
        w_orig = fixed_length_crop_or_pad(torch.from_numpy(sig_sp), target_samples=TARGET_SAMPLES, mode="center")
        return w_orig, w_swapped

    with ThreadPoolExecutor(max_workers=8) as ex:
        pairs_res = list(ex.map(swap_pair, range(n_pairs)))

    orig_sp_waves = [pr[0] for pr in pairs_res]
    swapped_waves = [pr[1] for pr in pairs_res]

    f_orig = extract_lfcc_waves(orig_sp_waves)
    f_swap = extract_lfcc_waves(swapped_waves)

    sc_orig = score_feats_batch(f_orig, model, device)
    sc_swap = score_feats_batch(f_swap, model, device)

    corr, _ = scipy.stats.pearsonr(sc_orig, sc_swap)
    flips = np.sum((sc_orig >= th_untrimmed) != (sc_swap >= th_untrimmed))
    flip_rate = float(flips / n_pairs * 100.0)
    swap_status = "PASS" if corr >= 0.85 and flip_rate <= 10.0 else "FAIL"
    print(f"  -> Silence Swap Pearson r: {corr:.4f}, Decision Flip Rate: {flip_rate:.2f}% [Status: {swap_status}]")

    print("Running Audit 4: Loudness and Bandwidth Confounder Controls...")
    q_rms = np.quantile(ev_rms, [0.333, 0.667])
    rms_groups = {
        "low_loudness": ev_rms <= q_rms[0],
        "mid_loudness": (ev_rms > q_rms[0]) & (ev_rms <= q_rms[1]),
        "high_loudness": ev_rms > q_rms[1],
    }

    rms_results = {}
    for g_name, mask in rms_groups.items():
        sub_l = ev_labels[mask]
        sub_s = scores_untrimmed[mask]
        sub_eer, _ = compute_eer(sub_l, sub_s)
        sub_auc = compute_roc_auc(sub_l, sub_s)
        sub_ci = bootstrap_confidence_interval(lambda l, s: compute_eer(l, s)[0], sub_l, sub_s, n_bootstraps=500, seed=seed)
        rms_results[g_name] = {
            "n_samples": int(np.sum(mask)),
            "eer": float(sub_eer),
            "ci_95": [float(sub_ci[0]), float(sub_ci[1])],
            "roc_auc": float(sub_auc),
        }

    q_cent = np.quantile(ev_centroid, [0.333, 0.667])
    cent_groups = {
        "narrowband_centroid": ev_centroid <= q_cent[0],
        "standard_centroid": (ev_centroid > q_cent[0]) & (ev_centroid <= q_cent[1]),
        "wideband_centroid": ev_centroid > q_cent[1],
    }

    cent_results = {}
    for g_name, mask in cent_groups.items():
        sub_l = ev_labels[mask]
        sub_s = scores_untrimmed[mask]
        sub_eer, _ = compute_eer(sub_l, sub_s)
        sub_auc = compute_roc_auc(sub_l, sub_s)
        sub_ci = bootstrap_confidence_interval(lambda l, s: compute_eer(l, s)[0], sub_l, sub_s, n_bootstraps=500, seed=seed)
        cent_results[g_name] = {
            "n_samples": int(np.sum(mask)),
            "eer": float(sub_eer),
            "ci_95": [float(sub_ci[0]), float(sub_ci[1])],
            "roc_auc": float(sub_auc),
        }

    prov = create_provenance_record("forensic_audits_large_scale", eval_manifest, seed=seed, checkpoint_path=checkpoint_path)

    payload = {
        **prov,
        "n_samples": len(ev_valid),
        "subset": bool(len(ev_valid) < len(ev_df_full)),
        "subset_flag": bool(len(ev_valid) < len(ev_df_full)),
        "silence_duration_classifier": {
            "n_train_samples": len(tr_valid),
            "n_eval_samples": len(ev_valid),
            "eer": float(sil_eer),
            "ci_95": [float(sil_ci[0]), float(sil_ci[1])],
            "roc_auc": float(sil_auc),
            "threshold": float(sil_th),
            "status": sil_status,
        },
        "vad_trimming": {
            "n_eval_samples": len(ev_valid),
            "untrimmed_eer": float(eer_untrimmed),
            "trimmed_eer": float(eer_trimmed),
            "ci_95": [float(trim_ci[0]), float(trim_ci[1])],
            "delta_eer": float(delta_trim),
            "status": trim_status,
        },
        "silence_swap": {
            "n_pairs": int(n_pairs),
            "n_trials": int(n_pairs * 2),
            "pearson_r": float(corr),
            "flip_rate_pct": float(flip_rate),
            "status": swap_status,
        },
        "loudness_controls": rms_results,
        "bandwidth_controls": cent_results,
    }

    guard_results_and_provenance(payload)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    with open(out_p / "audits.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(out_p / "crnn_silence_audit.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    tab_p = Path(tables_dir)
    tab_p.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "Audit Check": f"Silence-Duration-Only Classifier (N={len(ev_valid)})",
            "Observed Metric": f"EER = {sil_eer:.2f}% [{sil_ci[0]:.2f}%, {sil_ci[1]:.2f}%], AUC = {sil_auc:.4f}",
            "Criterion": "EER >= 35.0%, AUC <= 0.65",
            "Status": sil_status,
        },
        {
            "Audit Check": f"VAD-Trimmed vs Untrimmed Audio (N={len(ev_valid)})",
            "Observed Metric": f"Untrimmed: {eer_untrimmed:.2f}%, Trimmed: {eer_trimmed:.2f}%, Delta: {delta_trim:.2f} pts",
            "Criterion": "|Delta EER| <= 5.0 pts",
            "Status": trim_status,
        },
        {
            "Audit Check": f"Silence Swap Splicing (N={n_pairs} pairs)",
            "Observed Metric": f"Pearson r = {corr:.4f}, Flip Rate = {flip_rate:.2f}%",
            "Criterion": "r >= 0.85, Flip Rate <= 10.0%",
            "Status": swap_status,
        },
    ]

    df_tab = pd.DataFrame(rows)
    df_tab.to_markdown(tab_p / "table7_silence_audit.md", index=False)
    df_tab.to_latex(tab_p / "table7_silence_audit.tex", index=False)
    print("Saved Table 7 silence audit tables.")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Large Scale Audits on Eval Set")
    parser.add_argument("--eval-manifest", type=str, default="data/manifests/eval.parquet")
    parser.add_argument("--train-manifest", type=str, default="data/manifests/train.parquet")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crnn_lfcc_best.pt")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_large_scale_audits(
        eval_manifest=args.eval_manifest,
        train_manifest=args.train_manifest,
        checkpoint_path=args.checkpoint,
        n_eval_samples=args.samples,
        seed=args.seed,
        device_name=args.device,
    )
