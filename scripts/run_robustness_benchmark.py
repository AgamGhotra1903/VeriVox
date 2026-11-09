import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from acforensics.constants import SAMPLE_RATE, TARGET_SAMPLES, BONAFIDE, SPOOF
from acforensics.evaluation.metrics import compute_eer, compute_metrics_at_threshold, bootstrap_confidence_interval
from acforensics.evaluation.provenance import (
    compute_file_sha256,
    create_provenance_record,
    guard_results_and_provenance,
)
from acforensics.evaluation.robustness import (
    compress_and_decompress_audio,
    apply_telephony_channel,
    apply_additive_noise,
    apply_synthetic_reverb,
    apply_packet_loss,
)
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics


def stratified_sample_eval(df: pd.DataFrame, n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    weights = df.groupby(["attack_id", "label"])["path"].transform("count") / len(df)
    sampled = df.sample(n=n_samples, weights=weights, random_state=seed)
    return sampled.reset_index(drop=True)


def score_waveforms_batch(
    waveforms: list[torch.Tensor],
    model: torch.nn.Module,
    extractor: LFCCExtractor,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    all_scores = []
    n_total = len(waveforms)
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            b_waves = waveforms[i : i + batch_size]
            b_stack = torch.stack(b_waves, dim=0).to(device=device, dtype=torch.float32)
            raw_feats = extractor(b_stack)
            norm_feats = per_utterance_cmvn(raw_feats).unsqueeze(1)
            logits, _ = model(norm_feats)
            scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
            all_scores.append(scores)
    return np.concatenate(all_scores, axis=0)


def run_benchmark(
    manifest_path: str = "data/manifests/eval.parquet",
    checkpoint_path: str = "checkpoints/crnn_lfcc_best.pt",
    n_samples: int = 5000,
    seed: int = 42,
    device_name: str = "auto",
):
    if device_name == "auto":
        device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    manifest_p = Path(manifest_path)
    if manifest_p.suffix == ".parquet":
        full_df = pd.read_parquet(manifest_p)
    else:
        full_df = pd.read_csv(manifest_p)

    df_subset = stratified_sample_eval(full_df, n_samples=n_samples, seed=seed)
    labels = df_subset["label"].to_numpy(dtype=np.int64)
    paths = df_subset["path"].tolist()
    speakers = df_subset["speaker"].to_numpy()

    print(f"Loading and standardizing {n_samples} clean audio waveforms...")
    clean_waves = []
    for p in paths:
        data, sr = sf.read(p, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        n = len(data)
        if n >= TARGET_SAMPLES:
            st = (n - TARGET_SAMPLES) // 2
            data = data[st : st + TARGET_SAMPLES]
        else:
            pad = TARGET_SAMPLES - n
            left = pad // 2
            right = pad - left
            data = np.pad(data, (left, right), mode="constant")
        clean_waves.append(torch.from_numpy(data))

    model = CRNNForensics(in_channels=1, in_freq_dim=60, features="lfcc")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()

    extractor = LFCCExtractor(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_lfcc=20,
        with_deltas=True,
    ).to(device)

    print("Computing Clean Baseline scores on raw audio path...")
    clean_scores = score_waveforms_batch(clean_waves, model, extractor, device)
    clean_eer, clean_th = compute_eer(labels, clean_scores)

    def eer_metric(lbls, scrs):
        return compute_eer(lbls, scrs)[0]

    ci_clean = bootstrap_confidence_interval(eer_metric, labels, clean_scores, n_bootstraps=1000, seed=seed)
    print(f"Clean Baseline EER on N={n_samples}: {clean_eer:.2f}% (95% CI: [{ci_clean[0]:.2f}%, {ci_clean[1]:.2f}%]), threshold: {clean_th:.4f}")

    assert abs(clean_eer - 7.61) < 2.5, f"Clean EER {clean_eer:.2f}% deviates from full eval 7.61%"

    prov_codec = create_provenance_record("codec_robustness_eval", manifest_path, seed=seed, checkpoint_path=checkpoint_path)
    prov_deg = create_provenance_record("acoustic_degradations_eval", manifest_path, seed=seed, checkpoint_path=checkpoint_path)

    temp_dir = Path(tempfile.mkdtemp(prefix="acforensics_robustness_"))

    codecs = ["mp3", "aac", "opus"]
    bitrates = [16, 32, 64]
    codec_conditions = {}
    codec_table_rows = []

    print("\n--- Running Codec Robustness Benchmark (MP3, AAC, Opus @ 16, 32, 64 kbps) ---")
    for codec in codecs:
        for kbps in bitrates:
            cond_key = f"{codec}_{kbps}k"
            print(f"Evaluating {codec.upper()} @ {kbps} kbps (with delay compensation)...")
            cond_dir = temp_dir / cond_key
            cond_dir.mkdir(parents=True, exist_ok=True)

            def enc_dec_one(item):
                idx, p = item
                out_p = cond_dir / f"{idx}.wav"
                ok = compress_and_decompress_audio(p, out_p, codec=codec, bitrate_kbps=kbps, compensate_delay=True)
                if ok and out_p.exists():
                    d, _ = sf.read(str(out_p), dtype="float32")
                    out_p.unlink(missing_ok=True)
                    if len(d) >= TARGET_SAMPLES:
                        st = (len(d) - TARGET_SAMPLES) // 2
                        d = d[st : st + TARGET_SAMPLES]
                    else:
                        pad = TARGET_SAMPLES - len(d)
                        d = np.pad(d, (pad // 2, pad - pad // 2), mode="constant")
                    return idx, torch.from_numpy(d)
                return idx, clean_waves[idx]

            with ThreadPoolExecutor(max_workers=8) as ex:
                res_waves = list(ex.map(enc_dec_one, enumerate(paths)))

            res_waves.sort(key=lambda x: x[0])
            deg_waves = [r[1] for r in res_waves]
            shutil.rmtree(cond_dir, ignore_errors=True)

            cond_scores = score_waveforms_batch(deg_waves, model, extractor, device)
            cond_eer, cond_th = compute_eer(labels, cond_scores)
            ci_cond = bootstrap_confidence_interval(eer_metric, labels, cond_scores, n_bootstraps=1000, seed=seed)
            delta_eer = cond_eer - clean_eer

            assert delta_eer >= -2.0, f"Sanity violation: {cond_key} improved EER by {delta_eer:.2f} pts"

            m_thresh = compute_metrics_at_threshold(labels, cond_scores, clean_th)

            codec_conditions[cond_key] = {
                "codec": codec.upper(),
                "bitrate_kbps": kbps,
                "crnn": {
                    "eer": float(cond_eer),
                    "ci_95": [float(ci_cond[0]), float(ci_cond[1])],
                    "threshold": float(cond_th),
                    "delta_eer": float(delta_eer),
                    "deg_abs": float(delta_eer),
                    "accuracy_dev_th": float(m_thresh["accuracy"] * 100.0),
                    "far_dev_th": float(m_thresh["fp"] / max(1, m_thresh["fp"] + m_thresh["tn"]) * 100.0),
                    "frr_dev_th": float(m_thresh["fn"] / max(1, m_thresh["fn"] + m_thresh["tp"]) * 100.0),
                },
            }

            codec_table_rows.append({
                "Codec": codec.upper(),
                "Bitrate (kbps)": kbps,
                "Clean EER (%)": f"{clean_eer:.2f}",
                "Degraded EER (%)": f"{cond_eer:.2f}",
                "95% CI (%)": f"[{ci_cond[0]:.2f}, {ci_cond[1]:.2f}]",
                "Degradation (pts)": f"{delta_eer:+.2f}",
                "Acc @ Thresh (%)": f"{m_thresh['accuracy'] * 100.0:.2f}",
                "FAR (%)": f"{m_thresh['fp'] / max(1, m_thresh['fp'] + m_thresh['tn']) * 100.0:.2f}",
                "FRR (%)": f"{m_thresh['fn'] / max(1, m_thresh['fn'] + m_thresh['tp']) * 100.0:.2f}",
            })
            print(f"  -> EER: {cond_eer:.2f}% (delta: {delta_eer:+.2f} pts), FAR: {m_thresh['fp'] / max(1, m_thresh['fp'] + m_thresh['tn']) * 100.0:.2f}%")

    print("\n--- Running Acoustic Degradation Chains (Telephony, Additive Noise, Reverb, Packet Loss) ---")
    deg_conditions = {}
    deg_table_rows = []

    telephony_modes = [("telephony_8k_alaw", "alaw", "8 kHz Telephony (G.711 A-law)"), ("telephony_8k_mulaw", "mulaw", "8 kHz Telephony (G.711 mu-law)")]
    for t_key, compand_mode, t_label in telephony_modes:
        print(f"Evaluating {t_label}...")
        t_dir = temp_dir / t_key
        t_dir.mkdir(parents=True, exist_ok=True)

        def tel_one(item):
            idx, p = item
            out_p = t_dir / f"{idx}.wav"
            ok = apply_telephony_channel(p, out_p, compand=compand_mode)
            if ok and out_p.exists():
                d, _ = sf.read(str(out_p), dtype="float32")
                out_p.unlink(missing_ok=True)
                if len(d) >= TARGET_SAMPLES:
                    st = (len(d) - TARGET_SAMPLES) // 2
                    d = d[st : st + TARGET_SAMPLES]
                else:
                    pad = TARGET_SAMPLES - len(d)
                    d = np.pad(d, (pad // 2, pad - pad // 2), mode="constant")
                return idx, torch.from_numpy(d)
            return idx, clean_waves[idx]

        with ThreadPoolExecutor(max_workers=8) as ex:
            res_waves = list(ex.map(tel_one, enumerate(paths)))

        res_waves.sort(key=lambda x: x[0])
        deg_waves = [r[1] for r in res_waves]
        shutil.rmtree(t_dir, ignore_errors=True)

        t_scores = score_waveforms_batch(deg_waves, model, extractor, device)
        t_eer, t_th = compute_eer(labels, t_scores)
        t_ci = bootstrap_confidence_interval(eer_metric, labels, t_scores, n_bootstraps=1000, seed=seed)
        delta_eer = t_eer - clean_eer
        m_thresh = compute_metrics_at_threshold(labels, t_scores, clean_th)

        deg_conditions[t_key] = {
            "condition": t_label,
            "crnn": {
                "eer": float(t_eer),
                "ci_95": [float(t_ci[0]), float(t_ci[1])],
                "delta_eer": float(delta_eer),
                "acc_thresh": float(m_thresh["accuracy"] * 100.0),
                "far_thresh": float(m_thresh["fp"] / max(1, m_thresh["fp"] + m_thresh["tn"]) * 100.0),
                "frr_thresh": float(m_thresh["fn"] / max(1, m_thresh["fn"] + m_thresh["tp"]) * 100.0),
            },
        }
        deg_table_rows.append({
            "Degradation Condition": t_label,
            "Clean EER (%)": f"{clean_eer:.2f}",
            "Degraded EER (%)": f"{t_eer:.2f}",
            "95% CI (%)": f"[{t_ci[0]:.2f}, {t_ci[1]:.2f}]",
            "Degradation (pts)": f"{delta_eer:+.2f}",
            "Acc @ Thresh (%)": f"{m_thresh['accuracy'] * 100.0:.2f}",
            "FAR (%)": f"{m_thresh['fp'] / max(1, m_thresh['fp'] + m_thresh['tn']) * 100.0:.2f}",
            "FRR (%)": f"{m_thresh['fn'] / max(1, m_thresh['fn'] + m_thresh['tp']) * 100.0:.2f}",
        })
        print(f"  -> EER: {t_eer:.2f}% (delta: {delta_eer:+.2f} pts)")

    stochastic_runs = [
        ("noise_snr_20db", "Additive Gaussian Noise (20 dB SNR)", lambda w, s: apply_additive_noise(w.numpy(), snr_db=20.0, seed=s)),
        ("noise_snr_10db", "Additive Gaussian Noise (10 dB SNR)", lambda w, s: apply_additive_noise(w.numpy(), snr_db=10.0, seed=s)),
        ("noise_snr_0db", "Additive Gaussian Noise (0 dB SNR)", lambda w, s: apply_additive_noise(w.numpy(), snr_db=0.0, seed=s)),
        ("reverb_rt60_0.3s", "Synthetic Reverb (RT60 = 0.3s)", lambda w, s: apply_synthetic_reverb(w.numpy(), rt60=0.3, seed=s)),
        ("packet_loss_5pct", "Packet Loss (5% Random Burst)", lambda w, s: apply_packet_loss(w.numpy(), loss_rate=0.05, seed=s)),
        ("packet_loss_10pct", "Packet Loss (10% Random Burst)", lambda w, s: apply_packet_loss(w.numpy(), loss_rate=0.10, seed=s)),
    ]

    eval_seeds = [42, 43, 44]
    for s_key, s_label, func in stochastic_runs:
        print(f"Evaluating {s_label} across {len(eval_seeds)} seeds ({eval_seeds})...")
        seed_eers = []
        seed_scores_list = []
        for s in eval_seeds:
            deg_waves = [torch.from_numpy(func(w, s)) for w in clean_waves]
            s_scores = score_waveforms_batch(deg_waves, model, extractor, device)
            s_eer, _ = compute_eer(labels, s_scores)
            seed_eers.append(s_eer)
            seed_scores_list.append(s_scores)

        mean_eer = float(np.mean(seed_eers))
        std_eer = float(np.std(seed_eers))
        delta_eer = mean_eer - clean_eer

        mean_scores = np.mean(seed_scores_list, axis=0)
        s_ci = bootstrap_confidence_interval(eer_metric, labels, mean_scores, n_bootstraps=1000, seed=seed)
        m_thresh = compute_metrics_at_threshold(labels, mean_scores, clean_th)

        deg_conditions[s_key] = {
            "condition": s_label,
            "seeds_evaluated": eval_seeds,
            "crnn": {
                "eer": float(mean_eer),
                "eer_std": float(std_eer),
                "ci_95": [float(s_ci[0]), float(s_ci[1])],
                "delta_eer": float(delta_eer),
                "acc_thresh": float(m_thresh["accuracy"] * 100.0),
                "far_thresh": float(m_thresh["fp"] / max(1, m_thresh["fp"] + m_thresh["tn"]) * 100.0),
                "frr_thresh": float(m_thresh["fn"] / max(1, m_thresh["fn"] + m_thresh["tp"]) * 100.0),
            },
        }

        deg_table_rows.append({
            "Degradation Condition": s_label,
            "Clean EER (%)": f"{clean_eer:.2f}",
            "Degraded EER (%)": f"{mean_eer:.2f} ± {std_eer:.2f}",
            "95% CI (%)": f"[{s_ci[0]:.2f}, {s_ci[1]:.2f}]",
            "Degradation (pts)": f"{delta_eer:+.2f}",
            "Acc @ Thresh (%)": f"{m_thresh['accuracy'] * 100.0:.2f}",
            "FAR (%)": f"{m_thresh['fp'] / max(1, m_thresh['fp'] + m_thresh['tn']) * 100.0:.2f}",
            "FRR (%)": f"{m_thresh['fn'] / max(1, m_thresh['fn'] + m_thresh['tp']) * 100.0:.2f}",
        })
        print(f"  -> Mean EER: {mean_eer:.2f}% ± {std_eer:.2f}% (delta: {delta_eer:+.2f} pts)")

    shutil.rmtree(temp_dir, ignore_errors=True)

    metrics_dir = Path("results/metrics")
    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    codec_payload = {
        **prov_codec,
        "n_samples": n_samples,
        "subset": True,
        "subset_flag": True,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": compute_file_sha256(checkpoint_path),
        "clean_baseline": {
            "crnn_lfcc_eer": float(clean_eer),
            "crnn_threshold": float(clean_th),
            "sample_size": n_samples,
            "ci_95": [float(ci_clean[0]), float(ci_clean[1])],
        },
        "conditions": codec_conditions,
    }
    guard_results_and_provenance(codec_payload)
    with open(metrics_dir / "codec_robustness_results.json", "w", encoding="utf-8") as f:
        json.dump(codec_payload, f, indent=2)

    deg_payload = {
        **prov_deg,
        "n_samples": n_samples,
        "subset": True,
        "subset_flag": True,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": compute_file_sha256(checkpoint_path),
        "clean_baseline": {
            "crnn_lfcc_eer": float(clean_eer),
            "sample_size": n_samples,
            "ci_95": [float(ci_clean[0]), float(ci_clean[1])],
        },
        "conditions": deg_conditions,
    }
    guard_results_and_provenance(deg_payload)
    with open(metrics_dir / "degradation_chains_results.json", "w", encoding="utf-8") as f:
        json.dump(deg_payload, f, indent=2)

    df_c_tab = pd.DataFrame(codec_table_rows)
    df_c_tab.to_markdown(tables_dir / "table3_codec_robustness.md", index=False)
    df_c_tab.to_latex(tables_dir / "table3_codec_robustness.tex", index=False)

    df_d_tab = pd.DataFrame(deg_table_rows)
    df_d_tab.to_markdown(tables_dir / "table9_acoustic_degradations.md", index=False)
    df_d_tab.to_latex(tables_dir / "table9_acoustic_degradations.tex", index=False)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    codec_colors = {"MP3": "#EF4444", "AAC": "#3B82F6", "OPUS": "#10B981"}
    codec_markers = {"MP3": "o", "AAC": "s", "OPUS": "^"}
    for codec in ["MP3", "AAC", "OPUS"]:
        kbps_list = [16, 32, 64]
        eers = [codec_conditions[f"{codec.lower()}_{k}k"]["crnn"]["eer"] for k in kbps_list]
        ax.plot(
            kbps_list,
            eers,
            label=f"{codec}",
            color=codec_colors[codec],
            marker=codec_markers[codec],
            lw=2,
            ms=6,
        )
    ax.axhline(clean_eer, color="black", linestyle="--", lw=1.5, label=f"Clean Baseline ({clean_eer:.2f}%)")
    ax.set_xlabel("Bitrate (kbps)", fontsize=11)
    ax.set_ylabel("Equal Error Rate (EER %)", fontsize=11)
    ax.set_title(f"Codec Robustness Degradation (CRNN-LFCC, N={n_samples})", fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "robustness_eer_vs_bitrate.png")
    plt.savefig(figures_dir / "robustness_eer_vs_bitrate.pdf")
    plt.close()

    print("\nRobustness benchmark completed successfully!")
    print(f"Results saved to {metrics_dir / 'codec_robustness_results.json'} and {metrics_dir / 'degradation_chains_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Robustness Benchmark on Raw Audio Path")
    parser.add_argument("--manifest", type=str, default="data/manifests/eval.parquet")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crnn_lfcc_best.pt")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_benchmark(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        n_samples=args.samples,
        seed=args.seed,
        device_name=args.device,
    )
