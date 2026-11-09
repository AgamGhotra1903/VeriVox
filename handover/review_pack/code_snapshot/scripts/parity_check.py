import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from acforensics.constants import SAMPLE_RATE, TARGET_SAMPLES, BONAFIDE, SPOOF
from acforensics.evaluation.metrics import compute_eer, bootstrap_confidence_interval
from acforensics.evaluation.provenance import compute_file_sha256, create_provenance_record, guard_results_and_provenance
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics


def stratified_sample_eval(df: pd.DataFrame, n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    weights = df.groupby(["label", "attack_id"])["path"].transform("count") / len(df)
    sampled = df.sample(n=n_samples, weights=weights, random_state=seed)
    return sampled.reset_index(drop=True)


def run_parity_check(
    manifest_path: str = "data/manifests/eval.parquet",
    cached_npz_path: str = "data/cache/lfcc_eval.npz",
    checkpoint_path: str = "checkpoints/crnn_lfcc_best.pt",
    out_json: str = "results/metrics/parity_check.json",
    n_samples: int = 2000,
    seed: int = 42,
    device_name: str = "auto",
) -> dict:
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

    cached_p = Path(cached_npz_path)
    cached_data = np.load(cached_p)
    all_cached_features = cached_data["features"]
    all_cached_keys = cached_data["keys"].tolist()

    key_to_idx = {k: idx for idx, k in enumerate(all_cached_keys)}
    subset_keys = [Path(p).stem for p in df_subset["path"]]
    subset_cached_indices = [key_to_idx[k] for k in subset_keys]

    model = CRNNForensics(in_channels=1, in_freq_dim=60, features="lfcc")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()

    ext = LFCCExtractor(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_lfcc=20,
        with_deltas=True,
    ).to(device)

    cached_scores = []
    raw_scores = []
    ground_truth_labels = df_subset["label"].to_numpy(dtype=np.int64)

    batch_size = 64
    n_total = len(df_subset)

    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            b_paths = df_subset["path"].iloc[i : i + batch_size].tolist()
            b_c_indices = subset_cached_indices[i : i + batch_size]

            b_cached_raw = torch.from_numpy(all_cached_features[b_c_indices]).to(dtype=torch.float32, device=device)
            b_cached_norm = per_utterance_cmvn(b_cached_raw).unsqueeze(1)
            logits_c, _ = model(b_cached_norm)
            s_c = (logits_c[:, BONAFIDE] - logits_c[:, SPOOF]).cpu().numpy()
            cached_scores.append(s_c)

            b_waveforms = []
            for p in b_paths:
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
                b_waveforms.append(torch.from_numpy(data))

            b_wave = torch.stack(b_waveforms, dim=0).to(device=device, dtype=torch.float32)
            raw_feat = ext(b_wave)
            raw_norm = per_utterance_cmvn(raw_feat).unsqueeze(1)
            logits_r, _ = model(raw_norm)
            s_r = (logits_r[:, BONAFIDE] - logits_r[:, SPOOF]).cpu().numpy()
            raw_scores.append(s_r)

    s_cached_arr = np.concatenate(cached_scores, axis=0)
    s_raw_arr = np.concatenate(raw_scores, axis=0)

    score_diffs = np.abs(s_cached_arr - s_raw_arr)
    max_abs_diff = float(np.max(score_diffs))
    mean_abs_diff = float(np.mean(score_diffs))
    p95_abs_diff = float(np.percentile(score_diffs, 95))

    eer_cached, th_cached = compute_eer(ground_truth_labels, s_cached_arr)
    eer_raw, th_raw = compute_eer(ground_truth_labels, s_raw_arr)

    def eer_metric(lbls, scrs):
        return compute_eer(lbls, scrs)[0]

    ci_cached = bootstrap_confidence_interval(eer_metric, ground_truth_labels, s_cached_arr, n_bootstraps=1000, seed=seed)
    ci_raw = bootstrap_confidence_interval(eer_metric, ground_truth_labels, s_raw_arr, n_bootstraps=1000, seed=seed)

    eer_delta = abs(eer_cached - eer_raw)
    parity_passed = bool(max_abs_diff < 0.05 and eer_delta < 0.5)

    prov = create_provenance_record(
        dataset_name="ASVspoof 2019 LA Parity Check",
        manifest_path=manifest_path,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )

    result_payload = {
        **prov,
        "n_samples": n_samples,
        "subset": True,
        "subset_flag": True,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": compute_file_sha256(checkpoint_path),
        "parity_passed": parity_passed,
        "max_absolute_score_difference": max_abs_diff,
        "mean_absolute_score_difference": mean_abs_diff,
        "p95_absolute_score_difference": p95_abs_diff,
        "cached_path_eer": float(eer_cached),
        "cached_path_threshold": float(th_cached),
        "cached_path_eer_ci_95": [float(ci_cached[0]), float(ci_cached[1])],
        "raw_audio_path_eer": float(eer_raw),
        "raw_audio_path_threshold": float(th_raw),
        "raw_audio_path_eer_ci_95": [float(ci_raw[0]), float(ci_raw[1])],
        "eer_difference_pts": float(eer_delta),
        "specifications": {
            "sample_rate": SAMPLE_RATE,
            "target_samples": TARGET_SAMPLES,
            "crop_mode": "center_crop_4.0s",
            "pad_mode": "symmetric_constant_zero_pad",
            "normalization": "per_utterance_cmvn",
            "lfcc_filters": 20,
            "lfcc_coefficients": 20,
            "deltas": True,
            "deltadeltas": True,
            "total_features": 60,
            "peak_normalization": "none",
            "bonafide_label": BONAFIDE,
            "spoof_label": SPOOF,
        },
    }

    guard_results_and_provenance(result_payload)

    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    print(f"Parity check completed on N={n_samples} clips.")
    print(f"Max abs score diff: {max_abs_diff:.6e}")
    print(f"Cached path EER: {eer_cached:.2f}% (95% CI: [{ci_cached[0]:.2f}%, {ci_cached[1]:.2f}%])")
    print(f"Raw audio path EER: {eer_raw:.2f}% (95% CI: [{ci_raw[0]:.2f}%, {ci_raw[1]:.2f}%])")
    print(f"EER Delta: {eer_delta:.4f} pts")
    print(f"PARITY STATUS: {'PASS' if parity_passed else 'FAIL'}")

    return result_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acoustic Forensics Preprocessing Parity Check")
    parser.add_argument("--manifest", type=str, default="data/manifests/eval.parquet")
    parser.add_argument("--cached-npz", type=str, default="data/cache/lfcc_eval.npz")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crnn_lfcc_best.pt")
    parser.add_argument("--out-json", type=str, default="results/metrics/parity_check.json")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_parity_check(
        manifest_path=args.manifest,
        cached_npz_path=args.cached_npz,
        checkpoint_path=args.checkpoint,
        out_json=args.out_json,
        n_samples=args.samples,
        seed=args.seed,
        device_name=args.device,
    )
