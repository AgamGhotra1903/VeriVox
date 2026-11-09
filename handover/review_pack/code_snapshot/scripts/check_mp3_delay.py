import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from acforensics.constants import SAMPLE_RATE, TARGET_SAMPLES, BONAFIDE, SPOOF
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics
from acforensics.evaluation.metrics import compute_eer
from acforensics.evaluation.robustness import compress_and_decompress_audio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crnn_lfcc_seed43.pt")
    parser.add_argument("--eval-parquet", type=str, default="data/manifests/eval.parquet")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/metrics/mp3_delay_check.json")
    return parser.parse_args()


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    model = CRNNForensics(in_channels=1).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def score_batch(model: torch.nn.Module, extractor: LFCCExtractor, waves: list[torch.Tensor], device: torch.device, batch_size: int = 64) -> list[float]:
    feats = [
        per_utterance_cmvn(
            extractor(
                fixed_length_crop_or_pad(w, target_samples=TARGET_SAMPLES, mode="center").to(device=device, dtype=torch.float32)
            )
        )
        for w in waves
    ]
    scores = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            b = torch.stack(feats[i:i+batch_size]).unsqueeze(1).to(device)
            logits, _ = model(b)
            s = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()
            scores.extend(s)
    return [float(x) for x in scores]


def main():
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint} on {device}...")
    model = load_model(args.checkpoint, device)
    extractor = LFCCExtractor(sample_rate=SAMPLE_RATE).to(device)

    print(f"Loading eval dataset from {args.eval_parquet}...")
    df = pd.read_parquet(args.eval_parquet)
    sample_dfs = []
    for (att, lbl), group in df.groupby(["attack_id", "label"]):
        n = max(1, int(round(len(group) / len(df) * args.n_samples)))
        sample_dfs.append(group.sample(n=min(n, len(group)), random_state=args.seed))
    sub_df = pd.concat(sample_dfs)
    if len(sub_df) < args.n_samples:
        rem = df.loc[~df.index.isin(sub_df.index)]
        sub_df = pd.concat([sub_df, rem.sample(n=args.n_samples - len(sub_df), random_state=args.seed)])
    elif len(sub_df) > args.n_samples:
        sub_df = sub_df.sample(n=args.n_samples, random_state=args.seed)
    sub_df = sub_df.reset_index(drop=True)

    labels = sub_df["label"].to_numpy().astype(np.int64)
    print(f"Sampled {len(sub_df)} clips (Bona Fide: {np.sum(labels == BONAFIDE)}, Spoof: {np.sum(labels == SPOOF)})")

    temp_dir = Path(tempfile.mkdtemp(prefix="mp3_delay_check_"))

    clean_waves = []
    for p in sub_df["path"]:
        res = load_and_validate_audio(p, target_sr=SAMPLE_RATE)
        clean_waves.append(res.waveform)

    clean_scores = score_batch(model, extractor, clean_waves, device)
    clean_eer, _ = compute_eer(labels, np.array(clean_scores))
    print(f"Clean Subset EER: {clean_eer:.2f}%")

    conditions = [
        {"codec": "mp3", "bitrate": 16, "compensate_delay": True, "name": "mp3_16k_delay_comp_on"},
        {"codec": "mp3", "bitrate": 16, "compensate_delay": False, "name": "mp3_16k_delay_comp_off"},
        {"codec": "mp3", "bitrate": 64, "compensate_delay": True, "name": "mp3_64k_delay_comp_on"},
        {"codec": "mp3", "bitrate": 64, "compensate_delay": False, "name": "mp3_64k_delay_comp_off"},
    ]

    results = {
        "checkpoint": args.checkpoint,
        "n_samples": len(sub_df),
        "clean_eer": float(clean_eer),
        "conditions": {},
    }

    try:
        for cond in conditions:
            c_name = cond["name"]
            kbps = cond["bitrate"]
            comp = cond["compensate_delay"]
            print(f"Processing condition {c_name} (bitrate={kbps}k, delay_comp={comp})...")
            cond_waves = []
            for idx, p in enumerate(sub_df["path"]):
                out_wav = temp_dir / f"clip_{idx}_{c_name}.wav"
                success = compress_and_decompress_audio(
                    input_wav=p,
                    output_wav=out_wav,
                    codec="mp3",
                    bitrate_kbps=kbps,
                    target_sr=SAMPLE_RATE,
                    compensate_delay=comp,
                )
                if success and out_wav.exists():
                    res = load_and_validate_audio(out_wav, target_sr=SAMPLE_RATE)
                    cond_waves.append(res.waveform)
                else:
                    cond_waves.append(clean_waves[idx])

            cond_scores = score_batch(model, extractor, cond_waves, device)
            cond_eer, _ = compute_eer(labels, np.array(cond_scores))
            score_diffs = np.abs(np.array(cond_scores) - np.array(clean_scores))
            mean_score_diff = float(np.mean(score_diffs))
            results["conditions"][c_name] = {
                "bitrate_kbps": kbps,
                "compensate_delay": comp,
                "eer": float(cond_eer),
                "mean_absolute_score_diff_vs_clean": mean_score_diff,
                "eer_delta_vs_clean": float(cond_eer - clean_eer),
            }
            print(f"  {c_name} -> EER: {cond_eer:.2f}% (delta: {cond_eer - clean_eer:+.2f} pts, mean |diff|: {mean_score_diff:.4f})")

        eer_16k_on = results["conditions"]["mp3_16k_delay_comp_on"]["eer"]
        eer_16k_off = results["conditions"]["mp3_16k_delay_comp_off"]["eer"]
        delta_16k = eer_16k_off - eer_16k_on
        results["summary_delta_16k_off_minus_on"] = float(delta_16k)

        eer_64k_on = results["conditions"]["mp3_64k_delay_comp_on"]["eer"]
        eer_64k_off = results["conditions"]["mp3_64k_delay_comp_off"]["eer"]
        delta_64k = eer_64k_off - eer_64k_on
        results["summary_delta_64k_off_minus_on"] = float(delta_64k)

        print("\n=== MP3 DELAY COMPENSATION SUMMARY ===")
        print(f"MP3 16k: Comp ON = {eer_16k_on:.2f}%, Comp OFF = {eer_16k_off:.2f}%, Delta (OFF - ON) = {delta_16k:+.2f} pts")
        print(f"MP3 64k: Comp ON = {eer_64k_on:.2f}%, Comp OFF = {eer_64k_off:.2f}%, Delta (OFF - ON) = {delta_64k:+.2f} pts")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_path}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
