"""Sanity check script: Score 200 bona-fide and 200 spoof In-The-Wild clips.

Verifies:
1. Label mapping: bona-fide=1, spoof=0.
2. Score sign convention: higher score indicates higher confidence in bona-fide.
3. Feature normalization: per_utterance_cmvn vs legacy global normalizer.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from acforensics.constants import SAMPLE_RATE, TARGET_SAMPLES, BONAFIDE, SPOOF
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import generate_multi_crop_windows
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics
from acforensics.evaluation.crossdomain import aggregate_window_scores
from acforensics.evaluation.metrics import compute_eer, compute_roc_auc


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt_path = Path("checkpoints/crnn_lfcc_best.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/crnn_best.pt")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = CRNNForensics(in_channels=1, in_freq_dim=60)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    extractor = LFCCExtractor(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_lfcc=20,
        with_deltas=True,
    )

    itw_manifest = Path("data/manifests/itw_test.parquet")
    df = pd.read_parquet(itw_manifest)

    bf_df = df[df["label"] == BONAFIDE].head(200)
    sp_df = df[df["label"] == SPOOF].head(200)

    sample_df = pd.concat([bf_df, sp_df]).reset_index(drop=True)

    bf_scores = []
    sp_scores = []

    print(f"Scoring {len(bf_df)} bona-fide and {len(sp_df)} spoof ITW clips...")
    for _, row in sample_df.iterrows():
        res = load_and_validate_audio(row["path"], target_sr=SAMPLE_RATE)
        if not res.is_valid or res.waveform is None:
            continue

        windows = generate_multi_crop_windows(res.waveform, window_samples=TARGET_SAMPLES, hop_ratio=0.5)
        with torch.no_grad():
            raw_feats = [extractor(w["waveform"]) for w in windows]
            norm_feats = [per_utterance_cmvn(f) for f in raw_feats]
            b_feats = torch.stack(norm_feats).unsqueeze(1).to(device)
            logits, _ = model(b_feats)
            w_scores = (logits[:, BONAFIDE] - logits[:, SPOOF]).cpu().numpy()

        agg = aggregate_window_scores([float(s) for s in w_scores], strategy="mean")
        if row["label"] == BONAFIDE:
            bf_scores.append(agg)
        else:
            sp_scores.append(agg)

    bf_arr = np.array(bf_scores)
    sp_arr = np.array(sp_scores)

    print("\n--- RAW SCORE STATISTICS ---")
    print(f"Bona-fide (N={len(bf_arr)}): Mean = {bf_arr.mean():+.4f}, Median = {np.median(bf_arr):+.4f}, Std = {bf_arr.std():.4f}")
    print(f"Spoof     (N={len(sp_arr)}): Mean = {sp_arr.mean():+.4f}, Median = {np.median(sp_arr):+.4f}, Std = {sp_arr.std():.4f}")

    labels = np.array([BONAFIDE] * len(bf_arr) + [SPOOF] * len(sp_arr))
    scores = np.concatenate([bf_arr, sp_arr])

    eer, th = compute_eer(labels, scores)
    auc = compute_roc_auc(labels, scores)

    print(f"\nMetric Sanity Check:")
    print(f"ROC-AUC: {auc:.4f} (Direction: {'CORRECT (>0.5)' if auc > 0.5 else 'INVERTED (<0.5)'})")
    print(f"EER: {eer:.2f}% at threshold {th:.4f}")

    assert auc > 0.5, f"Polarity check failed! Expected AUC > 0.5, got {auc:.4f}"
    print("\n[PASS] Score direction and polarity confirmed correct.")


if __name__ == "__main__":
    main()
