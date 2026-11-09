import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from acforensics.constants import SAMPLE_RATE, TARGET_SAMPLES
from acforensics.features.lfcc import LFCCExtractor

def load_audio_exact(path_str: str, target_samples: int = TARGET_SAMPLES) -> torch.Tensor:
    data, sr = sf.read(path_str, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio.functional as F
        t = torch.from_numpy(data).unsqueeze(0)
        t = F.resample(t, orig_freq=sr, new_freq=SAMPLE_RATE)
        data = t.squeeze(0).numpy()
    n = len(data)
    if n >= target_samples:
        start = (n - target_samples) // 2
        data = data[start : start + target_samples]
    else:
        pad = target_samples - n
        left = pad // 2
        right = pad - left
        data = np.pad(data, (left, right), mode="constant")
    return torch.from_numpy(data)

def cache_split(manifest_path: str, out_cache: str, batch_size: int = 128) -> None:
    p_manifest = Path(manifest_path)
    p_out = Path(out_cache)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    if p_manifest.suffix == ".parquet":
        df = pd.read_parquet(p_manifest)
    else:
        df = pd.read_csv(p_manifest)
    paths = df["path"].tolist()
    labels = df["label"].to_numpy(dtype=np.int64)
    keys = [Path(p).stem for p in paths]
    total = len(paths)
    ext = LFCCExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    all_features = []
    all_pooled = []
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_paths = paths[i : i + batch_size]
            waveforms = [load_audio_exact(p) for p in batch_paths]
            batch_wave = torch.stack(waveforms, dim=0)
            feat = ext(batch_wave)
            mean_v = torch.mean(feat, dim=2)
            std_v = torch.std(feat, dim=2)
            pooled = torch.cat([mean_v, std_v], dim=1)
            all_features.append(feat.cpu().to(torch.float16).numpy())
            all_pooled.append(pooled.cpu().numpy())
    feat_arr = np.concatenate(all_features, axis=0)
    pooled_arr = np.concatenate(all_pooled, axis=0)
    np.savez_compressed(
        str(p_out),
        features=feat_arr,
        pooled=pooled_arr,
        labels=labels,
        keys=np.array(keys, dtype=str),
    )
    print(f"Cached {total} items to {p_out} (features shape: {feat_arr.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out-cache", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    cache_split(args.manifest, args.out_cache, args.batch_size)

