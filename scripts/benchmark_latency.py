from pathlib import Path
from typing import Any
import json
import time
import platform
import subprocess
import numpy as np
import torch
import soundfile as sf

from acforensics.constants import (
    SAMPLE_RATE,
    TARGET_SAMPLES,
    DEFAULT_RESULTS_DIR,
    BONAFIDE,
    SPOOF,
)
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import FeatureNormalizer
from acforensics.models.crnn import CRNNForensics
from acforensics.ensemble.predictor import EnsemblePredictor
from acforensics.evaluation.provenance import create_provenance_record
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)

def get_cpu_brand() -> str:
    if platform.system() == "Darwin":
        try:
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True)
            brand = res.stdout.strip()
            if brand:
                return brand
        except Exception:
            pass
    return platform.processor() or platform.machine()

def benchmark_pipeline(
    device_name: str,
    device: torch.device,
    model: torch.nn.Module,
    extractor: LFCCExtractor,
    normalizer: FeatureNormalizer,
    audio_path: Path,
    num_warmup: int = 50,
    num_timed: int = 500,
    threads: int = 1,
) -> dict[str, Any]:
    if device.type == "cpu":
        torch.set_num_threads(threads)

    times_end_to_end = []
    times_io = []
    times_feature = []
    times_forward = []

    for _ in range(num_warmup):
        res = load_and_validate_audio(audio_path, target_sr=SAMPLE_RATE)
        wave = fixed_length_crop_or_pad(res.waveform, target_samples=TARGET_SAMPLES, mode="center")
        feat = extractor(wave)
        if normalizer.is_fitted:
            feat = normalizer(feat)
        batch = feat.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(batch)
            if device.type == "mps":
                torch.mps.synchronize()
            _ = (logits[:, BONAFIDE] - logits[:, SPOOF]).item()

    for _ in range(num_timed):
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        res = load_and_validate_audio(audio_path, target_sr=SAMPLE_RATE)
        wave = fixed_length_crop_or_pad(res.waveform, target_samples=TARGET_SAMPLES, mode="center")
        t_io = time.perf_counter() - t0

        t1 = time.perf_counter()
        feat = extractor(wave)
        if normalizer.is_fitted:
            feat = normalizer(feat)
        t_feat = time.perf_counter() - t1

        t2 = time.perf_counter()
        batch = feat.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(batch)
            if device.type == "mps":
                torch.mps.synchronize()
            _ = (logits[:, BONAFIDE] - logits[:, SPOOF]).item()
        t_fwd = time.perf_counter() - t2

        t_total = time.perf_counter() - t_start

        times_end_to_end.append(t_total * 1000.0)
        times_io.append(t_io * 1000.0)
        times_feature.append(t_feat * 1000.0)
        times_forward.append(t_fwd * 1000.0)

    arr = np.array(times_end_to_end)
    clip_dur_s = float(TARGET_SAMPLES / SAMPLE_RATE)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    rtf = float((mean_val / 1000.0) / clip_dur_s)
    throughput = float(1000.0 / mean_val)

    return {
        "device": device_name,
        "threads": threads if device.type == "cpu" else "N/A",
        "num_warmup": num_warmup,
        "num_timed_runs": num_timed,
        "clip_duration_sec": clip_dur_s,
        "latency_ms": {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
        },
        "breakdown_ms": {
            "audio_io_and_resample": round(float(np.mean(times_io)), 2),
            "feature_extraction": round(float(np.mean(times_feature)), 2),
            "model_forward_pass": round(float(np.mean(times_forward)), 2),
        },
        "real_time_factor": round(rtf, 5),
        "throughput_clips_per_sec": round(throughput, 1),
    }

def main():
    logger.info("Initializing Latency and Throughput Benchmark...")
    results_dir = Path(DEFAULT_RESULTS_DIR)
    metrics_dir = results_dir / "metrics"
    tables_dir = results_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    dummy_audio_path = Path("data/temp_latency_test.wav")
    rng = np.random.RandomState(42)
    tone = rng.randn(TARGET_SAMPLES).astype(np.float32)
    sf.write(str(dummy_audio_path), tone, SAMPLE_RATE)

    cpu_device = torch.device("cpu")
    crnn_ckpt_path = Path("checkpoints/crnn_lfcc_best.pt")
    if not crnn_ckpt_path.exists():
        crnn_ckpt_path = Path("checkpoints/crnn_best.pt")
    ckpt = torch.load(str(crnn_ckpt_path), map_location=cpu_device, weights_only=False)

    model_cpu = CRNNForensics(in_channels=1, in_freq_dim=60)
    model_cpu.load_state_dict(ckpt["model_state_dict"])
    model_cpu.eval()

    extractor = LFCCExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    norm_file = Path("checkpoints/normalization_stats_lfcc.json")
    if not norm_file.exists():
        norm_file = Path("checkpoints/normalization_stats.json")
    normalizer = FeatureNormalizer().load(norm_file) if norm_file.exists() else FeatureNormalizer()

    eval_manifest = Path("data/manifests/eval.parquet")
    prov = create_provenance_record("latency_benchmark", eval_manifest, seed=42)

    cpu_brand = get_cpu_brand()
    try:
        res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
        total_ram_gb = round(int(res.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        total_ram_gb = 24.0

    logger.info("Benchmarking CRNN-LFCC on CPU (pinned threads=1)...")
    res_cpu_1 = benchmark_pipeline(
        device_name=f"CPU ({cpu_brand}, 1 thread)",
        device=cpu_device,
        model=model_cpu,
        extractor=extractor,
        normalizer=normalizer,
        audio_path=dummy_audio_path,
        num_warmup=50,
        num_timed=500,
        threads=1,
    )

    logger.info("Benchmarking CRNN-LFCC on CPU (multi-thread=4)...")
    res_cpu_4 = benchmark_pipeline(
        device_name=f"CPU ({cpu_brand}, 4 threads)",
        device=cpu_device,
        model=model_cpu,
        extractor=extractor,
        normalizer=normalizer,
        audio_path=dummy_audio_path,
        num_warmup=50,
        num_timed=500,
        threads=4,
    )

    res_mps = None
    if torch.backends.mps.is_available():
        logger.info("Benchmarking CRNN-LFCC on Apple Silicon GPU (MPS)...")
        mps_device = torch.device("mps")
        model_mps = CRNNForensics(in_channels=1, in_freq_dim=60)
        model_mps.load_state_dict(ckpt["model_state_dict"])
        model_mps.to(mps_device)
        model_mps.eval()
        res_mps = benchmark_pipeline(
            device_name="Apple Silicon GPU (MPS)",
            device=mps_device,
            model=model_mps,
            extractor=extractor,
            normalizer=normalizer,
            audio_path=dummy_audio_path,
            num_warmup=50,
            num_timed=500,
        )

    dummy_audio_path.unlink(missing_ok=True)

    latency_payload = {
        **prov,
        "hardware_environment": {
            "cpu_model": cpu_brand,
            "total_ram_gb": total_ram_gb,
            "os": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "benchmarks": {
            "cpu_single_thread": res_cpu_1,
            "cpu_multi_thread": res_cpu_4,
            "apple_silicon_mps": res_mps,
        },
    }

    out_json = metrics_dir / "latency.json"
    with open(out_json, "w") as f:
        json.dump(latency_payload, f, indent=2)
    logger.info("Saved latency benchmark results to %s", out_json)

    table_rows = []
    for k, bench in latency_payload["benchmarks"].items():
        if bench is not None:
            lat = bench["latency_ms"]
            brk = bench["breakdown_ms"]
            table_rows.append({
                "Execution Target": bench["device"],
                "Threads": bench["threads"],
                "Mean ± Std (ms)": f"{lat['mean']} ± {lat['std']}",
                "p50 (ms)": f"{lat['p50']}",
                "p95 (ms)": f"{lat['p95']}",
                "p99 (ms)": f"{lat['p99']}",
                "I/O (ms)": f"{brk['audio_io_and_resample']}",
                "Feat (ms)": f"{brk['feature_extraction']}",
                "Inference (ms)": f"{brk['model_forward_pass']}",
                "RTF": f"{bench['real_time_factor']}",
                "Clips / sec": f"{bench['throughput_clips_per_sec']}",
            })

    import pandas as pd
    df_lat = pd.DataFrame(table_rows)
    df_lat.to_markdown(tables_dir / "table_latency.md", index=False)
    df_lat.to_latex(tables_dir / "table_latency.tex", index=False)
    logger.info("Saved latency markdown and latex tables.")

if __name__ == "__main__":
    main()
