import argparse
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from acforensics.constants import (
    SAMPLE_RATE,
    TARGET_SAMPLES,
    DEFAULT_RESULTS_DIR,
    BONAFIDE,
    SPOOF,
)
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.ensemble.predictor import EnsemblePredictor
from acforensics.evaluation.provenance import (
    create_provenance_record,
    guard_results_and_provenance,
)
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics


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


def benchmark_single_crnn(
    device_name: str,
    device: torch.device,
    model: torch.nn.Module,
    extractor: LFCCExtractor,
    audio_path: Path,
    num_warmup: int = 50,
    num_timed: int = 500,
    threads: int = 1,
) -> dict[str, Any]:
    if device.type == "cpu":
        torch.set_num_threads(threads)

    for _ in range(num_warmup):
        res = load_and_validate_audio(audio_path, target_sr=SAMPLE_RATE)
        wave = fixed_length_crop_or_pad(res.waveform, target_samples=TARGET_SAMPLES, mode="center")
        feat = extractor(wave.to(device=device, dtype=torch.float32).unsqueeze(0))
        norm_f = per_utterance_cmvn(feat).unsqueeze(1)
        with torch.no_grad():
            logits, _ = model(norm_f)
            if device.type == "mps":
                torch.mps.synchronize()
            _ = float((logits[:, BONAFIDE] - logits[:, SPOOF]).item())

    times_total = []
    times_io = []
    times_feat = []
    times_fwd = []

    for _ in range(num_timed):
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        res = load_and_validate_audio(audio_path, target_sr=SAMPLE_RATE)
        wave = fixed_length_crop_or_pad(res.waveform, target_samples=TARGET_SAMPLES, mode="center")
        t_io = time.perf_counter() - t0

        t1 = time.perf_counter()
        feat = extractor(wave.to(device=device, dtype=torch.float32).unsqueeze(0))
        norm_f = per_utterance_cmvn(feat).unsqueeze(1)
        t_feat = time.perf_counter() - t1

        t2 = time.perf_counter()
        with torch.no_grad():
            logits, _ = model(norm_f)
            if device.type == "mps":
                torch.mps.synchronize()
            _ = float((logits[:, BONAFIDE] - logits[:, SPOOF]).item())
        t_fwd = time.perf_counter() - t2

        t_end = time.perf_counter()

        times_total.append((t_end - t_start) * 1000.0)
        times_io.append(t_io * 1000.0)
        times_feat.append(t_feat * 1000.0)
        times_fwd.append(t_fwd * 1000.0)

    arr = np.array(times_total, dtype=np.float64)
    clip_dur_s = float(TARGET_SAMPLES / SAMPLE_RATE)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    rtf = float((mean_v / 1000.0) / clip_dur_s)
    throughput = float(1000.0 / mean_v) if mean_v > 0 else 0.0

    return {
        "pipeline": "Primary CRNN-LFCC Full Path",
        "device": device_name,
        "threads": threads if device.type == "cpu" else "N/A",
        "num_warmup": num_warmup,
        "num_timed_runs": num_timed,
        "clip_duration_sec": clip_dur_s,
        "latency_ms": {
            "mean": round(mean_v, 2),
            "std": round(std_v, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
        },
        "breakdown_ms": {
            "decode_and_resample": round(float(np.mean(times_io)), 2),
            "lfcc_extraction": round(float(np.mean(times_feat)), 2),
            "neural_forward_pass": round(float(np.mean(times_fwd)), 2),
        },
        "real_time_factor": round(rtf, 5),
        "throughput_clips_per_sec": round(throughput, 1),
    }


def benchmark_ensemble_full_path(
    device_name: str,
    device: torch.device,
    ensemble_predictor: EnsemblePredictor,
    audio_path: Path,
    num_warmup: int = 20,
    num_timed: int = 200,
    threads: int = 1,
) -> dict[str, Any]:
    if device.type == "cpu":
        torch.set_num_threads(threads)

    for _ in range(num_warmup):
        _ = ensemble_predictor.predict_file(audio_path)
        if device.type == "mps":
            torch.mps.synchronize()

    times_total = []
    for _ in range(num_timed):
        t0 = time.perf_counter()
        _ = ensemble_predictor.predict_file(audio_path)
        if device.type == "mps":
            torch.mps.synchronize()
        times_total.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_total, dtype=np.float64)
    clip_dur_s = float(TARGET_SAMPLES / SAMPLE_RATE)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    rtf = float((mean_v / 1000.0) / clip_dur_s)
    throughput = float(1000.0 / mean_v) if mean_v > 0 else 0.0

    return {
        "pipeline": "Full Shipped Ensemble Path (LFCC+LogMel+CRNN+Classical+MetaLearner)",
        "device": device_name,
        "threads": threads if device.type == "cpu" else "N/A",
        "num_warmup": num_warmup,
        "num_timed_runs": num_timed,
        "clip_duration_sec": clip_dur_s,
        "latency_ms": {
            "mean": round(mean_v, 2),
            "std": round(std_v, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
        },
        "real_time_factor": round(rtf, 5),
        "throughput_clips_per_sec": round(throughput, 1),
    }


def run_latency_benchmark(
    manifest_path: str = "data/manifests/eval.parquet",
    checkpoint_path: str = "checkpoints/crnn_lfcc_best.pt",
    out_json: str = "results/metrics/latency.json",
    tables_dir: str = "results/tables",
    seed: int = 42,
) -> dict[str, Any]:
    print("Initializing Latency & Hardware Throughput Benchmark...")

    temp_audio = Path("data/temp_latency_test.wav")
    rng = np.random.RandomState(seed)
    tone = (0.3 * rng.randn(TARGET_SAMPLES)).astype(np.float32)
    sf.write(str(temp_audio), tone, SAMPLE_RATE)

    cpu_device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=cpu_device, weights_only=False)
    model_cpu = CRNNForensics(in_channels=1, in_freq_dim=60, features="lfcc")
    model_cpu.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model_cpu.eval()

    extractor = LFCCExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    ensemble_cpu = EnsemblePredictor(device=cpu_device)

    cpu_brand = get_cpu_brand()
    try:
        res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
        total_ram_gb = round(int(res.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        total_ram_gb = 24.0

    print("Benchmarking CRNN-LFCC on CPU (1 thread, 500 timed runs)...")
    res_crnn_cpu1 = benchmark_single_crnn(
        device_name=f"CPU ({cpu_brand}, 1 thread)",
        device=cpu_device,
        model=model_cpu,
        extractor=extractor,
        audio_path=temp_audio,
        num_warmup=50,
        num_timed=500,
        threads=1,
    )

    print("Benchmarking CRNN-LFCC on CPU (4 threads, 500 timed runs)...")
    res_crnn_cpu4 = benchmark_single_crnn(
        device_name=f"CPU ({cpu_brand}, 4 threads)",
        device=cpu_device,
        model=model_cpu,
        extractor=extractor,
        audio_path=temp_audio,
        num_warmup=50,
        num_timed=500,
        threads=4,
    )

    res_crnn_mps = None
    res_ens_mps = None
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print("Benchmarking CRNN-LFCC on Apple Silicon GPU (MPS, 500 timed runs)...")
        model_mps = CRNNForensics(in_channels=1, in_freq_dim=60, features="lfcc")
        model_mps.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        model_mps.to(mps_device)
        model_mps.eval()
        extractor_mps = LFCCExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True).to(mps_device)

        res_crnn_mps = benchmark_single_crnn(
            device_name="Apple Silicon GPU (MPS)",
            device=mps_device,
            model=model_mps,
            extractor=extractor_mps,
            audio_path=temp_audio,
            num_warmup=50,
            num_timed=500,
        )

        print("Benchmarking Full Shipped Ensemble on Apple Silicon GPU (MPS, 200 timed runs)...")
        ensemble_mps = EnsemblePredictor(device=mps_device)
        res_ens_mps = benchmark_ensemble_full_path(
            device_name="Apple Silicon GPU (MPS)",
            device=mps_device,
            ensemble_predictor=ensemble_mps,
            audio_path=temp_audio,
            num_warmup=20,
            num_timed=200,
        )

    print("Benchmarking Full Shipped Ensemble on CPU (4 threads, 200 timed runs)...")
    ensemble_cpu = EnsemblePredictor(device=cpu_device)
    res_ens_cpu4 = benchmark_ensemble_full_path(
        device_name=f"CPU ({cpu_brand}, 4 threads)",
        device=cpu_device,
        ensemble_predictor=ensemble_cpu,
        audio_path=temp_audio,
        num_warmup=20,
        num_timed=200,
        threads=4,
    )

    temp_audio.unlink(missing_ok=True)

    prov = create_provenance_record("latency_benchmark", manifest_path, seed=seed, checkpoint_path=checkpoint_path)

    payload = {
        **prov,
        "hardware_environment": {
            "cpu_model": cpu_brand,
            "total_ram_gb": total_ram_gb,
            "os": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "benchmarks": {
            "crnn_cpu_single_thread": res_crnn_cpu1,
            "crnn_cpu_multi_thread": res_crnn_cpu4,
            "crnn_apple_silicon_mps": res_crnn_mps,
            "ensemble_cpu_multi_thread": res_ens_cpu4,
            "ensemble_apple_silicon_mps": res_ens_mps,
        },
    }

    guard_results_and_provenance(payload)

    p_out = Path(out_json)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    p_tab = Path(tables_dir)
    p_tab.mkdir(parents=True, exist_ok=True)

    table_rows = []
    for bench_key, bench in payload["benchmarks"].items():
        if bench is not None:
            lat = bench["latency_ms"]
            brk = bench.get("breakdown_ms", {})
            io_s = f"{brk.get('decode_and_resample', '-')}"
            ft_s = f"{brk.get('lfcc_extraction', '-')}"
            nn_s = f"{brk.get('neural_forward_pass', '-')}"
            table_rows.append({
                "Pipeline": bench["pipeline"],
                "Execution Target": bench["device"],
                "Threads": bench["threads"],
                "Mean ± Std (ms)": f"{lat['mean']} ± {lat['std']}",
                "p50 (ms)": f"{lat['p50']}",
                "p95 (ms)": f"{lat['p95']}",
                "p99 (ms)": f"{lat['p99']}",
                "I/O (ms)": io_s,
                "Feat (ms)": ft_s,
                "Inference (ms)": nn_s,
                "RTF": f"{bench['real_time_factor']}",
                "Clips / sec": f"{bench['throughput_clips_per_sec']}",
            })

    df_lat = pd.DataFrame(table_rows)
    df_lat.to_markdown(p_tab / "table_latency.md", index=False)
    df_lat.to_latex(p_tab / "table_latency.tex", index=False)
    print("Saved Table Latency markdown and latex tables.")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Shipped Path Latency Benchmark")
    parser.add_argument("--manifest", type=str, default="data/manifests/eval.parquet")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crnn_lfcc_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_latency_benchmark(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
    )
