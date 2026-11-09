import argparse
import concurrent.futures
import io
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
import pandas as pd
import soundfile as sf
from fastapi.testclient import TestClient

from acforensics.api.main import app
from acforensics.constants import SAMPLE_RATE, TARGET_SAMPLES, DEFAULT_RESULTS_DIR
from acforensics.evaluation.provenance import (
    create_provenance_record,
    guard_results_and_provenance,
)
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def generate_test_audio_bytes(duration_sec: float = 4.0) -> bytes:
    n_samples = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)
    sig = (0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.1 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()


def run_load_test_concurrency(
    client: TestClient,
    audio_bytes: bytes,
    concurrency: int = 5,
    num_requests: int = 200,
) -> dict[str, Any]:
    latencies_ms = []
    successes = 0
    failures = 0
    errors = []

    def single_worker(req_idx: int) -> tuple[bool, float, str]:
        t0 = time.perf_counter()
        try:
            files = {"audio": (f"test_load_{req_idx}.wav", audio_bytes, "audio/wav")}
            resp = client.post("/api/analyze", files=files)
            if resp.status_code != 200:
                el = (time.perf_counter() - t0) * 1000.0
                return False, el, f"POST /api/analyze failed: {resp.status_code} {resp.text}"
            job_id = resp.json().get("job_id")
            if not job_id:
                el = (time.perf_counter() - t0) * 1000.0
                return False, el, "Missing job_id in analyze response"

            res_resp = client.get(f"/api/result/{job_id}")
            if res_resp.status_code != 200:
                el = (time.perf_counter() - t0) * 1000.0
                return False, el, f"GET /api/result failed: {res_resp.status_code} {res_resp.text}"

            res_data = res_resp.json()
            if "verdict" not in res_data or "crnn_score" not in res_data:
                el = (time.perf_counter() - t0) * 1000.0
                return False, el, "Missing verdict or score in analysis result"

            elapsed = (time.perf_counter() - t0) * 1000.0
            return True, elapsed, ""
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return False, elapsed, str(exc)

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_worker, i) for i in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            ok, el, err_msg = f.result()
            latencies_ms.append(el)
            if ok:
                successes += 1
            else:
                failures += 1
                if len(errors) < 10:
                    errors.append(err_msg)

    total_wall_sec = time.perf_counter() - t_start
    arr = np.array(latencies_ms, dtype=np.float64)
    rps = float(num_requests / total_wall_sec) if total_wall_sec > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": num_requests,
        "successes": successes,
        "failures": failures,
        "success_rate_pct": float(successes / max(1, num_requests) * 100.0),
        "wall_time_sec": round(float(total_wall_sec), 3),
        "rps": round(rps, 2),
        "latency_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(arr, 99)), 2),
        "latency_mean_ms": round(float(np.mean(arr)), 2),
        "latency_std_ms": round(float(np.std(arr)), 2),
        "error_sample": errors,
    }


def run_full_load_test(
    manifest_path: str = "data/manifests/eval.parquet",
    out_json: str = "results/metrics/api_load_test.json",
    tables_dir: str = "results/tables",
    concurrency_levels: tuple[int, ...] = (1, 5, 10, 20),
    num_requests_per_level: int = 200,
    clip_duration_sec: float = 4.0,
    seed: int = 42,
) -> dict[str, Any]:
    print(f"Generating {clip_duration_sec}s test audio payload for end-to-end API load test...")
    audio_bytes = generate_test_audio_bytes(duration_sec=clip_duration_sec)

    client = TestClient(app)

    print("Warming up API endpoints...")
    resp_h = client.get("/api/health")
    assert resp_h.status_code == 200, f"Health check failed: {resp_h.text}"

    results_by_concurrency = {}
    table_rows = []

    for c in concurrency_levels:
        print(f"Running load test: Concurrency = {c}, Total Requests = {num_requests_per_level} (End-to-End: Submit + Poll Result)...")
        res = run_load_test_concurrency(
            client=client,
            audio_bytes=audio_bytes,
            concurrency=c,
            num_requests=num_requests_per_level,
        )
        results_by_concurrency[f"concurrency_{c}"] = res
        print(f"  -> Concurrency {c:2d}: Success Rate = {res['success_rate_pct']:.1f}%, RPS = {res['rps']:.1f}, p50 = {res['latency_p50_ms']:.1f}ms, p95 = {res['latency_p95_ms']:.1f}ms, p99 = {res['latency_p99_ms']:.1f}ms")
        table_rows.append({
            "Concurrency": c,
            "Requests": num_requests_per_level,
            "Success Rate (%)": f"{res['success_rate_pct']:.1f}%",
            "Throughput (RPS)": f"{res['rps']:.1f}",
            "Latency p50 (ms)": f"{res['latency_p50_ms']:.1f}",
            "Latency p95 (ms)": f"{res['latency_p95_ms']:.1f}",
            "Latency p99 (ms)": f"{res['latency_p99_ms']:.1f}",
            "Mean Latency (ms)": f"{res['latency_mean_ms']:.1f}",
        })

    prov = create_provenance_record("fastapi_service_load_test", manifest_path, seed=seed)
    payload = {
        **prov,
        "endpoint": "/api/analyze + /api/result/{job_id}",
        "audio_duration_sec": clip_duration_sec,
        "requests_per_level": num_requests_per_level,
        "total_requests_all_levels": num_requests_per_level * len(concurrency_levels),
        "results_by_concurrency": results_by_concurrency,
    }

    guard_results_and_provenance(payload)

    p_out = Path(out_json)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    p_tab = Path(tables_dir)
    p_tab.mkdir(parents=True, exist_ok=True)
    df_tab = pd.DataFrame(table_rows)
    for p_dst in [p_tab / "table_api_load_test.md", p_tab / "table11_api_load_test.md"]:
        df_tab.to_markdown(p_dst, index=False)
    for p_dst in [p_tab / "table_api_load_test.tex", p_tab / "table11_api_load_test.tex"]:
        df_tab.to_latex(p_dst, index=False)

    print(f"API load test successfully finished and saved to {p_out}!")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End API Load Test")
    parser.add_argument("--manifest", type=str, default="data/manifests/eval.parquet")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_full_load_test(
        manifest_path=args.manifest,
        num_requests_per_level=args.requests,
        clip_duration_sec=args.duration,
        seed=args.seed,
    )
