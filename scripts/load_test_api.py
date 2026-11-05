from pathlib import Path
from typing import Any
import json
import io
import time
import concurrent.futures
import numpy as np
import pandas as pd
import soundfile as sf
from fastapi.testclient import TestClient

from acforensics.api.main import app
from acforensics.constants import SAMPLE_RATE, DEFAULT_RESULTS_DIR
from acforensics.evaluation.provenance import create_provenance_record
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)

def generate_test_audio_bytes(duration_sec: float = 1.0) -> bytes:
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False, dtype=np.float32)
    sig = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()

def run_load_test_concurrency(
    client: TestClient,
    audio_bytes: bytes,
    concurrency: int = 5,
    num_requests: int = 50,
) -> dict[str, Any]:
    latencies_ms = []
    successes = 0
    failures = 0

    def single_worker(req_idx: int) -> tuple[bool, float]:
        t0 = time.perf_counter()
        try:
            files = {"audio": (f"test_load_{req_idx}.wav", audio_bytes, "audio/wav")}
            resp = client.post("/api/analyze", files=files)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return (resp.status_code == 200, elapsed)
        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return (False, elapsed)

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_worker, i) for i in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            ok, el = f.result()
            latencies_ms.append(el)
            if ok:
                successes += 1
            else:
                failures += 1

    total_wall_sec = time.perf_counter() - t_start
    arr = np.array(latencies_ms, dtype=np.float64)
    rps = float(num_requests / total_wall_sec) if total_wall_sec > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": num_requests,
        "successes": successes,
        "failures": failures,
        "success_rate_pct": float(successes / max(1, num_requests) * 100.0),
        "wall_time_sec": float(total_wall_sec),
        "rps": rps,
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_p99_ms": float(np.percentile(arr, 99)),
        "latency_mean_ms": float(np.mean(arr)),
        "latency_std_ms": float(np.std(arr)),
    }

def main():
    results_dir = Path(DEFAULT_RESULTS_DIR)
    metrics_dir = results_dir / "metrics"
    tables_dir = results_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    eval_parquet = Path("data/manifests/eval.parquet")
    prov = create_provenance_record("fastapi_service_load_test", eval_parquet, seed=42)

    logger.info("Generating test audio payload...")
    audio_payload = generate_test_audio_bytes(duration_sec=1.0)

    test_client = TestClient(app)

    logger.info("Warming up FastAPI endpoints...")
    resp_health = test_client.get("/api/health")
    assert resp_health.status_code == 200, f"Health check failed: {resp_health.text}"

    resp_samples = test_client.get("/api/samples")
    assert resp_samples.status_code == 200

    resp_bench = test_client.get("/api/benchmarks")
    assert resp_bench.status_code == 200

    concurrency_levels = [1, 5, 10, 20]
    results_by_concurrency = {}
    table_rows = []

    for c in concurrency_levels:
        n_req = 40 if c > 1 else 20
        logger.info("Executing load test: concurrency=%d, total_requests=%d...", c, n_req)
        m = run_load_test_concurrency(test_client, audio_payload, concurrency=c, num_requests=n_req)
        results_by_concurrency[f"concurrency_{c}"] = m
        table_rows.append({
            "Concurrency": c,
            "Requests": n_req,
            "Success Rate (%)": f"{m['success_rate_pct']:.1f}%",
            "Throughput (RPS)": f"{m['rps']:.1f}",
            "Latency p50 (ms)": f"{m['latency_p50_ms']:.1f}",
            "Latency p95 (ms)": f"{m['latency_p95_ms']:.1f}",
            "Latency p99 (ms)": f"{m['latency_p99_ms']:.1f}",
            "Mean Latency (ms)": f"{m['latency_mean_ms']:.1f}",
        })

    summary = {
        **prov,
        "endpoint": "/api/analyze",
        "audio_duration_sec": 1.0,
        "results_by_concurrency": results_by_concurrency,
    }

    out_json = metrics_dir / "api_load_test.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    df_table = pd.DataFrame(table_rows)
    for p in [tables_dir / "table_api_load_test.md", tables_dir / "table11_api_load_test.md"]:
        df_table.to_markdown(p, index=False)
    for p in [tables_dir / "table_api_load_test.tex", tables_dir / "table11_api_load_test.tex"]:
        df_table.to_latex(p, index=False)

    logger.info("FastAPI load test successfully completed! Output: %s", out_json)

if __name__ == "__main__":
    main()

