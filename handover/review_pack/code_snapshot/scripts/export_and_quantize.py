from pathlib import Path
from typing import Any
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import onnx
import onnxruntime as ort

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    DEFAULT_RESULTS_DIR,
    LFCC_DEFAULT_DIM,
)
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.normalization import FeatureNormalizer
from acforensics.models.crnn import CRNNForensics
from acforensics.evaluation.metrics import compute_eer, compute_roc_auc
from acforensics.evaluation.provenance import create_provenance_record
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)

def benchmark_model_latency(
    runner_fn,
    dummy_input: Any,
    num_warmup: int = 50,
    num_timed: int = 500,
    clip_dur_sec: float = 4.0,
) -> dict[str, float]:
    for _ in range(num_warmup):
        _ = runner_fn(dummy_input)

    latencies_ms = []
    for _ in range(num_timed):
        t0 = time.perf_counter()
        _ = runner_fn(dummy_input)
        t_el = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(t_el)

    arr = np.array(latencies_ms, dtype=np.float64)
    mean_lat = float(np.mean(arr))
    rtf = float((mean_lat / 1000.0) / clip_dur_sec)
    thr = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": mean_lat,
        "std_ms": float(np.std(arr)),
        "rtf": rtf,
        "throughput_fps": thr,
    }

def main():
    torch.set_num_threads(1)
    if hasattr(torch.backends, "quantized"):
        torch.backends.quantized.engine = "qnnpack"

    results_dir = Path(DEFAULT_RESULTS_DIR)
    metrics_dir = results_dir / "metrics"
    tables_dir = results_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    eval_parquet = Path("data/manifests/eval.parquet")
    prov = create_provenance_record("model_export_and_quantization", eval_parquet, seed=42)

    logger.info("Loading baseline FP32 CRNN-LFCC model...")
    ckpt_path = Path("checkpoints/crnn_lfcc_best.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/crnn_best.pt")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    model_fp32 = CRNNForensics(in_channels=1, in_freq_dim=LFCC_DEFAULT_DIM, features="lfcc")
    model_fp32.load_state_dict(ckpt["model_state_dict"])
    model_fp32.eval()

    dummy_input = torch.randn(1, 1, 60, 401, dtype=torch.float32)

    logger.info("Exporting TorchScript...")
    ts_path = Path("checkpoints/crnn_lfcc.torchscript.pt")
    traced_model = torch.jit.trace(model_fp32, dummy_input, check_trace=False)
    torch.jit.save(traced_model, str(ts_path))

    logger.info("Exporting ONNX...")
    onnx_path = Path("checkpoints/crnn_lfcc.onnx")
    torch.onnx.export(
        model_fp32,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits", "features"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}},
        opset_version=14,
        dynamo=False,
    )
    ort_sess = ort.InferenceSession(str(onnx_path))

    logger.info("Applying Dynamic INT8 Quantization...")
    int8_path = Path("checkpoints/crnn_lfcc_int8.pt")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    torch.save(quantized_model.state_dict(), str(int8_path))

    logger.info("Benchmarking execution latency across model formats (pinned 1 thread)...")
    lat_fp32 = benchmark_model_latency(lambda x: model_fp32(x)[0], dummy_input)
    lat_ts = benchmark_model_latency(lambda x: traced_model(x)[0], dummy_input)
    lat_onnx = benchmark_model_latency(lambda x: ort_sess.run(None, {"input": x.numpy()})[0], dummy_input)
    lat_int8 = benchmark_model_latency(lambda x: quantized_model(x)[0], dummy_input)

    logger.info("Evaluating Accuracy and Score Fidelity on evaluation set...")
    eval_df = pd.read_parquet(eval_parquet)
    rng = np.random.RandomState(42)
    bona_pool = eval_df[eval_df["label"] == BONAFIDE]
    spoof_pool = eval_df[eval_df["label"] == SPOOF]
    idx_bona = rng.choice(bona_pool.index, size=500, replace=False)
    idx_spoof = rng.choice(spoof_pool.index, size=700, replace=False)
    sub_df = eval_df.loc[np.concatenate([idx_bona, idx_spoof])].copy()

    extractor = LFCCExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    norm_file = Path("checkpoints/normalization_stats_lfcc.json")
    if not norm_file.exists():
        norm_file = Path("checkpoints/normalization_stats.json")
    normalizer = FeatureNormalizer().load(norm_file) if norm_file.exists() else FeatureNormalizer()

    feats_list = []
    labels_list = []
    for _, row in sub_df.iterrows():
        res = load_and_validate_audio(row["path"], target_sr=SAMPLE_RATE)
        if res.is_valid and res.waveform is not None:
            cw = fixed_length_crop_or_pad(res.waveform, target_samples=TARGET_SAMPLES, mode="center")
            f = extractor(cw)
            if normalizer.is_fitted:
                f = normalizer(f)
            feats_list.append(f)
            labels_list.append(row["label"])

    batch_feats = torch.stack(feats_list).unsqueeze(1)
    eval_labels = np.array(labels_list, dtype=np.int64)

    with torch.no_grad():
        out_fp32, _ = model_fp32(batch_feats)
        scores_fp32 = (out_fp32[:, BONAFIDE] - out_fp32[:, SPOOF]).numpy()

        out_ts, _ = traced_model(batch_feats)
        scores_ts = (out_ts[:, BONAFIDE] - out_ts[:, SPOOF]).numpy()

        out_onnx = ort_sess.run(None, {"input": batch_feats.numpy()})[0]
        scores_onnx = out_onnx[:, BONAFIDE] - out_onnx[:, SPOOF]

        out_int8, _ = quantized_model(batch_feats)
        scores_int8 = (out_int8[:, BONAFIDE] - out_int8[:, SPOOF]).numpy()

    eer_fp32, _ = compute_eer(eval_labels, scores_fp32)
    auc_fp32 = compute_roc_auc(eval_labels, scores_fp32)

    eer_ts, _ = compute_eer(eval_labels, scores_ts)
    auc_ts = compute_roc_auc(eval_labels, scores_ts)

    eer_onnx, _ = compute_eer(eval_labels, scores_onnx)
    auc_onnx = compute_roc_auc(eval_labels, scores_onnx)

    eer_int8, _ = compute_eer(eval_labels, scores_int8)
    auc_int8 = compute_roc_auc(eval_labels, scores_int8)

    sz_fp32 = ckpt_path.stat().st_size / (1024 * 1024)
    sz_ts = ts_path.stat().st_size / (1024 * 1024)
    sz_onnx = onnx_path.stat().st_size / (1024 * 1024)
    sz_int8 = int8_path.stat().st_size / (1024 * 1024)

    diff_ts = float(np.max(np.abs(scores_fp32 - scores_ts)))
    diff_onnx = float(np.max(np.abs(scores_fp32 - scores_onnx)))
    diff_int8 = float(np.max(np.abs(scores_fp32 - scores_int8)))

    formats = {
        "PyTorch FP32": {
            "file_size_mb": sz_fp32,
            "compression_ratio": 1.0,
            "eer_pct": float(eer_fp32),
            "eer_delta_pts": 0.0,
            "roc_auc": float(auc_fp32),
            "max_score_diff": 0.0,
            "latency": lat_fp32,
        },
        "TorchScript JIT": {
            "file_size_mb": sz_ts,
            "compression_ratio": sz_fp32 / sz_ts if sz_ts > 0 else 1.0,
            "eer_pct": float(eer_ts),
            "eer_delta_pts": float(eer_ts - eer_fp32),
            "roc_auc": float(auc_ts),
            "max_score_diff": diff_ts,
            "latency": lat_ts,
        },
        "ONNX Runtime": {
            "file_size_mb": sz_onnx,
            "compression_ratio": sz_fp32 / sz_onnx if sz_onnx > 0 else 1.0,
            "eer_pct": float(eer_onnx),
            "eer_delta_pts": float(eer_onnx - eer_fp32),
            "roc_auc": float(auc_onnx),
            "max_score_diff": diff_onnx,
            "latency": lat_onnx,
        },
        "PyTorch Dynamic INT8": {
            "file_size_mb": sz_int8,
            "compression_ratio": sz_fp32 / sz_int8 if sz_int8 > 0 else 1.0,
            "eer_pct": float(eer_int8),
            "eer_delta_pts": float(eer_int8 - eer_fp32),
            "roc_auc": float(auc_int8),
            "max_score_diff": diff_int8,
            "latency": lat_int8,
        },
    }

    summary = {
        **prov,
        "sample_size": len(labels_list),
        "thread_count": 1,
        "quantization_engine": "qnnpack",
        "formats": formats,
    }

    out_json = metrics_dir / "quantization_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for fmt_name, d in formats.items():
        rows.append({
            "Format": fmt_name,
            "Size (MB)": f"{d['file_size_mb']:.2f}",
            "Compression": f"{d['compression_ratio']:.2f}x",
            "Eval EER (%)": f"{d['eer_pct']:.2f}",
            "EER Delta (pts)": f"{d['eer_delta_pts']:+.2f}",
            "ROC-AUC": f"{d['roc_auc']:.4f}",
            "Max Score Diff": f"{d['max_score_diff']:.2e}",
            "Latency p50 (ms)": f"{d['latency']['p50_ms']:.2f}",
            "Latency p95 (ms)": f"{d['latency']['p95_ms']:.2f}",
            "RTF": f"{d['latency']['rtf']:.4f}",
            "Throughput (fps)": f"{d['latency']['throughput_fps']:.1f}",
        })

    df_out = pd.DataFrame(rows)
    for p in [tables_dir / "table_quantization.md", tables_dir / "table10_quantization_and_export.md"]:
        df_out.to_markdown(p, index=False)
    for p in [tables_dir / "table_quantization.tex", tables_dir / "table10_quantization_and_export.tex"]:
        df_out.to_latex(p, index=False)

    logger.info("Export and Quantization Benchmark successfully completed! Output: %s", out_json)

if __name__ == "__main__":
    main()
