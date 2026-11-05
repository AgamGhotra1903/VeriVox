"""Pydantic data schemas for the Acoustic Forensics FastAPI service.

Author: Agam Ghotra (IIT2024226)
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health and operational state."""
    status: str = "healthy"
    version: str = "2.4.0"
    device: str
    models_loaded: list[str]
    memory_rss_mb: float


class SampleItem(BaseModel):
    """Reference audio sample available for instant analysis."""
    id: str
    name: str
    filename: str
    ground_truth: Literal["bonafide", "spoof"]
    description: str
    duration_sec: float
    filesize_bytes: int


class AudioQualityProfile(BaseModel):
    """Deterministic signal quality diagnostics."""
    duration_sec: float
    sample_rate: int
    channels: int
    snr_db: float
    clipping_ratio: float
    bandwidth_hz: float
    silence_ratio: float
    is_valid: bool
    warning: Optional[str] = None


class ModelScoreItem(BaseModel):
    """Individual model contribution within the stacked ensemble."""
    name: str
    score: float
    ensemble_weight: float
    contribution: float
    verdict: Literal["BONA FIDE (GENUINE)", "SPOOF (SYNTHETIC)", "INCONCLUSIVE"]
    description: str


class AnalysisResult(BaseModel):
    """Full forensic analysis report."""
    job_id: str
    verdict: Literal["BONA FIDE (GENUINE)", "SPOOF (SYNTHETIC)", "INCONCLUSIVE"]
    probability_bonafide: float
    confidence_band: Literal["high_confidence_bonafide", "high_confidence_spoof", "uncertainty_band"]
    confidence_note: str
    fused_score: float
    baseline_mean_score: float
    decision_threshold: float
    explanation: str
    caveat: str
    models: list[ModelScoreItem]
    quality: AudioQualityProfile
    waveform_peaks: list[float]
    attention_weights: list[float]
    lfcc_downsampled: list[list[float]]
    logmel_downsampled: list[list[float]]
    time_axis_sec: list[float]
    latency_ms: float
    created_at: str


class PipelineEvent(BaseModel):
    """Server-Sent Event for real-time analysis pipeline tracking."""
    job_id: str
    stage: str
    elapsed_ms: float
    detail: Optional[str] = None
    model: Optional[str] = None
    score: Optional[float] = None
    result_id: Optional[str] = None
    error: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """Response returned upon submitting audio for analysis."""
    job_id: str
    status: str = "processing"
    filename: str
    stream_url: str
    result_url: str


class StressTestItem(BaseModel):
    """Single codec/bitrate re-encoding outcome."""
    codec: str
    bitrate_kbps: int
    verdict: Literal["BONA FIDE (GENUINE)", "SPOOF (SYNTHETIC)", "INCONCLUSIVE"]
    probability_bonafide: float
    fused_score: float
    score_shift: float
    bandwidth_hz: float
    status: str = "ok"


class StressTestResponse(BaseModel):
    """Matrix of codec compression test outcomes."""
    job_id: str
    source_verdict: str
    source_probability: float
    source_score: float
    items: list[StressTestItem]


class BenchmarkTableRow(BaseModel):
    """Row from benchmark tables."""
    system: str
    model_type: str
    dev_eer: Optional[float] = None
    eval_eer: Optional[float] = None
    eval_auc: Optional[float] = None
    eval_min_tdcf: Optional[float] = None
    applied_threshold: Optional[str] = None


class BenchmarkAblationRow(BaseModel):
    """Row from feature, architecture, or regularization ablations."""
    variant: str
    description: str
    dev_eer: Optional[float] = None
    eval_eer: Optional[float] = None
    eval_auc: Optional[float] = None
    eval_min_tdcf: Optional[float] = None


class AttackEERRow(BaseModel):
    """Per-attack unseen technology evaluation."""
    attack_id: str
    taxonomy: str
    spoof_trials: int
    attack_eer: float
    roc_auc: float


class CodecRobustnessRow(BaseModel):
    """Codec robustness degradation benchmark."""
    codec: str
    bitrate_kbps: int
    clean_eer: float
    compressed_eer: float
    degradation_pts: float


class CrossDomainRow(BaseModel):
    """Cross-domain zero-shot evaluation on In-the-Wild."""
    model_system: str
    domain: str
    evaluated_clips: int
    optimal_eer: float
    roc_auc: float
    operating_threshold: str
    accuracy_at_dev_threshold: float


class BenchmarksResponse(BaseModel):
    """Dynamic response containing measured experimental results."""
    status: str = "available"
    master_table: list[BenchmarkTableRow]
    meta_learner_weights: list[dict[str, Any]]
    feature_ablation: list[BenchmarkAblationRow]
    architecture_ablation: list[BenchmarkAblationRow]
    regularization_ablation: list[BenchmarkAblationRow]
    per_attack_eer: list[AttackEERRow]
    codec_robustness: list[CodecRobustnessRow]
    crossdomain_itw: list[CrossDomainRow]
    roc_curves: dict[str, list[dict[str, float]]]
    silence_shortcut_audit: dict[str, Any]
    last_evaluated_at: str

