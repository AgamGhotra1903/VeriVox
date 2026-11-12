

export interface HealthResponse {
  status: string;
  version: string;
  device: string;
  models_loaded: string[];
  memory_rss_mb: number;
}

export interface SampleItem {
  id: string;
  name: string;
  filename: string;
  ground_truth: 'bonafide' | 'spoof';
  description: string;
  duration_sec: number;
  filesize_bytes: number;
}

export interface AudioQualityProfile {
  duration_sec: number;
  sample_rate: number;
  channels: number;
  snr_db: number;
  clipping_ratio: number;
  bandwidth_hz: number;
  silence_ratio: number;
  is_valid: boolean;
  warning?: string | null;
}

export interface ModelScoreItem {
  name: string;
  score: number;
  ensemble_weight: number;
  contribution: number;
  verdict: 'BONA FIDE (GENUINE)' | 'SPOOF (SYNTHETIC)' | 'INCONCLUSIVE';
  description: string;
}

export interface AnalysisResult {
  job_id: string;
  verdict: 'BONA FIDE (GENUINE)' | 'SPOOF (SYNTHETIC)' | 'INCONCLUSIVE';
  probability_bonafide: number;
  confidence_band: 'high_confidence_bonafide' | 'high_confidence_spoof' | 'uncertainty_band';
  confidence_note: string;
  fused_score: number;
  baseline_mean_score: number;
  decision_threshold: number;
  explanation: string;
  caveat: string;
  models: ModelScoreItem[];
  quality: AudioQualityProfile;
  waveform_peaks: number[];
  attention_weights: number[];
  lfcc_downsampled: number[][];
  logmel_downsampled: number[][];
  time_axis_sec: number[];
  latency_ms: number;
  created_at: string;
}

export interface PipelineEvent {
  job_id: string;
  stage: string;
  elapsed_ms: number;
  detail?: string | null;
  model?: string | null;
  score?: number | null;
  result_id?: string | null;
  error?: string | null;
}

export interface AnalyzeResponse {
  job_id: string;
  status: string;
  filename: string;
  stream_url: string;
  result_url: string;
}

export interface StressTestItem {
  codec: string;
  bitrate_kbps: number;
  verdict: 'BONA FIDE (GENUINE)' | 'SPOOF (SYNTHETIC)' | 'INCONCLUSIVE';
  probability_bonafide: number;
  fused_score: number;
  score_shift: number;
  bandwidth_hz: number;
  status: string;
}

export interface StressTestResponse {
  job_id: string;
  source_verdict: string;
  source_probability: number;
  source_score: number;
  items: StressTestItem[];
}

export interface BenchmarkTableRow {
  system: string;
  model_type: string;
  dev_eer?: number | null;
  eval_eer?: number | null;
  eval_auc?: number | null;
  eval_min_tdcf?: number | null;
  applied_threshold?: string | null;
}

export interface BenchmarkAblationRow {
  variant: string;
  description: string;
  dev_eer?: number | null;
  eval_eer?: number | null;
  eval_auc?: number | null;
  eval_min_tdcf?: number | null;
}

export interface AttackEERRow {
  attack_id: string;
  taxonomy: string;
  spoof_trials: number;
  attack_eer: number;
  roc_auc: number;
}

export interface CodecRobustnessRow {
  codec: string;
  bitrate_kbps: number;
  clean_eer: number;
  compressed_eer: number;
  degradation_pts: number;
}

export interface CrossDomainRow {
  model_system: string;
  domain: string;
  evaluated_clips: number;
  optimal_eer: number;
  roc_auc: number;
  operating_threshold: string;
  accuracy_at_dev_threshold: number;
}

export interface BenchmarksResponse {
  status: string;
  master_table: BenchmarkTableRow[];
  meta_learner_weights: Array<{ "Base Model": string; "Meta-Learner Weight": number; "Absolute Contribution": number }>;
  feature_ablation: BenchmarkAblationRow[];
  architecture_ablation: BenchmarkAblationRow[];
  regularization_ablation: BenchmarkAblationRow[];
  per_attack_eer: AttackEERRow[];
  codec_robustness: CodecRobustnessRow[];
  crossdomain_itw: CrossDomainRow[];
  roc_curves: Record<string, Array<{ fpr: number; tpr: number }>>;
  silence_shortcut_audit: Record<string, any>;
  last_evaluated_at: string;
}

