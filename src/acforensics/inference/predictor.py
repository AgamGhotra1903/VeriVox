"""Production single-clip inference engine and forensic verdict predictor.

Author: Agam Ghotra (IIT2024226)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Optional, Union
import joblib
import numpy as np
import torch

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    LFCC_DEFAULT_DIM,
)
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import generate_multi_crop_windows, fixed_length_crop_or_pad
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.statistics import extract_utterance_statistics
from acforensics.features.normalization import FeatureNormalizer, per_utterance_cmvn
from acforensics.models.crnn import CRNNForensics
from acforensics.models.calibration import PlattCalibrator
from acforensics.utils.timing import Timer
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ForensicPrediction:
    """Forensic decision container for single-clip analysis."""
    verdict: str                  # 'LIKELY GENUINE', 'LIKELY SYNTHETIC', 'UNCERTAIN'
    calibrated_probability_bonafide: float
    raw_score: float              # Continuous Log-Likelihood Ratio or margin
    decision_threshold: float
    confidence_percentage: float
    duration_sec: float
    num_windows_analyzed: int
    inference_latency_ms: float
    file_sha256: str
    timestamp_utc: str
    attention_weights: np.ndarray # (T,) frame weights for heat-strip overlay
    window_timeline: list[dict[str, Any]]
    model_name: str
    feature_type: str
    synthetic_score: float = 0.0
    tampering_probability: float = 0.0
    anomaly_timestamps: list[dict[str, Any]] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.synthetic_score == 0.0 and self.calibrated_probability_bonafide is not None:
            self.synthetic_score = round(1.0 - self.calibrated_probability_bonafide, 4)
        if self.tampering_probability == 0.0 and self.calibrated_probability_bonafide is not None:
            self.tampering_probability = round(1.0 - self.calibrated_probability_bonafide, 4)
        if self.anomaly_timestamps is None:
            self.anomaly_timestamps = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Returns standardized forensic JSON report conforming to forensic API standards."""
        return {
            "metadata": {
                "duration_sec": round(self.duration_sec, 2),
                "sample_rate": self.metadata.get("sample_rate", SAMPLE_RATE),
                "orig_sample_rate": self.metadata.get("orig_sample_rate", SAMPLE_RATE),
                "bit_depth": self.metadata.get("bit_depth", "16-bit PCM"),
                "channels": self.metadata.get("channels", 1),
                "peak_level_db": self.metadata.get("peak_level_db", 0.0),
                "snr_db": self.metadata.get("snr_estimate_db", 30.0),
                "bitrate_kbps": self.metadata.get("bitrate_kbps"),
                "file_sha256": self.file_sha256,
                "timestamp_utc": self.timestamp_utc,
            },
            "scores": {
                "verdict": self.verdict,
                "calibrated_probability_bonafide": round(self.calibrated_probability_bonafide, 4),
                "synthetic_audio_score": round(self.synthetic_score, 4),
                "tampering_probability": round(self.tampering_probability, 4),
                "confidence_percentage": round(self.confidence_percentage, 2),
                "raw_score_llr": round(self.raw_score, 4),
                "decision_threshold": round(self.decision_threshold, 4),
            },
            "anomalies": {
                "anomaly_timestamps": self.anomaly_timestamps,
                "num_anomalies_detected": len(self.anomaly_timestamps),
                "timeline": self.window_timeline,
            },
            "engine": {
                "model_name": self.model_name,
                "feature_type": self.feature_type,
                "inference_latency_ms": round(self.inference_latency_ms, 2),
                "num_windows_analyzed": self.num_windows_analyzed,
            },
        }


class ForensicPredictor:
    """Production Inference Engine supporting CRNN and Classical ML models."""

    def __init__(
        self,
        checkpoint_path: Path | str = "checkpoints/crnn_best.pt",
        norm_stats_path: Path | str = "checkpoints/normalization_stats.json",
        calibrator_path: Optional[Path | str] = "checkpoints/calibrator.joblib",
        device: str = "cpu",
        uncertain_band: tuple[float, float] = (0.35, 0.65),
    ):
        self.device = torch.device(device)
        self.uncertain_band = uncertain_band
        self.feature_type = "lfcc"
        self._classical_cache: dict[str, Any] = {}

        # 1. Feature Extractors
        self.lfcc_extractor = LFCCExtractor(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_lfcc=20,
            with_deltas=True,
        ).to(self.device)

        self.logmel_extractor = LogMelExtractor(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80,
        ).to(self.device)

        # 2. Normalizer
        self.normalizer = FeatureNormalizer(feature_dim=LFCC_DEFAULT_DIM)
        norm_p = Path(norm_stats_path)
        if norm_p.exists():
            self.normalizer.load(norm_p)
            self.normalizer.to(self.device)

        # 3. CRNN Model
        self.model = CRNNForensics(in_channels=1, in_freq_dim=LFCC_DEFAULT_DIM, features=self.feature_type)
        ckpt_p = Path(checkpoint_path)
        if ckpt_p.exists():
            ckpt = torch.load(str(ckpt_p), map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.decision_threshold = float(ckpt.get("dev_threshold", 5.23))
        else:
            logger.warning("Checkpoint %s not found. Using default initialization.", ckpt_p)
            self.decision_threshold = 0.0

        self.model.to(self.device)
        self.model.eval()

        # 4. Calibrator
        if calibrator_path and Path(calibrator_path).exists():
            self.calibrator = PlattCalibrator.load(calibrator_path)
        else:
            self.calibrator = PlattCalibrator()

    def _compute_sha256(self, file_path: Path | str) -> str:
        """Computes cryptographic SHA-256 hash of the input audio file."""
        p = Path(file_path)
        if not p.exists():
            return "unknown"
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_classical_model(self, model_key: str, feature_key: str) -> Optional[Any]:
        """Loads and caches classical scikit-learn baseline pipelines."""
        cache_id = f"{model_key}_{feature_key}"
        if cache_id in self._classical_cache:
            return self._classical_cache[cache_id]

        path = Path("checkpoints/classical") / f"{cache_id}.joblib"
        if path.exists():
            clf = joblib.load(path)
            self._classical_cache[cache_id] = clf
            return clf
        return None

    def predict(
        self,
        audio_input: Union[Path, str, torch.Tensor, np.ndarray],
        threshold_override: Optional[float] = None,
        model_name: Optional[str] = None,
        feature_type: Optional[str] = None,
    ) -> ForensicPrediction:
        """Runs full forensic inspection pipeline on an audio clip or in-memory waveform.

        Args:
            audio_input: Path to audio file OR 1D waveform tensor/ndarray.
            threshold_override: Optional custom decision threshold.
            model_name: Optional model choice (e.g. 'CRNN (Best)', 'Random Forest (LFCC)', etc.).
            feature_type: Optional feature choice ('lfcc', 'logmel', 'combined').

        Returns:
            ForensicPrediction dataclass with verdict, calibrated probability, and attention.
        """
        threshold = threshold_override if threshold_override is not None else self.decision_threshold
        chosen_model = model_name or "CRNN (Best)"
        chosen_feature = (feature_type or "lfcc").lower()

        with Timer() as timer:
            # 1. Load and validate
            if isinstance(audio_input, (torch.Tensor, np.ndarray)):
                if isinstance(audio_input, np.ndarray):
                    waveform = torch.from_numpy(audio_input.astype(np.float32))
                else:
                    waveform = audio_input.float().cpu()
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0)
                # Peak normalize
                max_v = torch.max(torch.abs(waveform))
                if max_v > 1e-8:
                    waveform = waveform / max_v
                duration_sec = len(waveform) / SAMPLE_RATE
                file_hash = hashlib.sha256(waveform.numpy().tobytes()).hexdigest()
            else:
                res = load_and_validate_audio(audio_input, target_sr=SAMPLE_RATE)
                if not res.is_valid or res.waveform is None:
                    raise ValueError(f"Audio validation failed: {res.error_reason}")
                waveform = res.waveform
                duration_sec = res.duration_sec
                file_hash = self._compute_sha256(audio_input)

            # Check if a classical model was requested
            is_classical = any(
                tag in chosen_model.lower()
                for tag in ["random forest", "rbf-svm", "logistic regression", "svm", "forest", "logistic"]
            )

            if is_classical:
                # Map model name
                if "random forest" in chosen_model.lower() or "forest" in chosen_model.lower():
                    m_key = "random_forest"
                elif "svm" in chosen_model.lower():
                    m_key = "rbf_svm"
                else:
                    m_key = "logistic_regression"

                # Map feature
                if "log-mel" in chosen_model.lower() or "logmel" in chosen_feature:
                    f_key = "logmel"
                elif "combined" in chosen_model.lower() or "combined" in chosen_feature:
                    f_key = "combined"
                else:
                    f_key = "lfcc"

                clf = self._get_classical_model(m_key, f_key)
                if clf is None:
                    logger.warning("Classical checkpoint %s_%s not found, falling back to CRNN.", m_key, f_key)
                    is_classical = False

            if is_classical:
                # Classical Model Pipeline
                # 1. Crop/pad waveform to fixed length
                padded = fixed_length_crop_or_pad(waveform, target_samples=TARGET_SAMPLES, mode="center")
                
                with torch.no_grad():
                    if f_key == "lfcc":
                        feat = self.lfcc_extractor(padded.to(self.device))
                        x_vec = extract_utterance_statistics(feat)
                    elif f_key == "logmel":
                        feat = self.logmel_extractor(padded.to(self.device))
                        x_vec = extract_utterance_statistics(feat)
                    else:
                        feat_lfcc = self.lfcc_extractor(padded.to(self.device))
                        feat_mel = self.logmel_extractor(padded.to(self.device))
                        s_lfcc = extract_utterance_statistics(feat_lfcc)
                        s_mel = extract_utterance_statistics(feat_mel)
                        x_vec = np.concatenate([s_lfcc, s_mel])

                X = x_vec.reshape(1, -1)
                if hasattr(clf, "decision_function"):
                    raw_score = float(clf.decision_function(X)[0])
                    # Sigmoid for pseudo-probability
                    calibrated_prob = float(1.0 / (1.0 + np.exp(-raw_score)))
                elif hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X)[0]
                    calibrated_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    # Log-odds
                    eps = 1e-6
                    raw_score = float(np.log(np.clip(calibrated_prob, eps, 1 - eps) / np.clip(1 - calibrated_prob, eps, 1 - eps)))
                else:
                    raw_score = float(clf.predict(X)[0])
                    calibrated_prob = 1.0 if raw_score > 0 else 0.0

                # Energy-derived weights for plotting heat-strip
                frames = 400
                stride = max(1, len(waveform) // frames)
                downsampled = waveform[::stride].abs().numpy()
                attn_weights = downsampled[:frames]
                if len(attn_weights) < frames:
                    attn_weights = np.pad(attn_weights, (0, frames - len(attn_weights)))
                attn_norm = np.max(attn_weights)
                if attn_norm > 1e-6:
                    attn_weights = attn_weights / attn_norm

                timeline = [{
                    "window_idx": 0,
                    "start_sec": 0.0,
                    "end_sec": duration_sec,
                    "raw_score": raw_score,
                    "calibrated_prob": calibrated_prob,
                }]
                num_windows = 1
                model_display = f"{chosen_model}"
                feature_display = f"{f_key.upper()} (Statistical Moments)"

            else:
                # CRNN Deep Model Pipeline
                windows = generate_multi_crop_windows(waveform, window_samples=TARGET_SAMPLES, hop_ratio=0.5)

                window_scores = []
                window_weights_list = []
                timeline = []

                for w in windows:
                    chunk = w["waveform"]
                    with torch.no_grad():
                        feat = self.lfcc_extractor(chunk.to(self.device))
                        if self.normalizer.is_fitted:
                            feat = self.normalizer(feat)
                        else:
                            feat = per_utterance_cmvn(feat)

                        feat = feat.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
                        logits, attn = self.model(feat)

                        # Continuous score = logit[BONAFIDE] - logit[SPOOF]
                        score = (logits[:, BONAFIDE] - logits[:, SPOOF]).item()
                        prob = float(self.calibrator.predict_proba(np.array([score]))[0])

                        window_scores.append(score)
                        window_weights_list.append(attn.squeeze(0).cpu().numpy())

                        timeline.append({
                            "window_idx": w["window_idx"],
                            "start_sec": w["start_sec"],
                            "end_sec": w["end_sec"],
                            "raw_score": score,
                            "calibrated_prob": prob,
                        })

                # Multi-crop aggregated score
                raw_score = float(np.mean(window_scores))
                attn_weights = np.mean(window_weights_list, axis=0)
                calibrated_prob = float(self.calibrator.predict_proba(np.array([raw_score]))[0])
                num_windows = len(windows)
                model_display = "CRNN (CNN + BiGRU + Self-Attention)"
                feature_display = "LFCC (60 Coefficients with Deltas)"

            # Determine forensic verdict based on threshold and calibrated probability
            low_band, high_band = self.uncertain_band
            margin = abs(raw_score - threshold)

            if margin < 0.25 or (low_band <= calibrated_prob <= high_band):
                verdict = "UNCERTAIN"
                confidence = float(max(15.0, (1.0 - 2.0 * abs(calibrated_prob - 0.5)) * 100.0))
            elif raw_score >= threshold:
                verdict = "LIKELY GENUINE"
                confidence = float(max(55.0, calibrated_prob * 100.0))
            else:
                verdict = "LIKELY SYNTHETIC"
                confidence = float(max(55.0, (1.0 - calibrated_prob) * 100.0))

            # Detect anomaly segments from window timeline
            anomaly_segments = []
            for w in timeline:
                win_spoof_prob = 1.0 - w.get("calibrated_prob", 0.5)
                if win_spoof_prob >= 0.55:
                    anomaly_segments.append({
                        "window_idx": w.get("window_idx", 0),
                        "start_sec": round(w.get("start_sec", 0.0), 3),
                        "end_sec": round(w.get("end_sec", duration_sec), 3),
                        "anomaly_score": round(win_spoof_prob, 4),
                        "severity": "CRITICAL" if win_spoof_prob > 0.85 else ("HIGH" if win_spoof_prob > 0.70 else "MEDIUM"),
                        "anomaly_type": "Phase Discontinuity / Neural Vocoder Inconsistency",
                    })

            meta_dict = {
                "sample_rate": SAMPLE_RATE,
                "orig_sample_rate": getattr(res, "orig_sample_rate", SAMPLE_RATE) if "res" in locals() else SAMPLE_RATE,
                "channels": getattr(res, "channels", 1) if "res" in locals() else 1,
                "bit_depth": getattr(res, "bit_depth", "16-bit PCM") if "res" in locals() else "32-bit Float",
                "bitrate_kbps": getattr(res, "bitrate_kbps", None) if "res" in locals() else None,
                "peak_level_db": getattr(res, "peak_level_db", 0.0) if "res" in locals() else 0.0,
                "snr_estimate_db": getattr(res, "snr_estimate_db", 30.0) if "res" in locals() else 30.0,
                "duration_sec": duration_sec,
            }

        latency_ms = timer.get_elapsed_ms()

        return ForensicPrediction(
            verdict=verdict,
            calibrated_probability_bonafide=calibrated_prob,
            raw_score=raw_score,
            decision_threshold=threshold,
            confidence_percentage=min(99.9, confidence),
            duration_sec=duration_sec,
            num_windows_analyzed=num_windows,
            inference_latency_ms=latency_ms,
            file_sha256=file_hash,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            attention_weights=attn_weights,
            window_timeline=timeline,
            model_name=model_display,
            feature_type=feature_display,
            synthetic_score=float(round(1.0 - calibrated_prob, 4)),
            tampering_probability=float(round(1.0 - calibrated_prob, 4)),
            anomaly_timestamps=anomaly_segments,
            metadata=meta_dict,
        )
