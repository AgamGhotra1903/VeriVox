"""Score matrix generation and persistence for score-level ensemble fusion.

Produces score matrices of shape (n_utterances, n_models) across dev, eval, and In-The-Wild
splits, persisting to disk as .npy files with accompanying utterance IDs and labels.

Author: Agam Ghotra (IIT2024226)
"""

import json
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    DEFAULT_RESULTS_DIR,
)
from acforensics.data.audio_io import load_and_validate_audio
from acforensics.data.segmentation import fixed_length_crop_or_pad
from acforensics.features.lfcc import LFCCExtractor
from acforensics.features.logmel import LogMelExtractor
from acforensics.features.normalization import FeatureNormalizer, per_utterance_cmvn
from acforensics.models.classical import extract_mean_std_features, predict_classical_score
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def build_score_matrix(
    models_dict: dict[str, Any],
    protocol_df: pd.DataFrame,
    device: Optional[torch.device] = None,
    split_name: str = "dev",
    output_dir: Optional[Path | str] = None,
    target_samples: int = TARGET_SAMPLES,
) -> dict[str, Any]:
    """Generates an (N, M) continuous score matrix across all base models for a given dataset split.

    Base models supported:
    - 'CRNN-LFCC': PyTorch CRNN on LFCC
    - 'CRNN-LogMel': PyTorch CRNN on Log-Mel
    - 'GMM': GMMForensics on LFCC frames
    - 'SVM': scikit-learn Pipeline with PCA and SVC
    - 'Random Forest': scikit-learn Pipeline with PCA and RandomForestClassifier
    - 'Logistic Regression': scikit-learn Pipeline with PCA and LogisticRegression

    All scores follow the invariant: Higher value = more bona fide (genuine).

    Args:
        models_dict: Dictionary mapping model names to model instances.
        protocol_df: DataFrame containing 'audio_path', 'label', and 'file_id'.
        device: PyTorch execution device.
        split_name: Name of the partition ('dev', 'eval', 'itw').
        output_dir: Target directory for saving .npy score matrices.
        target_samples: Target clip length in samples (default: 64,000 = 4.0s).

    Returns:
        Dictionary containing:
        - 'scores': (N, M) np.ndarray of float64 continuous scores
        - 'labels': (N,) np.ndarray of binary ground truth
        - 'file_ids': list of file ID strings
        - 'model_names': list of model name strings in column order
    """
    if device is None:
        device = torch.device("cpu")

    model_names = list(models_dict.keys())
    n_models = len(model_names)
    n_samples = len(protocol_df)

    logger.info("Building (%d x %d) score matrix for split '%s'...", n_samples, n_models, split_name)

    # Feature extractors
    lfcc_ext = LFCCExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_lfcc=20, with_deltas=True)
    logmel_ext = LogMelExtractor(sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_mels=80)

    # Normalizers
    norm_lfcc_file = Path("checkpoints/normalization_stats_lfcc.json")
    if not norm_lfcc_file.exists():
        norm_lfcc_file = Path("checkpoints/normalization_stats.json")
    norm_lfcc = FeatureNormalizer().load(norm_lfcc_file) if norm_lfcc_file.exists() else FeatureNormalizer()

    norm_logmel_file = Path("checkpoints/normalization_stats_logmel.json")
    norm_logmel = FeatureNormalizer().load(norm_logmel_file) if norm_logmel_file.exists() else FeatureNormalizer()

    scores_list = []
    labels_list = []
    file_ids_list = []

    for _, row in tqdm(protocol_df.iterrows(), total=len(protocol_df), desc=f"Scoring [{split_name}]"):
        audio_path = row["audio_path"]
        label = int(row["label"])
        file_id = str(row["file_id"])

        res = load_and_validate_audio(audio_path, target_sr=SAMPLE_RATE)
        if not res.is_valid or res.waveform is None:
            logger.warning("Corrupt or invalid audio file at %s: %s", audio_path, res.error_reason)
            wave = torch.zeros(target_samples, dtype=torch.float32)
        else:
            wave = fixed_length_crop_or_pad(res.waveform, target_samples=target_samples, mode="center")

        # 1. Extract raw features
        with torch.no_grad():
            f_lfcc = lfcc_ext(wave)     # (60, T)
            f_logmel = logmel_ext(wave) # (80, T)

            # Normalization for CRNN inputs
            f_lfcc_norm = norm_lfcc(f_lfcc) if norm_lfcc.is_fitted else per_utterance_cmvn(f_lfcc)
            f_logmel_norm = norm_logmel(f_logmel) if norm_logmel.is_fitted else per_utterance_cmvn(f_logmel)

        f_lfcc_np = f_lfcc.cpu().numpy()
        f_logmel_np = f_logmel.cpu().numpy()

        # Classical utterance vectors (mean + std)
        stat_lfcc = extract_mean_std_features(f_lfcc_np).reshape(1, -1)
        stat_logmel = extract_mean_std_features(f_logmel_np).reshape(1, -1)

        # 2. Query each base model for its continuous score
        utterance_scores = []
        for m_name in model_names:
            model = models_dict[m_name]
            score_val = 0.0

            if m_name == "CRNN-LFCC":
                with torch.no_grad():
                    inp = f_lfcc_norm.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 60, T)
                    score_val = float(model.predict_score(inp).cpu().item())
            elif m_name == "CRNN-LogMel":
                with torch.no_grad():
                    inp = f_logmel_norm.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 80, T)
                    score_val = float(model.predict_score(inp).cpu().item())
            elif m_name == "GMM":
                # GMM expects frames of shape (T, 60)
                score_val = float(model.score_utterance(f_lfcc_np.T))
            elif "LogMel" in m_name:
                score_val = float(predict_classical_score(model, stat_logmel)[0])
            else:
                # Classical models defaulting to LFCC mean+std
                score_val = float(predict_classical_score(model, stat_lfcc)[0])

            utterance_scores.append(score_val)

        scores_list.append(utterance_scores)
        labels_list.append(label)
        file_ids_list.append(file_id)

    scores_matrix = np.array(scores_list, dtype=np.float64)
    labels_arr = np.array(labels_list, dtype=np.int64)

    assert scores_matrix.shape == (n_samples, n_models), (
        f"Expected matrix shape ({n_samples}, {n_models}), got {scores_matrix.shape}"
    )

    result = {
        "scores": scores_matrix,
        "labels": labels_arr,
        "file_ids": file_ids_list,
        "model_names": model_names,
    }

    # Save to disk as .npy and .json
    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        scores_file = out_p / f"{split_name}_scores.npy"
        labels_file = out_p / f"{split_name}_labels.npy"
        ids_file = out_p / f"{split_name}_file_ids.npy"
        meta_file = out_p / f"{split_name}_models.json"

        np.save(scores_file, scores_matrix)
        np.save(labels_file, labels_arr)
        np.save(ids_file, np.array(file_ids_list))
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(model_names, f, indent=2)

        logger.info("Persisted score matrix to %s", scores_file)

    return result


def load_score_matrix(
    score_dir: Path | str,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Loads persisted score matrix, labels, utterance IDs, and model names from disk.

    Returns:
        Tuple of (scores_matrix, labels, file_ids, model_names).
    """
    p = Path(score_dir)
    scores = np.load(p / f"{split_name}_scores.npy")
    labels = np.load(p / f"{split_name}_labels.npy")
    file_ids = list(np.load(p / f"{split_name}_file_ids.npy"))
    with open(p / f"{split_name}_models.json", "r", encoding="utf-8") as f:
        model_names = json.load(f)

    return scores, labels, file_ids, model_names

