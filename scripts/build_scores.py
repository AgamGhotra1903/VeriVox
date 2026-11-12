"""Build Score Matrices CLI for Score-Level Fusion.

Extracts continuous scores from all base models:
- CRNN-LFCC
- CRNN-LogMel
- GMM (LFCC)
- SVM (LFCC + PCA)
- Random Forest (LFCC + PCA)
- Logistic Regression (LFCC + PCA)

Enforces strict zero-leakage assertions on utterance IDs.
Saves matrices (N, M) to disk as .npy files with file_ids and labels.

"""

import argparse
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import joblib
import numpy as np
import torch

from acforensics.constants import (
    LFCC_DEFAULT_DIM,
    LOGMEL_DEFAULT_DIM,
    DEFAULT_RESULTS_DIR,
)
from acforensics.data.protocols import load_asvspoof_protocol, load_in_the_wild_protocol
from acforensics.models.crnn import CRNNForensics
from acforensics.models.gmm import GMMForensics
from acforensics.ensemble.scores import build_score_matrix
from acforensics.ensemble.stacking import assert_zero_data_leakage
from acforensics.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def load_crnn_model(ckpt_path: Path, feature_type: str, in_dim: int, device: torch.device) -> CRNNForensics:
    """Loads trained CRNN model from checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    m_cfg = ckpt.get("model_config", {})
    gru_cfg = m_cfg.get("gru", {}) if isinstance(m_cfg.get("gru"), dict) else {}
    cnn_cfg = m_cfg.get("cnn", {}) if isinstance(m_cfg.get("cnn"), dict) else {}
    attn_cfg = m_cfg.get("attention", {}) if isinstance(m_cfg.get("attention"), dict) else {}
    clf_cfg = m_cfg.get("classifier", {}) if isinstance(m_cfg.get("classifier"), dict) else {}

    model = CRNNForensics(
        in_channels=1,
        in_freq_dim=in_dim,
        features=feature_type,
        use_gru=m_cfg.get("use_gru", True),
        use_attention=m_cfg.get("use_attention", True),
        pooling=m_cfg.get("pooling", "attention"),
        proj_dim=m_cfg.get("proj_dim", 128),
        gru_hidden=m_cfg.get("gru_hidden") or gru_cfg.get("hidden_size", 128),
        num_gru_layers=m_cfg.get("num_gru_layers") or gru_cfg.get("num_layers", 2),
        gru_dropout=m_cfg.get("gru_dropout") if "gru_dropout" in m_cfg else gru_cfg.get("dropout", 0.3),
        conv_dropout=m_cfg.get("conv_dropout") if "conv_dropout" in m_cfg else cnn_cfg.get("dropout", 0.1),
        classifier_dropout=m_cfg.get("classifier_dropout") if "classifier_dropout" in m_cfg else clf_cfg.get("dropout", 0.3),
        num_attn_heads=m_cfg.get("num_attn_heads") or attn_cfg.get("num_heads", 4),
        num_classes=m_cfg.get("num_classes", 2),
        init_weights=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Build Score Matrices for Ensemble Fusion")
    parser.add_argument("--smoke", action="store_true", help="Run on smoke test dataset")
    parser.add_argument("--device", type=str, default="auto", help="Execution device")
    parser.add_argument("--out-dir", type=str, default="results/scores", help="Output directory for scores")
    parser.add_argument("--include-itw", action="store_true", default=True, help="Include In-The-Wild scoring")
    args = parser.parse_args()

    setup_logging()

    if args.device == "auto":
        device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)

    base_dir = Path("data/smoke_test" if args.smoke else "data/raw/LA")
    if not (base_dir / "ASVspoof2019_LA_cm_protocols").exists():
        base_dir = Path("data/smoke_test")

    proto_dir = base_dir / "ASVspoof2019_LA_cm_protocols"
    train_proto = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_proto = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_proto = proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"

    train_df = load_asvspoof_protocol(train_proto, audio_dir=base_dir / "ASVspoof2019_LA_train" / "flac", split_name="train")
    dev_df = load_asvspoof_protocol(dev_proto, audio_dir=base_dir / "ASVspoof2019_LA_dev" / "flac", split_name="dev")
    eval_df = load_asvspoof_protocol(eval_proto, audio_dir=base_dir / "ASVspoof2019_LA_eval" / "flac", split_name="eval")

    # Optional In-The-Wild
    itw_df = None
    itw_meta = Path("data/smoke_test/in_the_wild/meta.csv" if args.smoke else "data/raw/in_the_wild/meta.csv")
    itw_audio = Path("data/smoke_test/in_the_wild/audio" if args.smoke else "data/raw/in_the_wild/audio")
    if itw_meta.exists() and itw_audio.exists():
        itw_df = load_in_the_wild_protocol(itw_meta, audio_dir=itw_audio)

    assert_zero_data_leakage(
        train_df["file_id"],
        dev_df["file_id"],
        eval_df["file_id"],
        itw_df["file_id"] if itw_df is not None else None,
    )

    models_dict = {}

    crnn_lfcc_path = Path("checkpoints/crnn_lfcc_best.pt")
    if not crnn_lfcc_path.exists():
        crnn_lfcc_path = Path("checkpoints/crnn_best.pt")
    assert crnn_lfcc_path.exists(), f"CRNN-LFCC checkpoint missing at {crnn_lfcc_path}. Run train_crnn.py first."
    models_dict["CRNN-LFCC"] = load_crnn_model(crnn_lfcc_path, "lfcc", LFCC_DEFAULT_DIM, device)

    crnn_logmel_path = Path("checkpoints/crnn_logmel_best.pt")
    assert crnn_logmel_path.exists(), f"CRNN-LogMel checkpoint missing at {crnn_logmel_path}. Run train_crnn.py first."
    models_dict["CRNN-LogMel"] = load_crnn_model(crnn_logmel_path, "logmel", LOGMEL_DEFAULT_DIM, device)

    gmm_path = Path("checkpoints/classical/gmm_lfcc.joblib")
    assert gmm_path.exists(), f"GMM checkpoint missing at {gmm_path}. Run train_classical.py first."
    models_dict["GMM"] = GMMForensics.load(gmm_path)

    svm_path = Path("checkpoints/classical/rbf_svm_lfcc.joblib")
    assert svm_path.exists(), f"SVM checkpoint missing at {svm_path}."
    models_dict["SVM"] = joblib.load(str(svm_path))

    rf_path = Path("checkpoints/classical/random_forest_lfcc.joblib")
    assert rf_path.exists(), f"Random Forest checkpoint missing at {rf_path}."
    models_dict["Random Forest"] = joblib.load(str(rf_path))

    lr_path = Path("checkpoints/classical/logistic_regression_lfcc.joblib")
    assert lr_path.exists(), f"Logistic Regression checkpoint missing at {lr_path}."
    models_dict["Logistic Regression"] = joblib.load(str(lr_path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_res = build_score_matrix(
        models_dict=models_dict,
        protocol_df=dev_df,
        device=device,
        split_name="dev",
        output_dir=out_dir,
    )

    eval_res = build_score_matrix(
        models_dict=models_dict,
        protocol_df=eval_df,
        device=device,
        split_name="eval",
        output_dir=out_dir,
    )

    # C. In-The-Wild split (cross-domain evaluation)
    if itw_df is not None and args.include_itw:
        itw_res = build_score_matrix(
            models_dict=models_dict,
            protocol_df=itw_df,
            device=device,
            split_name="itw",
            output_dir=out_dir,
        )

    print("\n" + "=" * 80)
    print("SCORE MATRICES PERSISTED SUCCESSFULLY:")
    print("=" * 80)
    print(f" Dev Score Matrix:   {dev_res['scores'].shape} -> {out_dir / 'dev_scores.npy'}")
    print(f" Eval Score Matrix:  {eval_res['scores'].shape} -> {out_dir / 'eval_scores.npy'}")
    if itw_df is not None:
        print(f" ITW Score Matrix:   {itw_res['scores'].shape} -> {out_dir / 'itw_scores.npy'}")
    print(f" Base Models ({len(models_dict)}): {list(models_dict.keys())}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
