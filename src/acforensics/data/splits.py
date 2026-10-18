"""Dataset partitioning and split isolation protocols ("AF-Split v1").

Enforces rigorous experimental separation:
- ASVspoof 2019 LA official splits: Train (A01-A06), Dev (A01-A06, disjoint speakers), Eval (A07-A19 unseen).
- In-The-Wild: Speaker-disjoint partitioning into ITW-CAL (~15% speakers) and ITW-TEST (~85% speakers).
- Extension splits: Split strictly by generator and speaker family, never by random utterance.

Author: Paras Bansal (IIT2024221), Agam Ghotra (IIT2024226)
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)


def split_in_the_wild_by_speaker(
    itw_df: pd.DataFrame,
    cal_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions In-The-Wild dataset into speaker-disjoint ITW-CAL and ITW-TEST splits.

    Crucial principle (Section 3.2):
    - ITW-CAL contains ~15% of speakers (chosen via deterministic seed). Used strictly
      for optional few-shot threshold adaptation.
    - ITW-TEST contains ~85% of speakers. Reserved for zero-shot cross-domain benchmarking.
    - Speaker disjointness is strictly verified and guaranteed.

    Args:
        itw_df: Standardized DataFrame of In-The-Wild utterances (must contain 'speaker_id' or 'speaker').
        cal_ratio: Fraction of speakers assigned to calibration split (default: 0.15).
        seed: Random seed for deterministic speaker partitioning (default: 42).

    Returns:
        tuple of (itw_cal_df, itw_test_df)

    Raises:
        ValueError: If speaker column is missing or speaker count is insufficient.
    """
    speaker_col = "speaker_id" if "speaker_id" in itw_df.columns else "speaker"
    if speaker_col not in itw_df.columns:
        raise ValueError(f"In-The-Wild DataFrame missing '{speaker_col}' column.")

    unique_speakers = sorted(itw_df[speaker_col].astype(str).unique())
    n_speakers = len(unique_speakers)

    if n_speakers < 2:
        logger.warning(
            "Only %d unique speaker(s) found in In-The-Wild DataFrame. "
            "Cannot construct strictly disjoint calibration/test partitions without multiple speakers.",
            n_speakers,
        )
        itw_cal_df = itw_df.iloc[:0].copy()
        itw_test_df = itw_df.copy()
        itw_test_df["split"] = "itw_test"
        return itw_cal_df, itw_test_df

    rng = np.random.RandomState(seed)
    shuffled_speakers = list(unique_speakers)
    rng.shuffle(shuffled_speakers)

    n_cal_speakers = max(1, int(round(n_speakers * cal_ratio)))
    cal_speakers = set(shuffled_speakers[:n_cal_speakers])
    test_speakers = set(shuffled_speakers[n_cal_speakers:])

    # Strict assertion: Disjointness
    overlap = cal_speakers.intersection(test_speakers)
    assert not overlap, f"Fatal split error: Speaker leakage detected! Overlapping speakers: {overlap}"

    itw_cal_df = itw_df[itw_df[speaker_col].isin(cal_speakers)].copy()
    itw_test_df = itw_df[itw_df[speaker_col].isin(test_speakers)].copy()

    itw_cal_df["split"] = "itw_cal"
    itw_test_df["split"] = "itw_test"

    logger.info(
        "In-The-Wild speaker-disjoint partition completed (Seed: %d):\n"
        "  Total Speakers: %d\n"
        "  ITW-CAL:  %d speakers, %d utterances (Bona fide: %d, Spoof: %d)\n"
        "  ITW-TEST: %d speakers, %d utterances (Bona fide: %d, Spoof: %d)",
        seed,
        n_speakers,
        len(cal_speakers),
        len(itw_cal_df),
        (itw_cal_df["label"] == BONAFIDE).sum(),
        (itw_cal_df["label"] == SPOOF).sum(),
        len(test_speakers),
        len(itw_test_df),
        (itw_test_df["label"] == BONAFIDE).sum(),
        (itw_test_df["label"] == SPOOF).sum(),
    )

    return itw_cal_df.reset_index(drop=True), itw_test_df.reset_index(drop=True)


def split_extension_by_generator(
    extension_df: pd.DataFrame,
    test_generator: str,
    val_generator: Optional[str] = None,
    generator_col: str = "attack_id",
    speaker_col: str = "speaker_id",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partitions multi-domain extension dataset (e.g. DFADD, WaveFake) by generator and speaker.

    Crucial principle (Section 3.3):
    - Split by GENERATOR, not by utterance.
    - Test split holds out an entire vocoder or synthesis family (unseen generator).
    - Val split holds out a distinct generator for hyperparameter tuning.
    - Bona fide speech is split by speaker to avoid speaker-shortcut bias.

    Args:
        extension_df: DataFrame of extension dataset.
        test_generator: Generator identifier held out completely for TEST-X.
        val_generator: Optional generator identifier held out for VAL-X.
        generator_col: Column indicating generator or attack ID.
        speaker_col: Column indicating speaker ID.
        seed: Random seed for speaker splitting of bona fide audio.

    Returns:
        tuple of (train_x_df, val_x_df, test_x_df)
    """
    if generator_col not in extension_df.columns:
        raise ValueError(f"Extension DataFrame missing generator column '{generator_col}'.")

    # Spoof partitions by generator
    test_spoofs = extension_df[
        (extension_df["label"] == SPOOF) & (extension_df[generator_col] == test_generator)
    ].copy()

    if val_generator is not None:
        val_spoofs = extension_df[
            (extension_df["label"] == SPOOF) & (extension_df[generator_col] == val_generator)
        ].copy()
        train_spoofs = extension_df[
            (extension_df["label"] == SPOOF)
            & (~extension_df[generator_col].isin([test_generator, val_generator]))
        ].copy()
    else:
        val_spoofs = extension_df.iloc[:0].copy()
        train_spoofs = extension_df[
            (extension_df["label"] == SPOOF) & (extension_df[generator_col] != test_generator)
        ].copy()

    # Bona fide partition by speaker
    bonafides = extension_df[extension_df["label"] == BONAFIDE].copy()
    if len(bonafides) > 0 and speaker_col in bonafides.columns:
        speakers = sorted(bonafides[speaker_col].astype(str).unique())
        rng = np.random.RandomState(seed)
        rng.shuffle(speakers)

        if len(speakers) >= 3:
            n_test = max(1, int(len(speakers) * 0.2))
            n_val = max(1, int(len(speakers) * 0.2)) if val_generator else 0
            test_spk = set(speakers[:n_test])
            val_spk = set(speakers[n_test : n_test + n_val])
            train_spk = set(speakers[n_test + n_val :])

            bf_test = bonafides[bonafides[speaker_col].isin(test_spk)]
            bf_val = bonafides[bonafides[speaker_col].isin(val_spk)]
            bf_train = bonafides[bonafides[speaker_col].isin(train_spk)]
        else:
            logger.warning(
                "Extension bona fide has only %d speaker(s) (e.g. LJSpeech single speaker). "
                "Warning: Speaker shortcut confound exists; results must be interpreted with caution.",
                len(speakers),
            )
            # Stratified random split if single speaker, with explicit documented caveat
            bf_train = bonafides.sample(frac=0.6, random_state=seed)
            bf_remain = bonafides.drop(bf_train.index)
            bf_val = bf_remain.sample(frac=0.5, random_state=seed) if val_generator else bonafides.iloc[:0]
            bf_test = bf_remain.drop(bf_val.index)
    else:
        bf_train = bonafides.iloc[:0].copy()
        bf_val = bonafides.iloc[:0].copy()
        bf_test = bonafides.iloc[:0].copy()

    train_x = pd.concat([train_spoofs, bf_train]).reset_index(drop=True)
    val_x = pd.concat([val_spoofs, bf_val]).reset_index(drop=True)
    test_x = pd.concat([test_spoofs, bf_test]).reset_index(drop=True)

    train_x["split"] = "train_x"
    val_x["split"] = "val_x"
    test_x["split"] = "test_x"

    logger.info(
        "Extension split by generator complete: TRAIN-X (%d), VAL-X (%d), TEST-X (%d - held out: %s)",
        len(train_x),
        len(val_x),
        len(test_x),
        test_generator,
    )
    return train_x, val_x, test_x


def summarize_splits(splits_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Produces a structured verification table of all active dataset splits.

    Returns summary DataFrame with columns:
    ['split', 'total', 'bonafide', 'spoof', 'bonafide_pct', 'unique_speakers', 'attacks']
    """
    rows = []
    for split_name, df in splits_dict.items():
        total = len(df)
        if total == 0:
            continue
        bf_cnt = int((df["label"] == BONAFIDE).sum())
        sp_cnt = int((df["label"] == SPOOF).sum())
        spk_col = "speaker_id" if "speaker_id" in df.columns else ("speaker" if "speaker" in df.columns else None)
        n_spk = int(df[spk_col].nunique()) if spk_col else -1

        attacks = sorted(df[df["label"] == SPOOF]["attack_id"].astype(str).unique())
        attacks_str = ", ".join(attacks) if len(attacks) <= 8 else f"{len(attacks)} attacks ({attacks[0]}..{attacks[-1]})"

        rows.append({
            "split": split_name,
            "total": total,
            "bonafide": bf_cnt,
            "spoof": sp_cnt,
            "bonafide_pct": round(bf_cnt / total * 100, 2),
            "unique_speakers": n_spk,
            "attacks": attacks_str,
        })
    return pd.DataFrame(rows)
