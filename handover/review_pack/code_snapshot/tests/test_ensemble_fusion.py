import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from acforensics.constants import BONAFIDE, SPOOF
from acforensics.ensemble.stacking import NonNegativeStackedEnsemble

def test_speaker_disjoint_calibration_split():
    speakers = np.array(["spk_1"] * 50 + ["spk_2"] * 50 + ["spk_3"] * 50 + ["spk_4"] * 50 + ["spk_5"] * 50)
    labels = np.array([BONAFIDE, SPOOF] * 125)
    features = np.random.randn(250, 10)
    sgkf = StratifiedGroupKFold(n_splits=5)
    train_idx, cal_idx = next(sgkf.split(features, labels, groups=speakers))
    train_spks = set(speakers[train_idx])
    cal_spks = set(speakers[cal_idx])
    assert len(train_spks.intersection(cal_spks)) == 0

def test_non_negative_stacked_ensemble_weights_and_monotonicity():
    scaler = StandardScaler()
    ref_data = np.random.randn(100, 3)
    scaler.fit(ref_data)
    weights = np.array([0.8, 0.4, 0.1])
    intercept = -0.05
    ensemble = NonNegativeStackedEnsemble(
        model_names=["m1", "m2", "m3"],
        weights=weights,
        intercept=intercept,
        scaler=scaler,
    )
    assert len(ensemble.weights) == 3
    assert np.all(ensemble.weights >= 0.0)
    high_score = ensemble.predict_score(np.array([[3.0, 3.0, 1.0]]))
    low_score = ensemble.predict_score(np.array([[-3.0, -3.0, -1.0]]))
    assert high_score[0] > low_score[0]
