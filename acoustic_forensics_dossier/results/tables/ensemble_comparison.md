| System                          | Type                   |   Dev EER (%) |   Eval EER (%) |   Eval ROC-AUC |   Eval min-tDCF | Applied Threshold   |
|:--------------------------------|:-----------------------|--------------:|---------------:|---------------:|----------------:|:--------------------|
| Single: CRNN-LFCC               | Base Model             |             0 |           0    |         1      |               0 | 19.2115 (Dev)       |
| Single: CRNN-LogMel             | Base Model             |            75 |          61.25 |         0.3012 |               1 | 1.6352 (Dev)        |
| Single: GMM                     | Base Model             |             0 |           0    |         1      |               0 | 268.5717 (Dev)      |
| Single: SVM                     | Base Model             |             0 |           0    |         1      |               0 | 0.6752 (Dev)        |
| Single: Random Forest           | Base Model             |             0 |           0    |         1      |               0 | 0.3640 (Dev)        |
| Single: Logistic Regression     | Base Model             |             0 |           0    |         1      |               0 | 1.0370 (Dev)        |
| Baseline Fusion (Mean Z-Scores) | Unweighted Average     |             0 |           0    |         1      |               0 | 1.0684 (Dev)        |
| Stacked Ensemble (Proposed)     | L2 Logistic Regression |             0 |           0    |         1      |               0 | 0.2428 (Dev)        |