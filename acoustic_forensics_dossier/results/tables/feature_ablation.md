| Feature Representation       | Input Dimension                          |   Dev EER (%) |   Eval EER (%) |   Eval ROC-AUC |   Eval min-tDCF |
|:-----------------------------|:-----------------------------------------|--------------:|---------------:|---------------:|----------------:|
| LFCC (60-dim)                | 60 (20 stat + 20 delta + 20 delta-delta) |          0    |           0    |         1      |          0      |
| Log-Mel (80-dim)             | 80 Mel filterbank energy bins            |         67.5  |          60    |         0.4356 |          1      |
| Both (LFCC + Log-Mel Fusion) | 140 combined spectral channels           |         26.25 |          13.75 |         0.9625 |          0.2315 |