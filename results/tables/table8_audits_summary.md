| Audit Check                     | Observed Metric                                    | Criterion                     | Status   |
|:--------------------------------|:---------------------------------------------------|:------------------------------|:---------|
| Silence Duration Classifier     | EER = 21.55%, AUC = 0.8579                         | EER >= 35.0%, AUC <= 0.65     | FAIL     |
| VAD-Trimmed vs Untrimmed Audio  | Untrimmed: 8.67%, Trimmed: 16.00%, Delta: 7.33 pts | |Delta EER| <= 5.0 pts        | FAIL     |
| Silence-Swap Score Stability    | Pearson r = 0.7131, Prediction Flip Rate = 36.5%   | r >= 0.80, Flip Rate <= 10.0% | FAIL     |
| Duration Control                | CRNN r = +0.0000, Ensemble r = +0.0000             | |r| <= 0.20                   | PASS     |
| Loudness Control (RMS dBFS)     | CRNN r = +0.0000, Ensemble r = +0.0000             | |r| <= 0.20                   | FAIL     |
| Bandwidth Control (95% Rolloff) | CRNN r = +0.0000, Ensemble r = +0.0000             | |r| <= 0.35                   | PASS     |