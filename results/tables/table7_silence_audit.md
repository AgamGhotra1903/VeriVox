| Audit Check                                | Observed Metric                                     | Criterion                     | Status   |
|:-------------------------------------------|:----------------------------------------------------|:------------------------------|:---------|
| Silence-Duration-Only Classifier (N=20000) | EER = 18.36% [17.79%, 18.85%], AUC = 0.8872         | EER >= 35.0%, AUC <= 0.65     | FAIL     |
| VAD-Trimmed vs Untrimmed Audio (N=20000)   | Untrimmed: 9.14%, Trimmed: 30.69%, Delta: 21.55 pts | |Delta EER| <= 5.0 pts        | FAIL     |
| Silence Swap Splicing (N=2000 pairs)       | Pearson r = 0.7036, Flip Rate = 34.85%              | r >= 0.85, Flip Rate <= 10.0% | FAIL     |