| Attack ID   | Taxonomy      | Synthesis Description                            |   Trials |   CRNN EER (%) |   CRNN min-tDCF |   Ensemble EER (%) |   Ensemble min-tDCF |
|:------------|:--------------|:-------------------------------------------------|---------:|---------------:|----------------:|-------------------:|--------------------:|
| A07         | Unseen Tech   | Neural TTS (LSTM-RNN + WORLD + GAN)              |     4914 |           1.05 |          0.0942 |               2.57 |              0.1284 |
| A08         | Unseen Tech   | Neural TTS (AR LSTM-RNN + neural source-filter)  |     4914 |           1.43 |          0.1008 |               0.35 |              0.0737 |
| A09         | Unseen Tech   | Neural TTS (LSTM-RNN + Vocaine)                  |     4914 |           0.14 |          0.0668 |               0    |              0.0627 |
| A10         | Unseen Tech   | Neural TTS (seq2seq + WaveRNN)                   |     4914 |           1.2  |          0.1003 |               6.8  |              0.2283 |
| A11         | Unseen Tech   | Neural TTS (seq2seq + Griffin-Lim)               |     4914 |           0.26 |          0.0708 |               0.73 |              0.0806 |
| A12         | Unseen Tech   | Neural TTS (WaveNet vocoder)                     |     4914 |           1.59 |          0.1131 |               8.65 |              0.2965 |
| A13         | Unseen Tech   | TTS-VC (Moment matching NN + waveform filtering) |     4914 |           0.98 |          0.0903 |               0.96 |              0.0888 |
| A14         | Unseen Tech   | TTS-VC (LSTM-RNN + STRAIGHT)                     |     4914 |           0.97 |          0.0926 |               2.34 |              0.1237 |
| A15         | Unseen Tech   | TTS-VC (LSTM-RNN + WaveNet)                      |     4914 |           0.54 |          0.0797 |               2.89 |              0.1309 |
| A16         | Known Variant | Waveform Concatenation TTS (A04 variant)         |     4914 |           1.48 |          0.1098 |               4.29 |              0.1714 |
| A17         | Unseen Tech   | Neural VC (VAE + waveform filtering)             |     4914 |          15.06 |          0.5038 |              24.36 |              0.6647 |
| A18         | Unseen Tech   | VC (i-vector / PLDA + MFCC-to-waveform)          |     4914 |          22.04 |          0.7024 |              34.33 |              0.956  |
| A19         | Known Variant | VC with spectral filtering (A06 variant)         |     4914 |           4.48 |          0.1925 |              14    |              0.4176 |
