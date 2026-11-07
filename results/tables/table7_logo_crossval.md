| Held-Out Generator   | Algorithm Family                |   Held-Out Dev EER (%) |   Held-Out Dev ROC-AUC | Pass/Fail (Threshold < 35%)   |
|:---------------------|:--------------------------------|-----------------------:|-----------------------:|:------------------------------|
| A01                  | Neural TTS (WaveNet)            |                   4.79 |                 0.9925 | PASS                          |
| A02                  | Neural TTS (WORLD)              |                   9.39 |                 0.9639 | PASS                          |
| A03                  | Vocoder TTS (Traditional)       |                  25.65 |                 0.8266 | PASS                          |
| A04                  | Waveform Concatenation TTS      |                  42.55 |                 0.5989 | FAIL                          |
| A05                  | Voice Conversion (Traditional)  |                  31.08 |                 0.758  | PASS                          |
| A06                  | Voice Conversion (WORLD)        |                  57.5  |                 0.3978 | FAIL                          |
| MEAN LOGO            | Average Across 6 Held-Out Folds |                  28.49 |                 0.7563 | FAIL                          |
