# Sources, Academic Attribution, and Provenance

This repository implements **Acoustic Forensics** (VeriVox), an open forensic detection system for deepfake and synthetic speech. All components, mathematical formulations, baseline references, and datasets are transparently attributed below in accordance with academic integrity and software provenance standards.

---

## 1. Provenance and Attribution Matrix

| Component | File Path | Original Authors / Organization | Publication / Source URL | License | Similarity | Adaptation / Scope |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Official min-tDCF & EER Metrics** | `src/acforensics/evaluation/official_tdcf/eval_metrics.py` | T. Kinnunen, K. A. Lee, H. Delgado, N. Evans, M. Todisco et al. | [IEEE/ACM TASLP 2020](https://doi.org/10.1109/TASLP.2020.2974944) / [ASVspoof 2019 Baseline Repo](https://github.com/asvspoof-challenge/2019) | BSD-3-Clause / Academic Open | **High** | Official evaluation functions (`compute_tDCF_legacy`, `compute_eer`, `compute_det_curve`) preserved verbatim to maintain exact mathematical comparability with the official benchmark. |
| **LFCC Feature Extraction** | `src/acforensics/features/lfcc.py` | M. Sahidullah, T. Kinnunen, C. Hanilçi | [Proc. Interspeech 2015](https://doi.org/10.21437/Interspeech.2015-467) | MIT / Open Academic | **Medium** | Re-implemented natively in PyTorch and torchaudio using `torch.stft`, custom triangular filterbank matrix multiplication, and orthonormal DCT-II matrices. |
| **GMM Baseline Formulation** | `src/acforensics/models/gmm.py` | M. Todisco, X. Wang, V. Vestman, M. Sahidullah et al. (ASVspoof Consortium) | [Proc. Interspeech 2019](https://doi.org/10.21437/Interspeech.2019-1417) | BSD-3-Clause / Academic | **Medium** | Algorithmic design mirrors the ASVspoof 2019 primary spectral baseline: two 64-component diagonal-covariance Gaussian Mixture Models fitted on frame-level LFCCs via scikit-learn. |
| **CRNN with Self-Attention** | `src/acforensics/models/crnn.py` | Original design based on CRNN principles (Choi et al., 2017; Jung et al., 2022) | [AASIST (Jung et al., ICASSP 2022)](https://doi.org/10.1109/ICASSP43922.2022.9746487) | MIT / Original | **Low** | Original PyTorch architecture: 4-stage Conv2D encoder (BatchNorm, ReLU, MaxPool, Conv Dropout), 2-layer BiGRU context aggregator, 4-head multi-head self-attention temporal pooling, and linear classifier. |
| **Stacked Generalization** | `src/acforensics/ensemble/stacking.py` | David H. Wolpert | [Neural Networks, Vol. 5, 1992](https://doi.org/10.1016/S0893-6080(05)80023-1) | Open Academic | **Low** | Original Python implementation of Wolpert's stacking paradigm with validation-anchored z-score standardization ($z_k = (s_k - \mu_k)/\sigma_k$) and $L_2$-regularized Logistic Regression meta-learner. |
| **ASVspoof 2019 LA Dataset** | `data/raw/LA/`, `data/manifests/` | Xin Wang, Junichi Yamagishi et al. (ASVspoof Consortium) | [Computer Speech & Language 2020](https://doi.org/10.1016/j.csl.2020.101114) | ASVspoof 2019 Agreement (CC BY-NC 4.0) | **N/A (Data)** | Primary benchmark dataset. Audio files are processed under strict speaker-disjoint boundaries (Train: 20 spk, Dev: 20 spk, Eval: 48 spk). |
| **In-The-Wild Dataset** | `data/raw/in_the_wild/` | Nicolas M. Müller, P. Czempin, F. Dieckmann, A. Froghyar, K. Böttinger | [Proc. Interspeech 2022](https://doi.org/10.21437/Interspeech.2022-10846) | CC BY-SA 4.0 | **N/A (Data)** | Unconstrained out-of-domain benchmark corpus comprising 25,589 audio recordings (15,403 bona fide, 10,186 spoof) evaluated strictly zero-shot. |

---

## 2. Component Details and Implementation Lineage

### 2.1 Official Evaluation Metrics (`src/acforensics/evaluation/official_tdcf/eval_metrics.py`)
- **Origin:** The functions `compute_tDCF_legacy`, `compute_eer`, and `compute_det_curve` were authored by Tomi Kinnunen et al. as part of the official ASVspoof 2019 Challenge evaluation kit.
- **Why Kept:** Computing min-tDCF and EER using different mathematical heuristics would invalidate direct numerical comparisons against the official ASVspoof 2019 challenge leaderboards and published literature.
- **License:** Released by the ASVspoof consortium under an open academic research license (BSD-compatible).

### 2.2 Linear Frequency Cepstral Coefficients (`src/acforensics/features/lfcc.py`)
- **Origin:** The superiority of linear-scale cepstral features over Mel-scale representations for synthetic speech detection was established by Sahidullah et al. (Interspeech 2015).
- **Implementation:** Custom native PyTorch module utilizing `torch.stft`, linear triangular filterbanks, logarithm compression, and orthonormal DCT-II transformation matrix multiplication. No vendored code from external repositories is used.

### 2.3 Deep CRNN Architecture (`src/acforensics/models/crnn.py`)
- **Origin:** Conceptualized from sound event detection and deep fake audio architectures (AASIST, RawNet2, LCNN).
- **Implementation:** Written from scratch using PyTorch primitives. Contains a 4-block 2D convolutional encoder, a bidirectional GRU sequence encoder, a learned 4-head multi-head self-attention pooling mechanism, and an MLP classification head. Total parameters: 1,439,778.

### 2.4 Stacked Ensemble Meta-Learner (`src/acforensics/ensemble/stacking.py`)
- **Origin:** Wolpert's stacked generalization (1992).
- **Implementation:** Custom class wrapping scikit-learn's `LogisticRegression`. Implements strict validation-anchored score standardization where means and standard deviations are computed strictly on the development set and frozen for evaluation, preventing data leakage.

---

## 3. License Compatibility Review

| Dependency / Component | License | Compatibility with Acoustic Forensics |
| :--- | :--- | :---: |
| Python Standard Library | PSF License | Compatible |
| PyTorch / Torchaudio | Modified BSD | Compatible |
| scikit-learn | BSD-3-Clause | Compatible |
| FastAPI | MIT | Compatible |
| ASVspoof official eval metrics | BSD-3-Clause / Academic Open | Compatible |
| React / Vite (Frontend) | MIT | Compatible |

**License Conflict Finding:** No copyleft (GPL / AGPL) libraries or code blocks are included in the source distribution. All licenses are permissive (MIT, BSD-3-Clause, Apache-2.0, or Academic Open Access), ensuring full compatibility for academic publication and open-source release.

---

## 4. How to Cite in Publications

If you use Acoustic Forensics or reference our benchmarks, please cite:

```bibtex
@inproceedings{acousticforensics2026,
  title     = {Acoustic Forensics: A Stacked Ensemble of CRNN and Classical Models for Synthetic Speech Detection},
  author    = {VeriVox Research Team},
  booktitle = {Proceedings of the IEEE International Conference on Acoustic Forensics and Biometrics},
  year      = {2026}
}
```

Please also cite the foundational benchmark papers:
1. **ASVspoof 2019 Challenge & Dataset:**
   ```bibtex
   @article{wang2020asvspoof,
     title   = {ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
     author  = {Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and Delgado, H{\'e}ctor and others},
     journal = {Computer Speech \& Language},
     volume  = {64},
     pages   = {101114},
     year    = {2020}
   }
   ```
2. **t-DCF Metric Formulation:**
   ```bibtex
   @article{kinnunen2020tandem,
     title   = {The tandem detection cost function for evaluation of spoofing countermeasures with speaker verification},
     author  = {Kinnunen, Tomi and Lee, Kong Aik and Delgado, H{\'e}ctor and Evans, Nicholas and others},
     journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
     volume  = {28},
     pages   = {783--795},
     year    = {2020}
   }
   ```
3. **In-The-Wild Dataset:**
   ```bibtex
   @inproceedings{muller2022does,
     title     = {Does audio deepfake detection generalize?},
     author    = {M{\"u}ller, Nicolas M and Czempin, Philip and Dieckmann, Franziska and Froghyar, Abduwali and B{\"o}ttinger, Konstantin},
     booktitle = {Proc. Interspeech},
     pages     = {2783--2787},
     year      = {2022}
   }
   ```
4. **LFCC for Anti-Spoofing:**
   ```bibtex
   @inproceedings{sahidullah2015comparison,
     title     = {A comparison of features for synthetic speech detection},
     author    = {Sahidullah, Md and Kinnunen, Tomi and Hanil{\c{c}}i, Cemal},
     booktitle = {Proc. Interspeech},
     pages     = {2087--2091},
     year      = {2015}
   }
   ```

