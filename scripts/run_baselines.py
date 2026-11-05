"""Classical Baselines Training CLI (Unified Entrypoint).

Delegates to train_classical.py for training GMM, SVM, Random Forest, and Logistic Regression
with PCA on train-only features and DEV-set hyperparameter tuning.

Author: Jashan Goyal (IIT2024223)
"""

from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.train_classical import main

if __name__ == "__main__":
    main()
