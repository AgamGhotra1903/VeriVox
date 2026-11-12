"""CRNN Model Training and Evaluation CLI (Unified Entrypoint).

Delegates to train_crnn.py for regularized CRNN training on LFCC or Log-Mel features.

"""

from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.train_crnn import main

if __name__ == "__main__":
    main()
