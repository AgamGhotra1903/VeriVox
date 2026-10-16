# Acoustic Forensics Makefile
SHELL := /bin/bash
PYTHON := $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PIP := $(shell if [ -f .venv/bin/pip ]; then echo .venv/bin/pip; else echo pip3; fi)

.PHONY: help setup data features baselines train evaluate robustness crossdomain errors report demo test smoke clean

help:
	@echo "Acoustic Forensics - Command Targets:"
	@echo "  make setup        - Initialize Python virtual environment and install dependencies"
	@echo "  make smoke        - Generate synthetic smoke-test data and run the end-to-end pipeline in debug mode"
	@echo "  make test         - Run full pytest test suite"
	@echo "  make data         - Validate and summarize ASVspoof 2019 / In-the-Wild dataset protocols"
	@echo "  make features     - Extract and cache LFCC and Log-Mel features"
	@echo "  make baselines    - Train and evaluate classical baselines (LR, RBF-SVM, RF)"
	@echo "  make train        - Train CRNN architecture on ASVspoof 2019 LA"
	@echo "  make evaluate     - Run complete evaluation suite on official eval split"
	@echo "  make robustness   - Run codec compression stress tests (MP3, AAC, Opus)"
	@echo "  make crossdomain  - Run zero-shot evaluation on In-the-Wild dataset"
	@echo "  make errors       - Run detailed error analysis and breakdown by attack/speaker"
	@echo "  make report       - Generate automated markdown tables and figures for reports"
	@echo "  make demo         - Launch the interactive Streamlit forensic detection dashboard"
	@echo "  make clean        - Remove caches, temporary files, and test artifacts"

export PYTHONPATH := src:$(PYTHONPATH)

setup:
	@echo "Setting up Python virtual environment..."
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Setup complete. Activate with: source .venv/bin/activate"

data:
	$(PYTHON) scripts/prepare_data.py --config configs/default.yaml

smoke:
	@echo "Running quick end-to-end smoke test pipeline..."
	$(PYTHON) scripts/prepare_data.py --config configs/default.yaml --generate-smoke-data
	$(PYTHON) -m pytest tests/ -v
	@echo "Smoke test passed successfully."

test:
	$(PYTHON) -m pytest tests/ -v --durations=10

features:
	$(PYTHON) scripts/extract_features.py --config configs/default.yaml

baselines:
	$(PYTHON) scripts/run_baselines.py --config configs/baselines.yaml

train:
	$(PYTHON) scripts/train.py --config configs/crnn.yaml

evaluate:
	$(PYTHON) scripts/evaluate.py --config configs/default.yaml

robustness:
	$(PYTHON) scripts/run_robustness.py --config configs/robustness.yaml

crossdomain:
	$(PYTHON) scripts/run_crossdomain.py --config configs/default.yaml

errors:
	$(PYTHON) scripts/run_error_analysis.py --config configs/default.yaml

report:
	$(PYTHON) scripts/build_report_tables.py --results-dir results/

demo:
	PYTHONPATH=. $(PYTHON) -m streamlit run app/streamlit_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/
