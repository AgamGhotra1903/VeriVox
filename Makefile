# Acoustic Forensics Makefile
SHELL := /bin/bash
PYTHON := $(shell if [ -f .venv/bin/python3 ]; then echo .venv/bin/python3; elif [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PIP := $(shell if [ -f .venv/bin/pip ]; then echo .venv/bin/pip; else echo pip3; fi)

.PHONY: help setup data manifests audit features baselines train evaluate robustness crossdomain errors report reproduce demo api web web-build app docker-up docker-down test smoke clean

help:
	@echo "Acoustic Forensics - Command Targets:"
	@echo "  make setup        - Initialize Python virtual environment and install dependencies"
	@echo "  make smoke        - Generate synthetic smoke-test data, prepare manifests, audit leakage, run tests"
	@echo "  make test         - Run full pytest test suite (97 tests)"
	@echo "  make reproduce    - Run end-to-end extraction, training, evaluation, audits, and report generation"
	@echo "  make api          - Launch the FastAPI forensic service on http://127.0.0.1:8000"
	@echo "  make web          - Launch the Vite React development server on http://127.0.0.1:3000"
	@echo "  make web-build    - Build production React single-page application into frontend/dist"
	@echo "  make app          - Build web assets and launch unified forensic workstation on port 8000"
	@echo "  make docker-up    - Build and launch multi-container stack via Docker Compose"
	@echo "  make docker-down  - Stop Docker Compose services"
	@echo "  make data         - Validate and summarize ASVspoof 2019 / In-the-Wild dataset protocols"
	@echo "  make manifests    - Prepare and serialize standardized Parquet manifests (AF-Split v1)"
	@echo "  make audit        - Run zero-leakage speaker, file ID, and audio content hash audit"
	@echo "  make features     - Extract and cache LFCC and Log-Mel features"
	@echo "  make baselines    - Train and evaluate classical baselines (LR, RBF-SVM, RF, GMM)"
	@echo "  make train        - Train CRNN architecture on ASVspoof 2019 LA"
	@echo "  make evaluate     - Run complete evaluation suite on official eval split"
	@echo "  make robustness   - Run codec compression stress tests (MP3, AAC, Opus)"
	@echo "  make crossdomain  - Run zero-shot evaluation on In-the-Wild dataset"
	@echo "  make errors       - Run detailed error analysis and breakdown by attack/speaker"
	@echo "  make report       - Generate automated markdown tables and figures for reports"
	@echo "  make demo         - Launch the legacy Streamlit dashboard"
	@echo "  make clean        - Remove caches, temporary files, and test artifacts"

export PYTHONPATH := src:$(PYTHONPATH)

setup:
	@echo "Setting up Python virtual environment..."
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Installing frontend dependencies..."
	cd frontend && npm install --legacy-peer-deps
	@echo "Setup complete. Activate with: source .venv/bin/activate"

data:
	$(PYTHON) scripts/prepare_data.py --config configs/default.yaml

manifests:
	$(PYTHON) scripts/prepare_manifests.py --config configs/default.yaml

audit:
	$(PYTHON) scripts/audit_leakage.py --manifests-dir data/manifests

smoke:
	@echo "Running quick end-to-end smoke test pipeline..."
	$(PYTHON) scripts/prepare_data.py --config configs/default.yaml --generate-smoke-data
	$(PYTHON) scripts/prepare_manifests.py --debug
	$(PYTHON) scripts/audit_leakage.py --manifests-dir data/manifests
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

reproduce:
	@echo "Running end-to-end reproducible evaluation pipeline..."
	$(PYTHON) scripts/audit_leakage.py --manifests-dir data/manifests
	$(PYTHON) scripts/cache_lfcc_features.py
	$(PYTHON) scripts/cache_logmel_features.py
	$(PYTHON) scripts/train_gmm_baseline.py
	$(PYTHON) scripts/train_classical_sweep.py
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/train_crnn_mps.py
	$(PYTHON) scripts/fit_stacker.py
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/run_stage5_eval_and_audits.py
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/run_stage6_robustness_and_generalization.py
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/benchmark_latency.py
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/export_and_quantize.py
	$(PYTHON) scripts/load_test_api.py
	$(PYTHON) scripts/build_report_tables.py --results-dir results/
	@echo "Complete reproduction pipeline executed successfully."

api:
	@echo "Starting FastAPI backend service on http://127.0.0.1:8000..."
	$(PYTHON) -m uvicorn acforensics.api.main:app --host 127.0.0.1 --port 8000 --reload

web:
	@echo "Starting Vite development server on http://127.0.0.1:3000..."
	cd frontend && npm run dev -- --host 127.0.0.1 --port 3000

web-build:
	@echo "Compiling React production build into frontend/dist..."
	cd frontend && npm run build

app: web-build
	@echo "Starting unified Forensic Instrument on http://127.0.0.1:8000..."
	$(PYTHON) -m uvicorn acforensics.api.main:app --host 127.0.0.1 --port 8000

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

demo: app

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ data/temp_uploads/ data/temp_stress/
