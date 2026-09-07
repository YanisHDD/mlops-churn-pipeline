PY = python
VENV = .venv
EXP = churn-exp

.PHONY: help init data baseline train evaluate test lint format run-api mlflow-ui verify-mlflow pipeline docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make init          - Create virtual environment and install dependencies"
	@echo "  make data          - Download the Telco Churn raw dataset"
	@echo "  make baseline      - Run baseline model verification without tuning"
	@echo "  make train         - Train model with GridSearchCV and log to MLflow"
	@echo "  make evaluate      - Evaluate model from Registry and generate plot artifacts"
	@echo "  make test          - Run pytest unit tests"
	@echo "  make lint          - Run ruff linter"
	@echo "  make format        - Format code using ruff"
	@echo "  make run-api       - Run FastAPI server locally on port 8000"
	@echo "  make mlflow-ui     - Run MLflow UI connected to sqlite:///mlflow.db"
	@echo "  make verify-mlflow - Verify MLflow server REST API headless"
	@echo "  make pipeline      - Run full pipeline: data -> baseline -> train -> evaluate -> test"
	@echo "  make docker-build  - Build Docker container image"
	@echo "  make docker-run    - Run containerized microservice"

init:
	$(PY) -m venv $(VENV) && . $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt

data:
	$(PY) download_data.py

baseline:
	$(PY) -m src.baseline --config configs/config.yaml

train:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) -m src.train --config configs/config.yaml

evaluate:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) -m src.evaluate --config configs/config.yaml

test:
	pytest -q

lint:
	ruff check src tests app.py scripts

format:
	ruff format src tests app.py scripts

run-api:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

verify-mlflow:
	$(PY) scripts/verify_mlflow_ui.py sqlite:///mlflow.db $(EXP) ChurnClassifier

pipeline: data baseline train evaluate test verify-mlflow

docker-build:
	docker build -t churn-classifier:latest .

docker-run:
	docker run -p 8000:8000 --name churn-api churn-classifier:latest

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache mlruns mlflow.db artifacts/*.png artifacts/predictions.csv

