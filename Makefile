PY = python
VENV = .venv
EXP = churn-prediction

.PHONY: help init data train evaluate test lint format run-api run-ui docker-build docker-run pipeline clean

help:
	@echo "Available commands:"
	@echo "  make init         - Create virtual environment and install dependencies"
	@echo "  make data         - Download the Telco Churn raw dataset"
	@echo "  make train        - Train model with GridSearchCV and log to MLflow"
	@echo "  make evaluate     - Evaluate model on test set and generate plot artifacts"
	@echo "  make test         - Run pytest unit tests"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Format code using ruff"
	@echo "  make run-api      - Run FastAPI server locally on port 8000"
	@echo "  make run-ui       - Run MLflow UI on http://127.0.0.1:5000"
	@echo "  make pipeline     - Run full pipeline: data -> train -> evaluate -> test"
	@echo "  make docker-build - Build Docker container image"
	@echo "  make docker-run   - Run containerized microservice"

init:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate || $(VENV)\Scripts\activate && pip install -U pip && pip install -r requirements.txt

data:
	$(PY) download_data.py

train:
	$(PY) src/train.py --config configs/config.yaml

evaluate:
	$(PY) src/evaluate.py --config configs/config.yaml

test:
	pytest -v

lint:
	ruff check .

format:
	ruff format .

run-api:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	mlflow ui --host 127.0.0.1 --port 5000

pipeline: data train evaluate test

docker-build:
	docker build -t churn-classifier-service:latest .

docker-run:
	docker run -p 8000:8000 --name churn-api churn-classifier-service:latest

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
