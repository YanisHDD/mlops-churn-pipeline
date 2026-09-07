PY_SYS ?= python
ifeq ($(OS),Windows_NT)
PY = .venv/Scripts/python.exe
else
PY = .venv/bin/python
endif

CONFIG ?= configs/config.yaml
EXP ?= churn-exp
MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
export MLFLOW_TRACKING_URI
export MLFLOW_EXPERIMENT_NAME = $(EXP)

.PHONY: help init data baseline train evaluate predict test lint format ui serve docker-build docker-run all clean

help:
	@$(PY_SYS) -c "import re; [print(f'{n:14s} {d}') for n, d in sorted(re.findall(r'^([a-z-]+):[^#]*## *(.*)$$', open('Makefile').read(), re.M))]"

init:
	$(PY_SYS) -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.download_data --config $(CONFIG)

baseline:
	$(PY) -m src.train --config $(CONFIG) --no-tuning

train:
	$(PY) -m src.train --config $(CONFIG)

evaluate:
	$(PY) -m src.evaluate --config $(CONFIG)

predict:
	$(PY) -m src.predict --config $(CONFIG) --input data/raw.csv --output reports/predictions_batch.csv

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

ui:
	$(PY) -m mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000

serve:
	$(PY) -m uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t churn-api:latest -f service/Dockerfile .

docker-run:
	docker run --rm -p 8000:8000 churn-api:latest

all: data train evaluate test

clean:
	$(PY_SYS) -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in ['reports', '.pytest_cache', '.ruff_cache'] + glob.glob('**/__pycache__', recursive=True)]"
